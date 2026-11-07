# --- 0. SETUP & LIBRARIES ---
library(terra)
library(zoo)
library(dplyr)
library(lubridate)
library(ggplot2)
library(ggpubr)
library(yaml)
library(stringr)
library(torch)

# Load configuration (pass YAML path as argument or default to Baltic)
args <- commandArgs(trailingOnly = TRUE)
config_path <- if (length(args) > 0) args[1] else "./input/config_baltic_monthly.yaml"
###config_path  <- "./input/config_swes_monthly.yaml"
cfg <- yaml::read_yaml(config_path)

# Initialize CUDA
device <- ifelse(cuda_is_available() && cfg$compute$device == "cuda", "cuda", "cpu")
torch_device(device)
message("Initialization Complete. Compute Device: ", device)

if (cuda_is_available()) {
  gc()
  cuda_empty_cache()
}

# Derived Parameters
CORR_LENGTH <- round(cfg$parameters$window_years * cfg$data$timesteps_per_year)
NYEARS <- cfg$data$end_year - cfg$data$start_year + 1
mask_str <- if (length(cfg$preprocessing$mask_months) > 0) paste0("_Mask", paste(cfg$preprocessing$mask_months, collapse="")) else "_NoMask"
run_date <- format(Sys.time(), "%Y-%m-%d")

# --- 1. INGESTION & SUBSETTING ---
message("Step 1: Loading Data (", cfg$region$name, ")...")
if (!file.exists(cfg$data$file_path)) stop("Input file not found: ", cfg$data$file_path)

PP_rast_full <- rast(cfg$data$file_path, subds = cfg$data$var_pp)
GG_rast_full <- rast(cfg$data$file_path, subds = cfg$data$var_gg)

# Temporal Subsetting
all_years <- as.numeric(substr(as.character(terra::time(PP_rast_full)), 1, 4))
idx_keep <- which(all_years >= cfg$data$start_year & all_years <= cfg$data$end_year)
if (length(idx_keep) == 0) stop("No data found for specified year range.")

PP_rast <- PP_rast_full[[idx_keep]]
GG_rast <- GG_rast_full[[idx_keep]]

# --- 2. PREPROCESSING & MASKING ---
message("Step 2: Preprocessing & Boundary Validation...")

if (cfg$preprocessing$mask_zero_pp) {
  zero_mask <- which(terra::values(PP_rast) == 0.0)
  if (length(zero_mask) > 0) {
    message(sprintf("WARNING: %d exact 0.0 boundary values detected. Converting to NA to prevent correlation artifacts.", length(zero_mask)))
    PP_rast[PP_rast == 0.0] <- NA
    GG_rast[GG_rast == 0.0] <- NA
  }
}

water_mask <- !is.na(PP_rast[[1]])
water_cells <- which(values(water_mask) == TRUE)
pp_matrix <- as.matrix(terra::extract(PP_rast, water_cells))
gg_matrix <- as.matrix(terra::extract(GG_rast, water_cells))
message(sprintf("Extracted %d valid water pixels across %d timesteps.", nrow(pp_matrix), ncol(pp_matrix)))

if (length(cfg$preprocessing$mask_months) > 0) {
  message("Applying strict seasonal masking for months: ", paste(cfg$preprocessing$mask_months, collapse=", "))
  dates <- terra::time(PP_rast)
  months <- as.numeric(format(dates, "%m"))
  winter_mask <- months %in% cfg$preprocessing$mask_months
  
  pp_matrix[, winter_mask] <- NA
  gg_matrix[, winter_mask] <- NA
}

# --- 3. GPU CUDA KERNEL ---
compute_batch_tti_gpu <- function(v1_batch, v2_batch, corr_len, max_lag, min_obs) {
  t_v1 <- torch_tensor(v1_batch, device = device, dtype = torch_float32())
  t_v2 <- torch_tensor(v2_batch, device = device, dtype = torch_float32())
  
  batch_size  <- t_v1$size(1)
  time_steps  <- t_v1$size(2)
  
  m1 <- (!torch_isnan(t_v1))$to(dtype = torch_float32())
  m2 <- (!torch_isnan(t_v2))$to(dtype = torch_float32())
  t_v1$masked_fill_(torch_isnan(t_v1), 0.0)
  t_v2$masked_fill_(torch_isnan(t_v2), 0.0)
  
  limit_idx <- time_steps - max_lag
  if (limit_idx < corr_len) return(list(tti = matrix(NA, batch_size, 1), lag = matrix(NA, batch_size, 1)))
  
  v1_unfolded <- t_v1[.., 1:limit_idx]$unfold(-1, corr_len, 1)
  m1_unfolded <- m1[.., 1:limit_idx]$unfold(-1, corr_len, 1)
  out_width <- v1_unfolded$size(2)
  
  best_corrs <- torch_full(c(batch_size, out_width), -2.0, device = device)
  best_lags  <- torch_full(c(batch_size, out_width), NA_real_, device = device)
  
  # Part A: Pearson Search
  for (lag in 0:max_lag) {
    start_col <- 1 + lag
    end_col   <- limit_idx + lag
    
    v2_unfolded <- t_v2[.., start_col:end_col]$unfold(-1, corr_len, 1)
    m2_unfolded <- m2[.., start_col:end_col]$unfold(-1, corr_len, 1)
    
    mask_combined <- m1_unfolded * m2_unfolded 
    n <- mask_combined$sum(dim = -1)
    
    x <- v1_unfolded * mask_combined
    y <- v2_unfolded * mask_combined
    
    sum_x  <- x$sum(dim = -1); sum_y  <- y$sum(dim = -1)
    sum_xy <- (x * y)$sum(dim = -1)
    sum_x2 <- (x * x)$sum(dim = -1); sum_y2 <- (y * y)$sum(dim = -1)
    
    num   <- n * sum_xy - sum_x * sum_y
    var_x <- n * sum_x2 - sum_x * sum_x
    var_y <- n * sum_y2 - sum_y * sum_y
    
    valid_denom <- (var_x > cfg$compute$eps_low) & (var_y > cfg$compute$eps_low) & (n >= min_obs)
    curr_corr <- torch_where(valid_denom, num / torch_sqrt(var_x * var_y), -2.0)
    
    is_better <- curr_corr > best_corrs
    best_corrs <- torch_where(is_better, curr_corr, best_corrs)
    best_lags  <- torch_where(is_better, torch_tensor(lag, device=device, dtype=torch_float32()), best_lags)
  }
  
  # Part B: Fast O(W log W) Spearman on Best Lag
  spearman_vals <- torch_full(c(batch_size, out_width), -2.0, device = device)
  
  compute_fast_rank <- function(x_input, mask) {
    x_clean <- torch_where(mask == 1.0, x_input, 1e12)
    ranks <- x_clean$argsort(dim = -1)$argsort(dim = -1)$to(dtype = torch_float32()) + 1
    return(ranks * mask)
  }
  
  for (lag in 0:max_lag) {
    is_this_lag <- (best_lags == lag)
    if (as.numeric(is_this_lag$sum()$cpu()) == 0) next
    
    start_col <- 1 + lag
    end_col   <- limit_idx + lag
    v2_curr   <- t_v2[.., start_col:end_col]$unfold(-1, corr_len, 1)
    m2_curr   <- m2[.., start_col:end_col]$unfold(-1, corr_len, 1)
    
    quotient <- v2_curr / v1_unfolded
    q_isfinite <- torch_isfinite(quotient)
    q_mask <- m1_unfolded * m2_curr * q_isfinite$to(dtype = torch_float32())
    
    rank_v1 <- compute_fast_rank(v1_unfolded, q_mask)
    rank_q  <- compute_fast_rank(quotient, q_mask)
    
    n_s <- q_mask$sum(dim = -1)
    xr <- rank_v1 * q_mask; yr <- rank_q  * q_mask
    
    sum_xr <- xr$sum(dim = -1); sum_yr <- yr$sum(dim = -1)
    sum_xyr <- (xr * yr)$sum(dim = -1)
    sum_x2r <- (xr * xr)$sum(dim = -1); sum_y2r <- (yr * yr)$sum(dim = -1)
    
    num_s   <- n_s * sum_xyr - sum_xr * sum_yr
    var_xr  <- n_s * sum_x2r - sum_xr * sum_xr
    var_yr  <- n_s * sum_y2r - sum_yr * sum_yr
    
    valid_s <- (var_xr > cfg$compute$eps_low) & (var_y > cfg$compute$eps_low) & (n_s >= min_obs)
    curr_s  <- torch_where(valid_s, num_s / torch_sqrt(var_xr * var_yr), -2.0)
    
    spearman_vals <- torch_where(is_this_lag, curr_s, spearman_vals)
  }
  
  # Part C: Finalize TTI
  spearman_clean <- torch_where(spearman_vals <= -1.9, -Inf, spearman_vals)
  pearson_clean  <- torch_where(best_corrs <= -1.9, -Inf, best_corrs)
  
  final_tti <- torch_maximum(pearson_clean, spearman_clean)
  final_tti <- torch_where(torch_isinf(final_tti), NaN, final_tti)
  best_lags <- torch_where(torch_isnan(final_tti), NaN, best_lags)
  
  return(list(tti = as.matrix(final_tti$cpu()), lag = as.matrix(best_lags$cpu())))
}

# --- 4. PARALLEL EXECUTION ---
message("Step 3: Initiating Tensor Processing...")
ts_length <- ncol(pp_matrix)
total_pixels <- nrow(pp_matrix)
batch_size <- if (ts_length > 600) cfg$compute$batch_size_long_ts else cfg$compute$batch_size_short_ts
num_batches <- ceiling(total_pixels / batch_size)

final_tti <- matrix(NA, nrow=total_pixels, ncol=ncol(pp_matrix))
final_lag <- matrix(NA, nrow=total_pixels, ncol=ncol(pp_matrix))

start_time <- Sys.time()
for (b in 1:num_batches) {
  idx_start <- (b - 1) * batch_size + 1
  idx_end   <- min(b * batch_size, total_pixels)
  cat(sprintf("\rProcessing GPU Batch %d/%d (Rows %d-%d)", b, num_batches, idx_start, idx_end))
  
  res <- compute_batch_tti_gpu(
    pp_matrix[idx_start:idx_end, , drop=FALSE], 
    gg_matrix[idx_start:idx_end, , drop=FALSE], 
    CORR_LENGTH, cfg$parameters$max_lag_steps, cfg$preprocessing$min_obs
  )
  
  offset_start <- 1 + floor((CORR_LENGTH - 1) / 2)
  col_idx <- offset_start:(offset_start + ncol(res$tti) - 1)
  
  final_tti[idx_start:idx_end, col_idx] <- res$tti
  final_lag[idx_start:idx_end, col_idx] <- res$lag
  
  if (b %% 2 == 0 && cuda_is_available()) { gc(); cuda_empty_cache() }
}
cat("\nGPU Execution Time: ", round(difftime(Sys.time(), start_time, units="mins"), 2), " mins\n")

# --- 5. RASTER REASSEMBLY & AREA-WEIGHTED AGGREGATION ---
message("Step 4: Spatial Reassembly and Area-Weighted Aggregation...")

template_raster <- PP_rast[[1]]
values(template_raster) <- NA

tti_raw <- rast(template_raster, nlyr = ncol(final_tti))
tti_raw[water_cells] <- final_tti
terra::time(tti_raw) <- terra::time(PP_rast)

lag_raw <- rast(template_raster, nlyr = ncol(final_lag))
lag_raw[water_cells] <- final_lag
terra::time(lag_raw) <- terra::time(PP_rast)

# Annual Raster Aggregation
tti_annual <- terra::tapp(tti_raw, "years", fun = if (cfg$parameters$calc_annual == "median") median else mean, na.rm = TRUE)
lag_annual <- terra::tapp(lag_raw, "years", fun = "mean", na.rm = TRUE)
names(tti_annual) <- substr(as.character(terra::time(tti_annual)), 1, 4)
names(lag_annual) <- substr(as.character(terra::time(lag_annual)), 1, 4)

# Calculate Spatial Basin Means (Area Weighted Fisher Z)
cell_areas <- terra::cellSize(template_raster, unit = "km")

compute_basin_fisher <- function(r_stack, area_grid) {
  z_stack <- atanh(terra::clamp(r_stack, lower=-0.9999, upper=0.9999))
  # Use exact cell area weighting for the global mean
  weighted_z <- terra::global(z_stack, fun = "mean", weights = area_grid, na.rm = TRUE)
  
  # Robustly extract the first column regardless of terra's internal naming
  return(tanh(weighted_z[[1]])) 
}
basin_tti_mean <- compute_basin_fisher(tti_annual, cell_areas)
basin_df <- data.frame(Year = as.numeric(names(tti_annual)), TTI = basin_tti_mean)


# --- 6. EXPORTING ---
message("Step 5: Exporting NetCDFs and CSVs natively via Terra...")
dir.create(cfg$output$output_dir, showWarnings = FALSE, recursive = TRUE)

# Dynamically extract actual processed years from the subset data
actual_start_year <- min(all_years[idx_keep])
actual_end_year <- max(all_years[idx_keep])

base_filename <- sprintf("%s/TTI_%s_%s_%d_%d", 
                         cfg$output$output_dir, 
                         cfg$region$name, 
                         cfg$region$model, 
                         actual_start_year, 
                         actual_end_year)

if (isTRUE(cfg$output$save_annual_nc)) {
  terra::writeCDF(tti_annual, filename = paste0(base_filename, "_annual_TTI.nc"), varname = "TTI", longname = "Annual Trophic Transfer Index", overwrite = TRUE)
  terra::writeCDF(lag_annual, filename = paste0(base_filename, "_annual_LAG.nc"), varname = "LAG", longname = "Annual Dominant Lag", overwrite = TRUE)
}

if (isTRUE(cfg$output$save_basin_csv)) {
  write.csv(basin_df, file = paste0(base_filename, "_basin_mean.csv"), row.names = FALSE)
  
  # Lag Area Frequency calculation
  year_index_df <- data.frame(Year = as.numeric(names(lag_annual)), Layer = 1:nlyr(lag_annual))
  freq_list <- lapply(1:nlyr(lag_annual), function(i) {
    f <- terra::freq(lag_annual[[i]])
    if (nrow(f) > 0) { f$Year <- year_index_df$Year[i]; return(f) }
  })
  lag_df <- do.call(rbind, freq_list) %>% rename(Lag = value) %>% filter(!is.na(Lag)) %>%
    group_by(Year, Lag) %>% summarise(AreaCount = sum(count), .groups = 'drop') %>%
    group_by(Year) %>% mutate(Percentage = AreaCount / sum(AreaCount) * 100) %>%
    ungroup() %>% mutate(Lag = as.factor(Lag))
  
  write.csv(lag_df, file = paste0(base_filename, "_lag_freq.csv"), row.names = FALSE)
}

# --- 7. VISUALIZATION ---
if (isTRUE(cfg$output$plot_timeseries)) {
  p_tti <- ggplot(basin_df, aes(x = Year, y = TTI)) +
    geom_line(linewidth = 2, color = 'skyblue') + geom_point(size = 3, color = 'darkred') +
    labs(title = paste(cfg$region$name, "Area-Weighted Mean TTI"), y = "TTI [-]", x = "Year") +
    theme_pubr(base_size = 14)
  
  ggsave(paste0(base_filename, "_timeseries.png"), plot = p_tti, width = 10, height = 6, dpi = 300)
}
# 2. Lag Dominance Timeseries Plot
if (nrow(lag_df) > 0) {
  p_lag <- ggplot(data = lag_df, aes(x = Year, y = Percentage, color = Lag, group = Lag)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 2.5) +
    labs(
      title = paste(cfg$region$name, "Sea: Dominance of Correlation Lag"),
      subtitle = "Proportion of area where PPR-GR correlation is maximal at time lag",
      x = "Year",
      y = "Percentage of Total Area (%)",
      color = "Max Correlation Lag"
    ) +
    theme_minimal(base_size = 14) +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 20)) +
    # Use a robust color scale depending on the number of possible lags
    scale_color_viridis_d(option = "turbo") + 
    theme(
      legend.position = "bottom",
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 12),
      panel.grid.minor = element_blank()
    )
  
  ggsave(paste0(base_filename, "_lag_percent.png"), plot = p_lag, width = 10, height = 6, dpi = 300)
}

message("Processing successfully completed.")

