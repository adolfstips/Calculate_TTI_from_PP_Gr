import sys
import os
import yaml
import time
import numpy as np
import pandas as pd
import xarray as xr
import torch
import matplotlib.pyplot as plt

def main():
    # --- 0. SETUP & LIBRARIES ---
    config_path = sys.argv[1] if len(sys.argv) > 1 else "input/config_baltic_monthly.yaml"
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device_str = "cuda" if torch.cuda.is_available() and cfg['compute']['device'] == "cuda" else "cpu"
    device = torch.device(device_str)
    print(f"Initialization Complete. Compute Device: {device_str.upper()}")

    if device_str == "cuda":
        torch.cuda.empty_cache()

    # Derived Parameters
    corr_length = int(round(cfg['parameters']['window_years'] * cfg['data']['timesteps_per_year']))
    max_lag = cfg['parameters']['max_lag_steps']
    min_obs = cfg['preprocessing']['min_obs']
    eps_low = float(cfg['compute']['eps_low'])

    # --- 1. INGESTION & SUBSETTING ---
    print(f"Step 1: Loading Data ({cfg['region']['name']})...")
    ds = xr.open_dataset(cfg['data']['file_path'])
    
    # Temporal Subsetting
    start_yr = str(cfg['data']['start_year'])
    end_yr = str(cfg['data']['end_year'])
    ds = ds.sel(time=slice(start_yr, end_yr))
    
    if len(ds.time) == 0:
        raise ValueError("No data found for specified year range.")

    actual_start_year = int(ds['time.year'].min())
    actual_end_year = int(ds['time.year'].max())

    # --- 2. PREPROCESSING & MASKING ---
    print("Step 2: Preprocessing & Boundary Validation...")
    
    pp_da = ds[cfg['data']['var_pp']].load()
    gg_da = ds[cfg['data']['var_gg']].load()

    if cfg['preprocessing'].get('mask_zero_pp', False):
        zero_mask_count = int((pp_da == 0.0).sum())
        if zero_mask_count > 0:
            print(f"WARNING: {zero_mask_count} exact 0.0 boundary values detected. Converting to NA.")
            pp_da = pp_da.where(pp_da != 0.0)
            gg_da = gg_da.where(gg_da != 0.0)

    # Flatten spatial dimensions for matrix processing: Shape (pixels, time)
    pp_stack = pp_da.values.reshape((len(ds.time), -1)).T
    gg_stack = gg_da.values.reshape((len(ds.time), -1)).T

    # Determine valid water pixels based on first timestep
    water_mask = ~np.isnan(pp_stack[:, 0])
    valid_cells = np.where(water_mask)[0]
    
    pp_matrix = pp_stack[valid_cells, :]
    gg_matrix = gg_stack[valid_cells, :]
    print(f"Extracted {len(valid_cells)} valid water pixels across {pp_matrix.shape[1]} timesteps.")

    mask_months = cfg['preprocessing'].get('mask_months', [])
    if mask_months:
        print(f"Applying strict seasonal masking for months: {mask_months}")
        months = ds['time.month'].values
        winter_mask = np.isin(months, mask_months)
        pp_matrix[:, winter_mask] = np.nan
        gg_matrix[:, winter_mask] = np.nan

    # --- 3. GPU CUDA KERNEL ---
    def compute_batch_tti_gpu(v1_batch, v2_batch):
        t_v1 = torch.tensor(v1_batch, device=device, dtype=torch.float32)
        t_v2 = torch.tensor(v2_batch, device=device, dtype=torch.float32)
        
        batch_size, time_steps = t_v1.shape
        
        m1 = ~torch.isnan(t_v1)
        m2 = ~torch.isnan(t_v2)
        t_v1 = torch.nan_to_num(t_v1, nan=0.0)
        t_v2 = torch.nan_to_num(t_v2, nan=0.0)
        
        limit_idx = time_steps - max_lag
        if limit_idx < corr_length:
            return np.full((batch_size, 1), np.nan), np.full((batch_size, 1), np.nan)
        
        v1_unf = t_v1[:, :limit_idx].unfold(-1, corr_length, 1)
        m1_unf = m1[:, :limit_idx].type(torch.float32).unfold(-1, corr_length, 1)
        out_width = v1_unf.shape[1]
        
        best_corrs = torch.full((batch_size, out_width), -2.0, device=device)
        best_lags = torch.full((batch_size, out_width), torch.nan, device=device)
        
        # Part A: Pearson Search
        for lag in range(max_lag + 1):
            v2_unf = t_v2[:, lag : limit_idx + lag].unfold(-1, corr_length, 1)
            m2_unf = m2[:, lag : limit_idx + lag].type(torch.float32).unfold(-1, corr_length, 1)
            
            mask_comb = m1_unf * m2_unf 
            n = mask_comb.sum(dim=-1)
            
            x = v1_unf * mask_comb
            y = v2_unf * mask_comb
            
            sum_x = x.sum(dim=-1); sum_y = y.sum(dim=-1)
            num = n * (x * y).sum(dim=-1) - sum_x * sum_y
            var_x = n * (x * x).sum(dim=-1) - sum_x * sum_x
            var_y = n * (y * y).sum(dim=-1) - sum_y * sum_y
            
            valid = (var_x > eps_low) & (var_y > eps_low) & (n >= min_obs)
            curr_corr = torch.where(valid, num / torch.sqrt(var_x * var_y), torch.tensor(-2.0, device=device))
            
            is_better = curr_corr > best_corrs
            best_corrs = torch.where(is_better, curr_corr, best_corrs)
            best_lags = torch.where(is_better, torch.tensor(float(lag), device=device), best_lags)
        
        # Part B: Fast O(W log W) Spearman on Best Lag
        spearman_vals = torch.full((batch_size, out_width), -2.0, device=device)
        
        def compute_fast_rank(x_input, mask):
            x_clean = torch.where(mask == 1.0, x_input, torch.tensor(1e12, device=device))
            ranks = x_clean.argsort(dim=-1).argsort(dim=-1).type(torch.float32) + 1.0
            return ranks * mask

        for lag in range(max_lag + 1):
            is_this_lag = (best_lags == lag)
            if int(is_this_lag.sum().item()) == 0:
                continue
            
            v2_curr = t_v2[:, lag : limit_idx + lag].unfold(-1, corr_length, 1)
            m2_curr = m2[:, lag : limit_idx + lag].type(torch.float32).unfold(-1, corr_length, 1)
            
            quotient = v2_curr / v1_unf
            q_mask = m1_unf * m2_curr * torch.isfinite(quotient).type(torch.float32)
            
            rank_v1 = compute_fast_rank(v1_unf, q_mask)
            rank_q  = compute_fast_rank(quotient, q_mask)
            
            n_s = q_mask.sum(dim=-1)
            xr = rank_v1 * q_mask; yr = rank_q * q_mask
            
            sum_xr = xr.sum(dim=-1); sum_yr = yr.sum(dim=-1)
            num_s = n_s * (xr * yr).sum(dim=-1) - sum_xr * sum_yr
            var_xr = n_s * (xr * xr).sum(dim=-1) - sum_xr * sum_xr
            var_yr = n_s * (yr * yr).sum(dim=-1) - sum_yr * sum_yr
            
            valid_s = (var_xr > eps_low) & (var_yr > eps_low) & (n_s >= min_obs)
            curr_s = torch.where(valid_s, num_s / torch.sqrt(var_xr * var_yr), torch.tensor(-2.0, device=device))
            
            spearman_vals = torch.where(is_this_lag, curr_s, spearman_vals)
            
        # Part C: Finalize TTI
        spearman_clean = torch.where(spearman_vals <= -1.9, torch.tensor(-float('inf'), device=device), spearman_vals)
        pearson_clean  = torch.where(best_corrs <= -1.9, torch.tensor(-float('inf'), device=device), best_corrs)
        
        final_tti = torch.maximum(pearson_clean, spearman_clean)
        final_tti = torch.where(torch.isinf(final_tti), torch.tensor(float('nan'), device=device), final_tti)
        best_lags = torch.where(torch.isnan(final_tti), torch.tensor(float('nan'), device=device), best_lags)
        
        return final_tti.cpu().numpy(), best_lags.cpu().numpy()

    # --- 4. PARALLEL EXECUTION ---
    print("Step 3: Initiating Tensor Processing...")
    total_pixels = pp_matrix.shape[0]
    time_steps = pp_matrix.shape[1]
    batch_size = cfg['compute']['batch_size_long_ts'] if time_steps > 600 else cfg['compute']['batch_size_short_ts']
    num_batches = int(np.ceil(total_pixels / batch_size))
    
    final_tti_mat = np.full((total_pixels, time_steps), np.nan)
    final_lag_mat = np.full((total_pixels, time_steps), np.nan)

    start_time = time.time()
    for b in range(num_batches):
        idx_start = b * batch_size
        idx_end = min((b + 1) * batch_size, total_pixels)
        print(f"\rProcessing GPU Batch {b+1}/{num_batches} (Rows {idx_start}-{idx_end})", end="")
        
        tti_batch, lag_batch = compute_batch_tti_gpu(pp_matrix[idx_start:idx_end, :], gg_matrix[idx_start:idx_end, :])
        
        offset_start = corr_length // 2
        col_end = offset_start + tti_batch.shape[1]
        
        final_tti_mat[idx_start:idx_end, offset_start:col_end] = tti_batch
        final_lag_mat[idx_start:idx_end, offset_start:col_end] = lag_batch

        if b % 2 == 0 and device_str == "cuda":
            torch.cuda.empty_cache()
            
    print(f"\nGPU Execution Time: {round((time.time() - start_time) / 60.0, 2)} mins")


    # --- 5. RASTER REASSEMBLY & AREA-WEIGHTED AGGREGATION ---
    print("Step 4: Spatial Reassembly and Area-Weighted Aggregation...")
    
    # Dynamically detect spatial dimension names (GETM uses latc/lonc, others might use lat/lon or latitude/longitude)
    lat_name = next((n for n in ['latc', 'lat', 'latitude', 'y'] if n in ds.dims), 'lat')
    lon_name = next((n for n in ['lonc', 'lon', 'longitude', 'x'] if n in ds.dims), 'lon')

    # Reassemble to (time, lat, lon)
    tti_full_stack = np.full((time_steps, len(ds[lat_name]), len(ds[lon_name])), np.nan)
    lag_full_stack = np.full((time_steps, len(ds[lat_name]), len(ds[lon_name])), np.nan)
    
    # Numpy advanced indexing to map pixels back to 2D space
    lat_indices, lon_indices = np.unravel_index(valid_cells, (len(ds[lat_name]), len(ds[lon_name])))
    tti_full_stack[:, lat_indices, lon_indices] = final_tti_mat.T
    lag_full_stack[:, lat_indices, lon_indices] = final_lag_mat.T

    # Create xarray DataArrays
    da_tti = xr.DataArray(tti_full_stack, coords=[ds.time, ds[lat_name], ds[lon_name]], dims=['time', lat_name, lon_name], name="TTI")
    da_lag = xr.DataArray(lag_full_stack, coords=[ds.time, ds[lat_name], ds[lon_name]], dims=['time', lat_name, lon_name], name="LAG")

    # Annual Temporal Aggregation
    if cfg['parameters']['calc_annual'] == "median":
        tti_annual = da_tti.groupby('time.year').median(dim='time', skipna=True)
    else:
        # Fisher Mean
        z_tti = np.arctanh(da_tti.clip(-0.9999, 0.9999))
        tti_annual = np.tanh(z_tti.groupby('time.year').mean(dim='time', skipna=True))
        
    lag_annual = da_lag.groupby('time.year').mean(dim='time', skipna=True)

    # Spatial Basin Means (Cosine Latitude Weighting)
    weights = np.cos(np.deg2rad(tti_annual[lat_name]))
    weights.name = "weights"
    
    z_annual = np.arctanh(tti_annual.clip(-0.9999, 0.9999))
    basin_z_mean = z_annual.weighted(weights).mean(dim=[lat_name, lon_name], skipna=True)
    basin_tti_mean = np.tanh(basin_z_mean)
    
    basin_df = pd.DataFrame({'Year': basin_tti_mean.year.values, 'TTI': basin_tti_mean.values})


# --- 6. EXPORTING ---
    print("Step 5: Exporting NetCDFs and CSVs natively via xarray...")
    os.makedirs(cfg['output']['output_dir'], exist_ok=True)
    
    base_file = f"{cfg['output']['output_dir']}/TTI_{cfg['region']['name']}_{cfg['region']['model']}_{actual_start_year}_{actual_end_year}"

    if cfg['output'].get('save_annual_nc', False):
        # Convert the 'year' integer coordinate to a CF-compliant 'time' datetime coordinate (centered on July 1st)
        time_coords = pd.to_datetime([f"{int(y)}-07-01" for y in tti_annual['year'].values])
        
        tti_nc = tti_annual.rename({'year': 'time'}).assign_coords(time=time_coords)
        lag_nc = lag_annual.rename({'year': 'time'}).assign_coords(time=time_coords)
        
        tti_nc.attrs['long_name'] = "Annual Trophic Transfer Index"
        lag_nc.attrs['long_name'] = "Annual Dominant Lag"
        
        # Save with 'time' as the unlimited dimension to ensure CDO compatibility
        tti_nc.to_netcdf(f"{base_file}_annual_TTI.nc", unlimited_dims=['time'])
        lag_nc.to_netcdf(f"{base_file}_annual_LAG.nc", unlimited_dims=['time'])

    if cfg['output'].get('save_basin_csv', False):
        basin_df.to_csv(f"{base_file}_basin_mean.csv", index=False)
        
        # Lag Area Frequency calculation
        lag_records = []
        for yr in lag_annual.year.values:
            yr_lag = lag_annual.sel(year=yr).values.flatten()
            yr_lag = yr_lag[~np.isnan(yr_lag)]
            if len(yr_lag) > 0:
                unique, counts = np.unique(np.round(yr_lag), return_counts=True)
                for u, c in zip(unique, counts):
                    lag_records.append({'Year': yr, 'Lag': u, 'AreaCount': c})
                    
        if lag_records:
            lag_df = pd.DataFrame(lag_records)
            lag_df['Percentage'] = lag_df.groupby('Year')['AreaCount'].transform(lambda x: (x / x.sum()) * 100)
            lag_df.to_csv(f"{base_file}_lag_freq.csv", index=False)
            
            

    # --- 7. VISUALIZATION ---
    if cfg['output'].get('plot_timeseries', False):
        # 1. TTI Plot
        plt.figure(figsize=(10, 6))
        plt.plot(basin_df['Year'], basin_df['TTI'], marker='o', color='skyblue', markerfacecolor='darkred', markersize=8, linewidth=2)
        plt.title(f"{cfg['region']['name']} Area-Weighted Mean TTI", fontweight='bold', fontsize=16)
        plt.ylabel("TTI [-]", fontweight='bold')
        plt.xlabel("Year", fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{base_file}_timeseries.png", dpi=300)
        plt.close()

        # 2. Lag Dominance Plot
        if 'lag_df' in locals() and not lag_df.empty:
            plt.figure(figsize=(10, 6))
            for lag_val, grp in lag_df.groupby('Lag'):
                plt.plot(grp['Year'], grp['Percentage'], marker='o', label=str(int(lag_val)), linewidth=2)
            plt.title(f"{cfg['region']['name']} Sea: Dominance of Correlation Lag", fontweight='bold', fontsize=16)
            plt.ylabel("Percentage of Total Area (%)", fontweight='bold')
            plt.xlabel("Year", fontweight='bold')
            plt.ylim(0, 100)
            plt.legend(title="Max Correlation Lag", loc='lower center', ncol=len(lag_df['Lag'].unique()))
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"{base_file}_lag_percent.png", dpi=300)
            plt.close()

    print("Python processing successfully completed.")

if __name__ == "__main__":
    main()
