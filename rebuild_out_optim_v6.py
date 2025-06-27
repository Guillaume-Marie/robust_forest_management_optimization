#!/usr/bin/env python3
"""
rebuild_out_optim_v6.py
=======================

Rewritten for clarity with explicit variable names.
Usage:
  python3 rebuild_out_optim_v6.py <simulation_list.txt> <selection_method> [improvement_threshold]

Selection methods: classic, maximin, safetyfirst, classic_threshold
"""
import os
import re
import sys
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rioxarray as rxr

# --------------------------------------------------------------------------------
# Configuration of variables to include in cost calculation
# Each entry defines:
#   - type: 'benefit' or 'cost'
#   - weight: numeric weight in the cost function
#   - agg: 'sum' or 'mean' aggregation over time
#   - year_min/year_max: temporal bounds for aggregation
#   - pattern: identifies the source file pattern ('stomate' or 'sechiba')
# Variables will be normalized to [0..1] before combining.
# --------------------------------------------------------------------------------
VARIABLES_CONFIG = {
    # example:
     'HARVEST_FOREST_c': {
         'type': 'benefit',
         'weight': 1.0,
         'agg': 'sum',
         'year_min': 2012,
         'year_max': 2099,
         'pattern': 'stomate'
     },
}
BAU_KEYWORD = 'BAU'
BASE_PATH = '/home/mguill/Documents/output_CSF_v2/subset'
FILE_PATTERN_TEMPLATE = '{simulation}_{year}0101_{year}1231_1Y_{pattern}_history.nc'

# Output filenames
OUTPUT_MAP_NC = 'Fout_CSF_selected_robust.tif'
OUTPUT_MAP_PNG_TEMPLATE = 'Fout_CSF_SSPs_{variable}_{method}{threshold}.png'
OUTPUT_ZONES_CSV = 'climate_zones_top3_csf.csv'
SHAPEFILE = 'enz_v8_4326.shp'
SHAPE_ZONE_ID = 'EnZ_name'
SELECTED_VAR_NAME = 'HARVEST_FOREST_c'
# Chargement du masque océan (0 = terre, 1 = océan)
OCEAN_MASK_PATH = "/home/mguill/Documents/robust_forest_management_optimization/mask_océan.tif"
# Chargement robuste du masque océan
ocean_mask = rxr.open_rasterio(OCEAN_MASK_PATH)

# Extraire bande utile
if "band" in ocean_mask.dims:
    ocean_mask = ocean_mask.sel(band=1)

# Renommer dimensions raster en 'lon/lat'
if "x" in ocean_mask.dims and "y" in ocean_mask.dims:
    ocean_mask = ocean_mask.rename({"x": "lon", "y": "lat"})

# Assure CRS correct
ocean_mask = ocean_mask.rio.write_crs("EPSG:4326", inplace=True)

def parse_simulation_name(simulation_entry):
    """
    Split a simulation name into management practice and SSP scenario.
    Returns (management_name, scenario_name).
    """
    match = re.search(r"(.+)(SSP\d+)$", simulation_entry)
    if match:
        return match.group(1), match.group(2)
    return simulation_entry, 'NOSSP'


def get_overall_time_bounds(variable_config):
    """
    Determine global min/max years across all configured variables.
    """
    min_years = [cfg['year_min'] for cfg in variable_config.values()]
    max_years = [cfg['year_max'] for cfg in variable_config.values()]
    return min(min_years), max(max_years)


def load_and_aggregate_variable_data(simulation_name):
    """
    For a given simulation, load and aggregate each variable's NetCDF files
    according to its configuration. Returns an xarray.Dataset of DataArrays.
    """
    global_year_min, global_year_max = get_overall_time_bounds(VARIABLES_CONFIG)
    # Group variables by file pattern
    pattern_to_variables_map = {}
    for var_name, cfg in VARIABLES_CONFIG.items():
        pattern = cfg['pattern']
        pattern_to_variables_map.setdefault(pattern, []).append(var_name)

    aggregated_data_vars = {}
    for pattern, variable_list in pattern_to_variables_map.items():
        # Find all existing files for this simulation and pattern
        file_paths = []
        for year in range(global_year_min, global_year_max + 1):
            filename = FILE_PATTERN_TEMPLATE.format(
                simulation=simulation_name,
                year=year,
                pattern=pattern
            )
            full_path = os.path.join(BASE_PATH, filename)
            if os.path.exists(full_path):
                file_paths.append(full_path)
        if not file_paths:
            continue
        # Load and concatenate datasets along time counter
        #dataset_list = [xr.open_dataset(fp, decode_times=False) for fp in file_paths]
        dataset_list = []
        for fp in file_paths:
          ds = xr.open_dataset(fp, decode_times=False, mask_and_scale=True, decode_cf=True)
        dataset_list.append(ds)

        concatenated_dataset = (
            xr.concat(dataset_list, dim='time_counter')
            if len(dataset_list) > 1 else dataset_list[0]
        )
        for ds in dataset_list:
            ds.close()
        # Extract and aggregate each variable
        for var_name in variable_list:
            if var_name not in concatenated_dataset:
                continue
            data_array = concatenated_dataset[var_name]
            if 'veget' in data_array.dims:
                data_array = data_array.mean(dim='veget')
            cfg = VARIABLES_CONFIG[var_name]
            # Slice time dimension according to config
            start_index = max(0, cfg['year_min'] - global_year_min)
            end_index = min(
                data_array.sizes.get('time_counter', 0) - 1,
                cfg['year_max'] - global_year_min
            )
            if start_index > end_index:
                continue
            sliced_array = data_array.isel(time_counter=slice(start_index, end_index + 1))
            # Aggregate
            if cfg['agg'] == 'sum':
                aggregated_array = sliced_array.sum(dim='time_counter')
            else:
                aggregated_array = sliced_array.mean(dim='time_counter')
            aggregated_data_vars[var_name] = aggregated_array
    if not aggregated_data_vars:
        return None
    return xr.Dataset(aggregated_data_vars)


def compute_cost_values(variable_arrays_map):
    """
    Given a mapping of variable name to its aggregated DataArray,
    compute the combined cost DataArray.
    """
    combined_cost = None
    for var_name, array in variable_arrays_map.items():
        cfg = VARIABLES_CONFIG[var_name]
        if cfg['type'] == 'benefit':
            cost_array = -cfg['weight'] * array
        else:
            cost_array = cfg['weight'] * array
        combined_cost = cost_array if combined_cost is None else combined_cost + cost_array
    return combined_cost


def generate_sorted_scenarios_output(management_cost_map, ocean_mask_2d, output_netcdf='Fout_CSF_sorted_robust.nc'):

    """
    From a dict management_cost_map[management_practice][ssp] of cost DataArrays,
    produce two variables:
      - sorted_scenario_id(csf_rank, lat, lon)
      - sorted_cost(csf_rank, lat, lon)
    Writes a CSV scenario_id_map.csv mapping IDs to names.
    Returns the output Dataset.
    """
    scenario_identifiers = []
    cost_arrays = []
    for management_name, ssp_map in management_cost_map.items():
        for ssp, cost_array in ssp_map.items():
            scenario_identifiers.append(f"{management_name}_{ssp}")
            cost_arrays.append(cost_array)
    if not cost_arrays:
        return None
    # Concatenate along new 'scenario' dimension
    cost_data_array_3d = xr.concat(cost_arrays, dim='scenario')
    cost_data_array_3d = cost_data_array_3d.assign_coords(
        scenario=('scenario', scenario_identifiers)
    ).transpose('scenario', 'lat', 'lon')
    n_scenarios = len(scenario_identifiers)
    lat_vals = cost_data_array_3d.coords['lat'].values
    lon_vals = cost_data_array_3d.coords['lon'].values
    # Ocean mask: all NaN across scenarios
    ocean_pixel_mask = ocean_mask_2d
    cost_np = cost_data_array_3d.values
    sorted_indices_ascending = np.argsort(cost_np, axis=0)
    sorted_cost_values_np = np.take_along_axis(cost_np, sorted_indices_ascending, axis=0)
    scenario_ids_array = np.arange(n_scenarios, dtype=int) + 1
    sorted_scenario_ids_np = scenario_ids_array[sorted_indices_ascending]
    # Wrap into DataArrays
    cost_sorted_data_array = xr.DataArray(
        sorted_cost_values_np,
        dims=('csf_rank', 'lat', 'lon'),
        coords={
            'csf_rank': np.arange(n_scenarios),
            'lat': lat_vals,
            'lon': lon_vals
        }
    )
    sorted_scenario_id_array = xr.DataArray(
        sorted_scenario_ids_np,
        dims=('csf_rank', 'lat', 'lon'),
        coords={
            'csf_rank': np.arange(n_scenarios),
            'lat': lat_vals,
            'lon': lon_vals
        }
    )
    # Apply ocean mask: scenario_id=0, cost=NaN
    ocean_mask_3d = ocean_pixel_mask.broadcast_like(sorted_scenario_id_array.isel(csf_rank=0))
    ocean_mask_3d = ocean_mask_3d.expand_dims(
        dim={'csf_rank': np.arange(n_scenarios)}, axis=0
    )
    sorted_scenario_id_array = sorted_scenario_id_array.where(~ocean_mask_3d, 0)
    cost_sorted_data_array = cost_sorted_data_array.where(~ocean_mask_3d, np.nan)
    output_dataset = xr.Dataset({
        'CSF_sorted_id': sorted_scenario_id_array,
        'Cost_sorted': cost_sorted_data_array
    })
    # Write scenario map CSV
    scenario_mapping_csv_filename = 'scenario_id_map.csv'
    with open(scenario_mapping_csv_filename, 'w', encoding='utf-8') as csv_file:
        csv_file.write('scenario_id,scenario_name\n')
        csv_file.write('0,OCEAN\n')
        for idx, name in enumerate(scenario_identifiers, start=1):
            csv_file.write(f"{idx},{name}\n")
    output_dataset = output_dataset.sortby('lat', ascending=True)
    output_dataset.to_netcdf(output_netcdf)
    print(f"=> Sorted scenarios NetCDF written to {output_netcdf}")
    return output_dataset


def create_selected_output(
    simulation_list,
    selection_method='classic',
    threshold_improvement=0.0
):
    """
    Load, normalize, compute cost, and select best practice per pixel.
    Returns map of management names to integer IDs.
    """
    management_data_map = {}
    global_variable_mins = {}
    global_variable_maxs = {}
    # (1) Load and accumulate min/max for normalization
    for sim in simulation_list:
        management_name, ssp = parse_simulation_name(sim)
        ds_agg = load_and_aggregate_variable_data(sim)
        if ds_agg is None:
            continue
        management_data_map.setdefault(management_name, {})[ssp] = ds_agg
        for variable_name in ds_agg.data_vars:
            vmin = ds_agg[variable_name].min().values
            vmax = ds_agg[variable_name].max().values
            if variable_name not in global_variable_mins:
                global_variable_mins[variable_name] = vmin
                global_variable_maxs[variable_name] = vmax
            else:
                global_variable_mins[variable_name] = min(global_variable_mins[variable_name], vmin)
                global_variable_maxs[variable_name] = max(global_variable_maxs[variable_name], vmax)
    if not management_data_map:
        print("No valid simulations found. Exiting.")
        return {}
    # (2) Normalize all variables to [0..1]
    for management_name, ssp_map in management_data_map.items():
        for ssp, ds_tmp in ssp_map.items():
            for variable_name in ds_tmp.data_vars:
                vmin = global_variable_mins[variable_name]
                vmax = global_variable_maxs[variable_name]
                if np.isclose(vmin, vmax):
                    ds_tmp[variable_name] = 0.0
                else:
                    ds_tmp[variable_name] = (ds_tmp[variable_name] - vmin) / (vmax - vmin)
            management_data_map[management_name][ssp] = ds_tmp

    # (3) Compute cost maps
    for management_name, ssp_map in management_data_map.items():
        for ssp, ds_tmp in ssp_map.items():
            variable_arrays_map = {v: ds_tmp[v] for v in ds_tmp.data_vars}
            cost_array = compute_cost_values(variable_arrays_map)
            management_data_map[management_name][ssp] = cost_array
                                 
    if pixel_debug:
        latitudes = next(iter(management_data_map.values())).values()
        sample_array = next(iter(next(iter(management_data_map.values())).values()))
        lat_vals = sample_array.coords["lat"].values
        lon_vals = sample_array.coords["lon"].values

        if pixel_debug == "random":
            lat_idx = np.random.randint(0, len(lat_vals))
            lon_idx = np.random.randint(0, len(lon_vals))
        else:
            lat_target, lon_target = pixel_debug
            lat_idx = np.argmin(np.abs(lat_vals - lat_target))
            lon_idx = np.argmin(np.abs(lon_vals - lon_target))

        print("\n🔍 DEBUG POUR LE PIXEL :")
        print(f"   ➤ Latitude: {lat_vals[lat_idx]:.2f}")
        print(f"   ➤ Longitude: {lon_vals[lon_idx]:.2f}\n")

        for mgmt in sorted(management_data_map.keys()):
            print(f"\n🪵 Management: {mgmt}")
            for ssp in sorted(management_data_map[mgmt].keys()):
                ds = management_data_map[mgmt][ssp]
                for var in sorted(ds.data_vars):
                        raw_val = ds[var].isel(lat=lat_idx, lon=lon_idx).values.item()
                        vmin = global_variable_mins[var]
                        vmax = global_variable_maxs[var]
                        if np.isclose(vmin, vmax):
                            norm_val = 0.0
                        else:
                            norm_val = (raw_val - vmin) / (vmax - vmin)
                        print(f"      - {var:<20s} = {raw_val:>10.4f} (norm: {norm_val:.4f})")


    # (4) Generate sorted scenarios NetCDF and scenario map       
    # Convertir le masque océan en booléen 2D (True = océan)
    ocean_mask_2d = ocean_mask == 1
    print(ocean_mask_2d)
    _ = generate_sorted_scenarios_output(management_data_map, ocean_mask_2d)
    # (5) Select best scenario per pixel
    sample_management = next(iter(management_data_map))
    sample_ssp = next(iter(management_data_map[sample_management]))
    reference_cost_array = management_data_map[sample_management][sample_ssp]
    latitudes = reference_cost_array.coords['lat']
    longitudes = reference_cost_array.coords['lon']
    selected_id_array = xr.DataArray(
        np.zeros(reference_cost_array.shape, dtype=int),
        coords={'lat': latitudes, 'lon': longitudes}, dims=['lat', 'lon']
    )
    final_cost_array = xr.full_like(selected_id_array, np.inf, dtype=float)
    management_names_list = sorted(management_data_map.keys())
    management_name_to_id_map = {m: i+1 for i, m in enumerate(management_names_list)}
    bau_candidates = [m for m in management_names_list if BAU_KEYWORD in m.upper()]
    bau_management_name = bau_candidates[0] if bau_candidates else None

    for lat_idx in range(len(latitudes)):
        for lon_idx in range(len(longitudes)):
            if ocean_mask_2d.isel(lat=lat_idx, lon=lon_idx):
                selected_id_array[lat_idx, lon_idx] = 0
                final_cost_array[lat_idx, lon_idx] = np.inf
                continue
            best_management = None
            best_value = np.inf
            for management_name in management_names_list:
                scenario_values_list = [
                    management_data_map[management_name][ssp].isel(lat=lat_idx, lon=lon_idx).values
                    for ssp in management_data_map[management_name]
                ]
                scenario_values = np.array(scenario_values_list)
                if np.any(np.isnan(scenario_values)):
                    continue
                if selection_method == 'maximin':
                    worst_case = scenario_values.max()
                    if worst_case < best_value:
                        best_value = worst_case
                        best_management = management_name
                elif selection_method == 'safetyfirst' and bau_management_name:
                    worst_case = scenario_values.max()
                    bau_stack = xr.concat(
                        [management_data_map[bau_management_name][ssp] for ssp in management_data_map[bau_management_name]],
                        dim='ssp'
                    )
                    bau_worst = bau_stack.isel(lat=lat_idx, lon=lon_idx).max().values
                    if worst_case <= bau_worst:
                        average_value = scenario_values.mean()
                        if average_value < best_value:
                            best_value = average_value
                            best_management = management_name
                else:
                    average_value = scenario_values.mean()
                    if average_value < best_value:
                        best_value = average_value
                        best_management = management_name
            # classic_threshold post-check
            if selection_method == 'classic_threshold' and bau_management_name and best_management:
                bau_stack = xr.concat(
                    [management_data_map[bau_management_name][ssp] for ssp in management_data_map[bau_management_name]],
                    dim='ssp'
                )
                bau_average = bau_stack.isel(lat=lat_idx, lon=lon_idx).mean().values
                if bau_average != 0:
                    improvement_gain = (bau_average - best_value) / abs(bau_average)
                else:
                    improvement_gain = 0
                if improvement_gain < threshold_improvement:
                    best_management = bau_management_name
                    best_value = bau_average
            if best_management:
                selected_id_array[lat_idx, lon_idx] = management_name_to_id_map[best_management]
                final_cost_array[lat_idx, lon_idx] = best_value
            else:
                selected_id_array[lat_idx, lon_idx] = 0
                final_cost_array[lat_idx, lon_idx] = np.inf
    final_dataset = xr.Dataset({
        'CSF_selected': selected_id_array,
        'Cost_selected': final_cost_array
    })
    # Write flat CSV of selections
    flat_df = final_dataset.to_dataframe().reset_index()

    # Supprimer les pixels océans (CSF_selected = 0)
    flat_df = flat_df[flat_df['CSF_selected'] != 0]

    # Remplacer les valeurs manquantes par NaN (pour propreté)
    flat_df['CSF_selected'] = flat_df['CSF_selected'].replace({0: pd.NA})

    id_name_df = pd.DataFrame([
        {'CSF_selected': id_val, 'scenario_name': name}
        for name, id_val in management_name_to_id_map.items()
    ])
    merged_df = flat_df.merge(id_name_df, on='CSF_selected', how='left')
    output_csv_filename = f"Fout_CSF_selected_{SELECTED_VAR_NAME}_{selection_method}.csv"
    merged_df.to_csv(output_csv_filename, index=False)
    print(f"→ Selection CSV written to {output_csv_filename}")
    return management_name_to_id_map


def invert_management_id_map(management_id_map):
    """Invert mapping from management name to ID."""
    return {v: k for k, v in management_id_map.items()}


def visualize_scenario_map_from_csv(csv_file, png_file):
    """
    Plot selected management scenario map from a flat CSV.
    """
    df = pd.read_csv(csv_file)
    if not {'lat', 'lon', 'CSF_selected', 'scenario_name'}.issubset(df.columns):
        print("CSV missing required columns.")
        return
    df['CSF_selected'] = df['CSF_selected'].fillna(0).astype(int)
    latitudes = sorted(df['lat'].unique())
    longitudes = sorted(df['lon'].unique())
    grid = np.full((len(latitudes), len(longitudes)), np.nan)
    lat_index = {lat: i for i, lat in enumerate(latitudes)}
    lon_index = {lon: j for j, lon in enumerate(longitudes)}
    for _, row in df.iterrows():
        i = lat_index[row['lat']]
        j = lon_index[row['lon']]
        if row['CSF_selected'] > 0:
            grid[i, j] = row['CSF_selected']
    id_to_name = df.dropna(subset=['CSF_selected']).drop_duplicates('CSF_selected').set_index('CSF_selected')['scenario_name'].to_dict()
    used_ids = sorted(id_to_name.keys())
    cmap = plt.get_cmap('Spectral', len(used_ids))
    cmap.set_bad('white')
    norm = mcolors.BoundaryNorm(np.arange(0.5, len(used_ids)+1.5), len(used_ids))
    fig, ax = plt.subplots(figsize=(10, 7))
    img = ax.pcolormesh(longitudes, latitudes, grid, cmap=cmap, norm=norm, shading='auto')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Selected Forest Management Practice')
    cbar = fig.colorbar(img, ax=ax, ticks=used_ids)
    cbar.ax.set_yticklabels([id_to_name[i] for i in used_ids])
    plt.tight_layout()
    plt.savefig(png_file, dpi=150)
    plt.close(fig)
    print(f"→ Map image saved to {png_file}")


def perform_zonal_statistics(
    shapefile_path,
    netcdf_file,
    variable_name,
    id_to_name_map,
    zone_id_field=SHAPE_ZONE_ID,
    output_csv=OUTPUT_ZONES_CSV
):
    """
    Compute zonal statistics of selected scenario map over climate zones.
    """
    if not os.path.exists(netcdf_file):
        return
    ds = xr.open_dataset(netcdf_file)
    if variable_name not in ds:
        ds.close()
        return
    da = ds[variable_name]
    ds.close()
    da = da.rio.write_crs('EPSG:4326', inplace=True)
    tmp_raster = 'CSF_selected_map_tmp.tif'
    da.rio.to_raster(tmp_raster)
    gdf = gpd.read_file(shapefile_path)
    stats = zonal_stats(
        gdf,
        tmp_raster,
        categorical=True,
        nodata=np.nan
    )
    results = []
    for row, stat in zip(gdf.itertuples(), stats):
        zone = getattr(row, zone_id_field)
        if stat is None:
            results.append({
                'zone_id': zone,
                'top1': None, 'count1': 0,
                'top2': None, 'count2': 0,
                'top3': None, 'count3': 0
            })
            continue
        items = [(k, v) for k, v in stat.items() if k != 0]
        items.sort(key=lambda x: x[1], reverse=True)
        named = [(id_to_name_map.get(k, f"ID_{k}"), v) for k, v in items]
        while len(named) < 3:
            named.append((None, 0))
        results.append({
            'zone_id': zone,
            'top1': named[0][0], 'count1': named[0][1],
            'top2': named[1][0], 'count2': named[1][1],
            'top3': named[2][0], 'count3': named[2][1]
        })
    df_out = pd.DataFrame(results)
    df_out['pct1'] = df_out['count1'] / (df_out['count1'] + df_out['count2'] + df_out['count3']) * 100
    df_out['pct2'] = df_out['count2'] / (df_out['count1'] + df_out['count2'] + df_out['count3']) * 100
    df_out['pct3'] = df_out['count3'] / (df_out['count1'] + df_out['count2'] + df_out['count3']) * 100
    df_out.to_csv(output_csv, index=False)
    print(f"=> Zonal statistics written to {output_csv}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python rebuild_out_optim_v6.py <simulation_list.txt> <selection_method> [improvement_threshold]")
        sys.exit(1)
    simulation_list_file = sys.argv[1]
    selection_method = sys.argv[2].lower()
    threshold_improvement = 0.0

    import random
    pixel_debug = None
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--pixel="):
            val = arg.split("=")[1]
            lat_str, lon_str = val.split(",")
            pixel_debug = (float(lat_str), float(lon_str))
        elif arg == "--pixel":
            pixel_debug = "random"

    if selection_method == 'classic_threshold' and len(sys.argv) >= 4:
        threshold_improvement = float(sys.argv[3])
    if not os.path.exists(simulation_list_file):
        print(f"Simulation list file not found: {simulation_list_file}")
        sys.exit(1)
    with open(simulation_list_file, 'r') as f:
        simulation_entries = [line.strip() for line in f if line.strip()]
    management_id_map = create_selected_output(
        simulation_entries,
        selection_method=selection_method,
        threshold_improvement=threshold_improvement
    )
    if not management_id_map:
        sys.exit(0)
    invert_map = invert_management_id_map(management_id_map)
    csv_name = f"Fout_CSF_selected_{SELECTED_VAR_NAME}_{selection_method}.csv"
    png_name = OUTPUT_MAP_PNG_TEMPLATE.format(
        variable=SELECTED_VAR_NAME,
        method=selection_method,
        threshold=str(threshold_improvement) if threshold_improvement else ''
    )
    visualize_scenario_map_from_csv(csv_name, png_name)
    perform_zonal_statistics(
        SHAPEFILE,
        OUTPUT_MAP_NC,
        SELECTED_VAR_NAME,
        invert_map
    )
    print("Finished.")
