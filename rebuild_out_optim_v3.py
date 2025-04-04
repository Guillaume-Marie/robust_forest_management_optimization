#!/usr/bin/env python3
"""
Detailed Description
--------------------
This script identifies a single "best" CSF (forest management scenario) for each pixel 
from a set of NetCDF outputs, using one of several "robust" selection methods. The 
supported selection methods are:

  • classic           : Minimizes the average cost across all SSPs (Shared Socioeconomic Pathways).
  • maximin           : Chooses the scenario that yields the lowest worst-case cost (i.e., the best among the worst outcomes).
  • safetyfirst       : Compares each scenario's worst outcome against the BAU (Business as Usual) worst outcome; 
                        then, among scenarios that do not exceed BAU's worst outcome, selects the one with the best average.
  • classic_threshold : Similar to 'classic', but an additional improvement threshold is required compared to BAU. 
                        If the newly selected scenario fails to improve on BAU by at least that threshold, BAU remains chosen.

Cost function definition:
-------------------------
We define a 'benefit' variable to have negative cost (because higher benefit is better), 
and a 'cost' variable to have positive cost (since higher cost is worse). Each variable is 
weighted in the total cost function. The lower the total "cost", the better the scenario.

Usage:
------
  python rebuild_out_optim_v3.py <simulation_input_list.txt> <robust_method> [improvement_threshold]

Examples:
---------
  1) python rebuild_out_optim_v3.py CSF_experiment_list.txt classic
     -> Uses the classic method to pick the scenario with the lowest average cost (across all SSPs).

  2) python rebuild_out_optim_v3.py CSF_experiment_list.txt maximin
     -> Uses the maximin robust selection approach.

  3) python rebuild_out_optim_v3.py CSF_experiment_list.txt classic_threshold 0.10
     -> Uses the classic threshold approach, requiring a 10% improvement over BAU to switch from BAU.

Outputs:
--------
  • Fout_CSF_selected_robust.nc : The NetCDF file storing the chosen scenario (by integer ID) and its resulting cost.
  • Fout_CSF_selected_robust.png: A quick visualization of the chosen scenario across the grid.
  • climate_zones_top3_csf.csv  : A CSV file with zonal statistics (top-3 scenarios) for each polygon zone in a shapefile.

Implementation Details:
-----------------------
  • The script first reads the list of simulations from <simulation_input_list.txt>.
  • For each simulation, the relevant NetCDF files (annual slices) from YEAR_MIN to YEAR_MAX are opened, 
    concatenated, and aggregated into a single statistic per pixel (usually a mean over time).
  • A combined cost function is computed from the listed variables in VARIABLES_INFO.
  • Scenarios are then compared based on the user-selected robust_method.
  • Finally, the script saves:
      - a NetCDF map of the best scenario ID,
      - a PNG visualization,
      - and zonal statistics in CSV format (top-3 scenarios per region).

"""

import xarray as xr
import numpy as np
import sys
import os
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rioxarray
from rasterstats import zonal_stats
import geopandas as gpd
import pandas as pd

# --------------------------------------------------------------------------------
# Global parameters
# --------------------------------------------------------------------------------

BASE_PATH = "/home/mguill/Documents/output_CSF_v2"
YEAR_MIN = 2089
YEAR_MAX = 2099
FILE_PATTERN = "{simulation}_{year}0101_{year}1231_1Y_stomate_history.nc"


# Part A
VARIABLES_INFO = {
    "TOTAL_SOIL_c": ("benefit", 1.0),
    "NBP_pool_c"   : ("benefit", 1.0),
}
AGGREGATION_MODES = {
    "TOTAL_SOIL_c": "mean",
    "NBP_pool_c"  : "mean",
}
BAU_KEYWORD = "BAU"
OUTPUT_SELECTED_NC = "Fout_CSF_selected_robust.nc"

# Part B
OUTPUT_PNG = "Fout_CSF_selected_robust.png"

# Part C
SHAPEFILE = "enz_v8_4326.shp"
SHAPE_ZONE_ID = "EnZ_name"
VAR_NAME = "CSF_selected"
OUTPUT_TIF = "Fout_CSF_selected_robust.tif"
OUTPUT_TIF_4326 = "Fout_CSF_selected_robust_4326.tif"
OUTPUT_CSV = "climate_zones_top3_csf.csv"

def parse_mgmt_and_ssp(sim_name):
    """
    Extract the management name and SSP suffix from a simulation name.
    For example, if sim_name="BAUSSP126", mgmt_name="BAU" and ssp="SSP126".
    Returns (mgmt_name, ssp). If no SSP is detected, returns "NOSSP" for the second value.
    """
    match = re.search(r"(.+)(SSP\d+)$", sim_name)
    if match:
        mgmt_name = match.group(1)
        ssp = match.group(2)
        return mgmt_name, ssp
    else:
        return sim_name, "NOSSP"


def load_and_aggregate(sim_name):
    """
    Reads the NetCDF files for the specified simulation from YEAR_MIN..YEAR_MAX,
    concatenates them along time, and computes an aggregation (mean or sum in time).
    Returns an xarray.Dataset with the relevant variables if found, or None if no files are found.
    """
    files_found = []
    for yr in range(YEAR_MIN, YEAR_MAX + 1):
        fn = FILE_PATTERN.format(simulation=sim_name, year=yr)
        fp = os.path.join(BASE_PATH, fn)
        if os.path.exists(fp):
            files_found.append(fp)
    if not files_found:
        return None

    ds_list = [xr.open_dataset(f, decode_times=False) for f in files_found]
    if len(ds_list) == 1:
        ds_concat = ds_list[0]
    else:
        ds_concat = xr.concat(ds_list, dim="time_counter")

    for ds in ds_list:
        ds.close()

    data_vars = {}
    for var_name, (var_type, weight) in VARIABLES_INFO.items():
        if var_name in ds_concat:
            arr = ds_concat[var_name]
            # If there's a 'veget' dimension, average across it (simplify).
            if "veget" in arr.dims:
                arr = arr.mean(dim="veget")
            # Sum or mean over the time dimension.
            if AGGREGATION_MODES.get(var_name, "mean") == "sum":
                arr = arr.sum(dim="time_counter")
            else:
                arr = arr.mean(dim="time_counter")
            data_vars[var_name] = arr

    return xr.Dataset(data_vars)


def cost_function(var_dict):
    """
    Combine "benefit" variables (cost = -value) and "cost" variables (cost = +value)
    according to the weights specified in VARIABLES_INFO.
    """
    combined = None
    for var_name, (var_type, weight) in VARIABLES_INFO.items():
        if var_name not in var_dict:
            continue
        arr = var_dict[var_name]
        if var_type == "benefit":
            this_cost = -weight * arr
        else:
            this_cost = weight * arr
        combined = this_cost if (combined is None) else (combined + this_cost)
    return combined


def create_selected_output(sim_list, robust_method="classic", improvement_threshold=0.0):
    """
    Collects all simulations, computes the 'cost_function', and selects
    a scenario per pixel according to 'robust_method'.

    If `robust_method == "classic_threshold"`, we compare to BAU:
        - If the selected scenario does not achieve an improvement 
          >= improvement_threshold * 100% over BAU, we stick to BAU.
        - Otherwise, we adopt the newly selected scenario.

    Returns a dictionary mgmt_id_map = {mgmt_name : int_id}, and writes 
    'Fout_CSF_selected_robust.nc' to disk with the chosen scenario ID and cost.
    """
    # 1) Load cost maps for each (mgmt, ssp).
    mgmt_dict = {}  # mgmt_dict[mgmt_name][ssp] = cost_map (xarray)
    for sim in sim_list:
        mgmt, ssp = parse_mgmt_and_ssp(sim)
        ds_agg = load_and_aggregate(sim)
        if ds_agg is None:
            continue
        var_dict = {}
        for v, (typ, w) in VARIABLES_INFO.items():
            if v in ds_agg:
                var_dict[v] = ds_agg[v]
        cost_map = cost_function(var_dict)
        mgmt_dict.setdefault(mgmt, {})[ssp] = cost_map

    if not mgmt_dict:
        print("No valid simulations found. Exiting.")
        return {}

    # Use the first valid scenario to reference latitude/longitude
    sample_m = next(iter(mgmt_dict))
    sample_s = next(iter(mgmt_dict[sample_m]))
    ref_cost = mgmt_dict[sample_m][sample_s]
    latc = ref_cost.coords["lat"]
    lonc = ref_cost.coords["lon"]

    # DataArray for the chosen scenario's ID
    sel_da = xr.DataArray(
        np.zeros(ref_cost.squeeze(drop=True).shape, dtype=int),
        coords={"lat": latc, "lon": lonc},
        dims=["lat", "lon"]
    )
    # DataArray for the chosen scenario's cost
    final_cost = xr.full_like(sel_da, np.inf, dtype=float)

    mgmt_list_keys = sorted(mgmt_dict.keys())
    mgmt_id_map = {m: i + 1 for i, m in enumerate(mgmt_list_keys)}

    # Identify the BAU scenario, if any
    bau_mg_list = [m for m in mgmt_dict if BAU_KEYWORD in m.upper()]
    bau_name = bau_mg_list[0] if bau_mg_list else None

    # 2) Loop over pixels
    for i in range(len(latc)):
        for j in range(len(lonc)):
            best_mgmt = None
            best_val = np.inf

            for mg in mgmt_list_keys:
                # Retrieve all ssp values (worst-case or average, depending on the method)
                ssp_vals = []
                for ssp, c_map in mgmt_dict[mg].items():
                    val_c = c_map.isel(lat=i, lon=j).values
                    ssp_vals.append(val_c)

                if not ssp_vals or np.any(np.isnan(ssp_vals)):
                    # Invalid or ocean region
                    continue

                if robust_method == "maximin":
                    # We look at the worst cost (max) and try to minimize that.
                    w_cost = np.max(ssp_vals)
                    if w_cost < best_val:
                        best_val = w_cost
                        best_mgmt = mg

                elif robust_method == "safetyfirst" and bau_name is not None:
                    # Compare scenario's worst cost to BAU's worst cost
                    w_cost = np.max(ssp_vals)
                    bau_cost_list = list(mgmt_dict[bau_name].values())
                    bau_stacked = xr.concat(bau_cost_list, dim="ssp")
                    local_bau_worst = bau_stacked.isel(lat=i, lon=j).max().values
                    if w_cost <= local_bau_worst:
                        # Among those that beat or match BAU's worst cost, pick the best average
                        avg_ = np.mean(ssp_vals)
                        if avg_ < best_val:
                            best_val = avg_
                            best_mgmt = mg

                elif robust_method == "classic_threshold":
                    # First, use the "classic" logic: minimize the average
                    avg_ = np.mean(ssp_vals)
                    if avg_ < best_val:
                        best_val = avg_
                        best_mgmt = mg

                else:
                    # "classic" => minimize the average cost
                    avg_ = np.mean(ssp_vals)
                    if avg_ < best_val:
                        best_val = avg_
                        best_mgmt = mg

            # If using "classic_threshold", compare to BAU and only adopt if improvement >= threshold
            if robust_method == "classic_threshold" and bau_name is not None:
                # Compute BAU cost (average across its SSPs)
                cost_list_bau = list(mgmt_dict[bau_name].values())
                stacked_bau = xr.concat(cost_list_bau, dim="ssp")
                bau_val = stacked_bau.isel(lat=i, lon=j).mean().values

                if not np.isnan(bau_val):
                    # Improvement calculation: (bau_val - best_val) / abs(bau_val)
                    # if it doesn't reach improvement_threshold, remain with BAU
                    gain = (bau_val - best_val) / abs(bau_val)
                    if gain < improvement_threshold:
                        best_val = bau_val
                        best_mgmt = bau_name

            # Record final result
            if best_mgmt is None or np.isinf(best_val):
                sel_da[i, j] = 0
                final_cost[i, j] = np.inf
            else:
                sel_da[i, j] = mgmt_id_map[best_mgmt]
                final_cost[i, j] = best_val

    ds_out = xr.Dataset({"CSF_selected": sel_da, "Cost_selected": final_cost})
    ds_out.to_netcdf(OUTPUT_SELECTED_NC)
    ds_out.close()
    print(f"=> Created {OUTPUT_SELECTED_NC} with method={robust_method}, threshold={improvement_threshold}")
    return mgmt_id_map


def invert_mgmt_ids(mgmt_id_map):
    """
    Inverts a dictionary {mgmt_name: int_id} to {int_id: mgmt_name}.
    """
    return {v: k for k, v in mgmt_id_map.items()}


def visualize_named_map(nc_file=OUTPUT_SELECTED_NC, id_to_name=None, png_file=OUTPUT_PNG):
    """
    Creates a quick color-coded map of the selected scenario IDs, 
    labeling them by name in a colorbar.
    """
    if not id_to_name:
        print("No id_to_name dict provided, skipping map.")
        return

    ds = xr.open_dataset(nc_file)
    if "CSF_selected" not in ds:
        print(f"'CSF_selected' not found in {nc_file}, cannot visualize.")
        return

    arr = ds["CSF_selected"].copy()
    arr = arr.where(arr != 0, np.nan)  # ocean => NaN
    data_2d = arr.values

    used_ids = np.unique(data_2d[~np.isnan(data_2d)]).astype(int)
    if len(used_ids) == 0:
        print("No valid IDs in CSF_selected. Nothing to plot.")
        return

    max_id = used_ids.max()
    ncat = max_id if max_id > 0 else 1

    cmap = plt.get_cmap("tab20", ncat)
    cmap.set_bad('white')
    boundaries = np.arange(0.5, ncat + 1.5, 1)
    norm = mcolors.BoundaryNorm(boundaries, ncat)

    fig, ax = plt.subplots(figsize=(8, 6))
    lats = arr.coords["lat"].values
    lons = arr.coords["lon"].values

    im = ax.pcolormesh(lons, lats, data_2d,
                       cmap=cmap, norm=norm, shading='auto')
    ax.set_title("CSF_selected (Named Legend)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cb = fig.colorbar(im, ax=ax, boundaries=boundaries, ticks=range(1, ncat + 1))
    tick_labels = []
    for i in range(1, ncat + 1):
        tick_labels.append(id_to_name.get(i, f"ID_{i}"))
    cb.ax.set_yticklabels(tick_labels)

    fig.tight_layout()
    plt.savefig(png_file, dpi=150)
    plt.close()
    print(f"=> Saved scenario map as {png_file}")


# --------------------------------------------------------------------------------
# Zonal stats integration
# --------------------------------------------------------------------------------

def do_zonal_stats(shapefile, netcdf_file, varname,
                   id_to_name, zone_id_field=SHAPE_ZONE_ID,
                   out_csv=OUTPUT_CSV):
    """
    Performs zonal statistics using the selected scenario map.
    We write out a CSV listing, for each zone, the top-3 scenario IDs (and their counts).
    """
    ds = xr.open_dataset(netcdf_file)
    if varname not in ds:
        print(f"Variable '{varname}' not in {netcdf_file}.")
        return
    da = ds[varname]
    ds.close()

    da = da.rio.write_crs("EPSG:4326", inplace=True)
    out_tif = "CSF_selected_map.tif"
    da.rio.to_raster(out_tif)

    gdf = gpd.read_file(shapefile)
    if gdf.crs is None:
        print("WARNING: the shapefile lacks a defined projection (crs).")

    stats = zonal_stats(
        gdf,
        out_tif,
        layer=0,
        categorical=True,
        nodata=np.nan
    )

    results = []
    for i, row in enumerate(gdf.itertuples()):
        zone_key = getattr(row, zone_id_field)
        cat_dict = stats[i]
        if cat_dict is None:
            results.append({
                'zone_id': zone_key,
                'top1_csf': None, 'top1_count': 0,
                'top2_csf': None, 'top2_count': 0,
                'top3_csf': None, 'top3_count': 0,
            })
            continue

        cat_items = [(k, v) for k, v in cat_dict.items() if k != 0]
        cat_items.sort(key=lambda x: x[1], reverse=True)

        cat_items_named = []
        for k, v in cat_items:
            scenario_name = id_to_name.get(k, f"UnknownID_{k}")
            cat_items_named.append((scenario_name, v))

        top3 = cat_items_named[:3]
        while len(top3) < 3:
            top3.append(("None", 0))

        results.append({
            'zone_id': zone_key,
            'top1_csf':  top3[0][0], 'top1_count': top3[0][1],
            'top2_csf':  top3[1][0], 'top2_count': top3[1][1],
            'top3_csf':  top3[2][0], 'top3_count': top3[2][1],
        })

    df_out = pd.DataFrame(results)

    # Compute percentage: sum of the top-3 'count' is the total for that zone
    df_out['top1_pct'] = (
        df_out['top1_count']
        / (df_out['top1_count'] + df_out['top2_count'] + df_out['top3_count'])
        * 100
    )
    df_out['top2_pct'] = (
        df_out['top2_count']
        / (df_out['top1_count'] + df_out['top2_count'] + df_out['top3_count'])
        * 100
    )
    df_out['top3_pct'] = (
        df_out['top3_count']
        / (df_out['top1_count'] + df_out['top2_count'] + df_out['top3_count'])
        * 100
    )

    df_out.to_csv(out_csv, index=False)
    print(f"=> Wrote zonal stats to {out_csv}")
    print(df_out.head(10))


# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rebuild_out_optim_v3.py <simulation_input_list.txt> <robust_method> [improvement_threshold]")
        sys.exit()

    sim_list_file = sys.argv[1]
    robust_method = sys.argv[2].lower()

    # Read simulation list
    with open(sim_list_file, "r") as f:
        sim_list = [line.strip() for line in f if line.strip()]

    # If using 'classic_threshold', parse improvement threshold from argv
    improvement_threshold = 0.0
    if robust_method == "classic_threshold":
        if len(sys.argv) >= 4:
            improvement_threshold = float(sys.argv[3])
        else:
            print("No threshold provided, defaulting to 0.0 (similar to classic).")

    # A) Select the CSF (writes Fout_CSF_selected_robust.nc)
    mgmt_ids = create_selected_output(
        sim_list,
        robust_method=robust_method,
        improvement_threshold=improvement_threshold
    )
    print("Part A done.")

    # B) Visualization
    id_to_name = invert_mgmt_ids(mgmt_ids)
    visualize_named_map(
        nc_file=OUTPUT_SELECTED_NC,
        id_to_name=id_to_name,
        png_file=OUTPUT_PNG
    )
    print("Part B done.")

    # C) Zonal stats => top3
    do_zonal_stats(
        shapefile=SHAPEFILE,
        netcdf_file=OUTPUT_SELECTED_NC,
        varname=VAR_NAME,
        id_to_name=id_to_name,
        zone_id_field=SHAPE_ZONE_ID,
        out_csv=OUTPUT_CSV
    )
    print("Part C done.")
    print("All done.")
