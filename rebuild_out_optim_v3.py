#!/usr/bin/env python3
"""
rebuild_out_optim_v2.py
=======================

Combined script:
  1) Performs robust selection of management scenarios per pixel (original logic).
  2) Creates Fout_CSF_selected_robust.nc (with best scenario ID).
  3) Exports that as a GeoTIFF after assigning EPSG:4326.
  4) Reprojects to EPSG:3035 to match the shapefile's CRS.
  5) Uses rasterstats.zonal_stats to compute frequencies of scenario IDs within each polygon.
  6) Maps scenario IDs to names (via mgmt_id_map) and extracts top 3 CSFs per zone.

Usage:
  python rebuild_out_optim_v3.py <simulation_input_list.txt> <robust_method>
  # e.g.: python rebuild_out_optim_v3.py CSF_experiment_list.txt maximin

  # The script will produce:
  #   - Fout_CSF_selected_robust.nc
  #   - Fout_CSF_selected_robust.tif    (in EPSG:4326)
  #   - Fout_CSF_selected_robust_3035.tif
  #   - climate_zones_top3_csf.csv      (zonal stats result)

Requires libraries:
  - xarray
  - numpy
  - matplotlib
  - rioxarray (for .rio reproject)
  - geopandas
  - rasterstats (for zonal_stats)
"""

import xarray as xr
import numpy as np
import sys
import os
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# For the zonal part:
import rioxarray
from rasterstats import zonal_stats
import geopandas as gpd
import pandas as pd


# -----------------------------------------------------------
# PART A) Original robust selection logic
# -----------------------------------------------------------

BASE_PATH = "/home/mguill/Documents/output_CSF_v2"
ANNEE_MIN = 2089
ANNEE_MAX = 2099
FILE_PATTERN = "{simulation}_{year}0101_{year}1231_1Y_stomate_history.nc"

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
OUTPUT_PNG = "Fout_CSF_selected_robust.png"

def parse_mgmt_and_ssp(sim_name):
    match = re.search(r"(.+)(SSP\d+)$", sim_name)
    if match:
        mgmt_name = match.group(1)
        ssp = match.group(2)
        return mgmt_name, ssp
    else:
        return sim_name, "NOSSP"


def load_and_aggregate(sim_name):
    """
    Read the netcdf files from BASE_PATH/sim_name... 
    and average from ANNEE_MIN..ANNEE_MAX.
    Returns an xarray Dataset with aggregated variables
    (still 2D in lat/lon).
    """
    files_found = []
    for yr in range(ANNEE_MIN, ANNEE_MAX+1):
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
    for v, (typ, w) in VARIABLES_INFO.items():
        if v in ds_concat:
            arr = ds_concat[v]
            if "veget" in arr.dims:
                # Weighted or not? In your original code you did a simple mean.
                # If you want a weighted mean with veget_max, do so here.
                arr = arr.mean(dim="veget")
            # sum or mean over time
            if AGGREGATION_MODES.get(v,"mean") == "sum":
                arr = arr.sum(dim="time_counter")
            else:
                arr = arr.mean(dim="time_counter")
            data_vars[v] = arr

    # On agrège ensuite veget_max si elle est disponible
    if "VEGET_MAX" in ds_concat:
        # Par exemple, on la moyenne sur time_counter
        data_vars["VEGET_MAX"] = ds_concat["VEGET_MAX"].mean(dim="time_counter")

    return xr.Dataset(data_vars)


def cost_function(var_dict):
    """
    Convert variables to a single cost measure:
      - "benefit" => cost = -weight*arr
      - "cost"    => cost =  weight*arr
    Then sum.
    """
    combined = None
    for var_name, (var_type, weight) in VARIABLES_INFO.items():
        if var_name not in var_dict:
            continue
        arr = var_dict[var_name]
        if var_type == "benefit":
            this_cost = -weight*arr
        else:
            this_cost = weight*arr
        if combined is None:
            combined = this_cost
        else:
            combined = combined + this_cost
    return combined


def create_fout_selected(sim_list, robust_method="classic"):
    """
    Gathers cost data from all sims => group by mgmt_name, ssp
    Then picks scenario per pixel using robust_method (classic, maximin, safetyfirst, etc.).
    Writes Fout_CSF_selected_robust.nc
    Returns a dictionary mgmt_id_map = { mgmt_name : int_id }
    """
    mgmt_dict = {}  # mgmt_dict[mgmt_name][ssp] = cost_map
    for sim in sim_list:
        mgmt, ssp = parse_mgmt_and_ssp(sim)
        ds_agg = load_and_aggregate(sim)
        if ds_agg is None:
            continue
        var_dict = {}
        for v, (typ, w) in VARIABLES_INFO.items():
            if v in ds_agg:
                arr = ds_agg[v]
                if "veget" in arr.dims and "VEGET_MAX" in ds_agg:
                    # moyenne pondérée par veget_max (exemple)
                    weights = ds_agg["VEGET_MAX"]
                    weighted_sum = (arr * weights).sum(dim="veget")
                    total_weights = weights.sum(dim="veget")
                    arr = weighted_sum / total_weights
                var_dict[v] = arr
        cost_map = cost_function(var_dict)
        mgmt_dict.setdefault(mgmt, {})[ssp] = cost_map

    if not mgmt_dict:
        print("No valid simulations found. Exiting.")
        return {}

    sample_m = next(iter(mgmt_dict))
    sample_s = next(iter(mgmt_dict[sample_m]))
    ref_cost = mgmt_dict[sample_m][sample_s]
    latc = ref_cost.coords["lat"]
    lonc = ref_cost.coords["lon"]

    sel_da = xr.DataArray(
        np.zeros(ref_cost.squeeze(drop=True).shape, dtype=int),
        coords={"lat": latc, "lon": lonc},
        dims=["lat", "lon"]
    )
    final_cost = xr.full_like(sel_da, np.inf, dtype=float)

    # For safetyfirst, define BAU threshold
    bau_thresh = None
    if robust_method == "safetyfirst":
        bau_mg = [m for m in mgmt_dict if BAU_KEYWORD in m.upper()]
        if bau_mg:
            bau = bau_mg[0]
            cost_list = list(mgmt_dict[bau].values())
            if cost_list:
                stacked = xr.concat(cost_list, dim="ssp")
                bau_thresh = stacked.max(dim="ssp")

    mgmt_list = sorted(mgmt_dict.keys())
    mgmt_id_map = {m: i+1 for i,m in enumerate(mgmt_list)}

    for i in range(len(latc)):
        for j in range(len(lonc)):
            # skip ocean or NaN?
            val_ref = ref_cost.isel(lat=i, lon=j).values
            if np.isnan(val_ref):
                # Mark as 0 => ocean/no scenario
                sel_da[i, j] = 0
                final_cost[i, j] = np.inf
                continue

            best_mgmt = None
            best_val = np.inf

            for mg in mgmt_list:
                ssp_vals = []
                for ssp, cmap in mgmt_dict[mg].items():
                    ssp_vals.append(cmap.isel(lat=i, lon=j).values)
                if not ssp_vals:
                    continue

                if robust_method == "maximin":
                    w_cost = np.max(ssp_vals)
                    if w_cost < best_val:
                        best_val = w_cost
                        best_mgmt = mg
                elif robust_method == "safetyfirst" and bau_thresh is not None:
                    w_cost = np.max(ssp_vals)
                    local_bau = bau_thresh.isel(lat=i, lon=j).values
                    if w_cost <= local_bau:
                        avg_ = np.mean(ssp_vals)
                        if avg_ < best_val:
                            best_val = avg_
                            best_mgmt = mg
                else:
                    # "classic" => min average
                    avg_ = np.mean(ssp_vals)
                    if avg_ < best_val:
                        best_val = avg_
                        best_mgmt = mg

            if robust_method == "safetyfirst" and best_mgmt is None and bau_thresh is not None:
                # fallback to BAU if none is feasible
                bau = bau_mg[0]
                cost_list = list(mgmt_dict[bau].values())
                stacked = xr.concat(cost_list, dim="ssp")
                local_bau = stacked.isel(lat=i, lon=j).values
                # e.g. worst or mean?
                best_val = np.mean(local_bau)
                best_mgmt = bau

            if best_mgmt is not None:
                sel_da[i,j] = mgmt_id_map[best_mgmt]
                final_cost[i,j] = best_val
            else:
                # ocean or no scenario
                sel_da[i,j] = 0
                final_cost[i,j] = np.inf

    ds_out = xr.Dataset({"CSF_selected": sel_da, "Cost_selected": final_cost})
    ds_out.to_netcdf(OUTPUT_SELECTED_NC)
    ds_out.close()
    print(f"=> Created {OUTPUT_SELECTED_NC} with method={robust_method}")
    return mgmt_id_map


def invert_mgmt_ids(mgmt_id_map):
    return {v: k for k, v in mgmt_id_map.items()}


def visualize_named_map(nc_file=OUTPUT_SELECTED_NC, id_to_name=None, png_file=OUTPUT_PNG):
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
    ncat = max_id if max_id>0 else 1

    cmap = plt.get_cmap("tab20", ncat)
    cmap.set_bad('white')
    boundaries = np.arange(0.5, ncat + 1.5, 1)
    norm = mcolors.BoundaryNorm(boundaries, ncat)

    fig, ax = plt.subplots(figsize=(8,6))
    lats = arr.coords["lat"].values
    lons = arr.coords["lon"].values

    im = ax.pcolormesh(lons, lats, data_2d,
                       cmap=cmap, norm=norm, shading='auto')
    ax.set_title("CSF_selected (Named Legend)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cb = fig.colorbar(im, ax=ax, boundaries=boundaries, ticks=range(1, ncat+1))
    tick_labels = []
    for i in range(1, ncat+1):
        if i in id_to_name:
            tick_labels.append(id_to_name[i])
        else:
            tick_labels.append(f"ID_{i}")
    cb.ax.set_yticklabels(tick_labels)

    fig.tight_layout()
    plt.savefig(png_file, dpi=150)
    plt.close()
    print(f"=> Saved scenario map as {png_file}")


# -----------------------------------------------------------
# PART B) Zonal stats integrated
# -----------------------------------------------------------

SHAPEFILE = "enz_v8_4326.shp"           # e.g. Europe's climate zones
SHAPE_ZONE_ID = "EnZ_name"         # or some attribute naming each zone
VAR_NAME = "CSF_selected"          # from Fout_CSF_selected_robust.nc
OUTPUT_TIF = "Fout_CSF_selected_robust.tif"
OUTPUT_TIF_3035 = "Fout_CSF_selected_robust_4326.tif"
OUTPUT_CSV = "climate_zones_top3_csf.csv"

def do_zonal_stats(shapefile, netcdf_file, varname,
                   id_to_name, zone_id_field=SHAPE_ZONE_ID,
                   out_csv=OUTPUT_CSV):
    """
    1) Opens the netcdf_file, extracts 'varname',
    2) Assigns EPSG:4326 (assuming lat/lon),
    3) Saves as GeoTIFF => OUTPUT_TIF,
    4) Reprojects to EPSG:3035 => OUTPUT_TIF_3035,
    5) Runs rasterstats.zonal_stats with the shapefile (EPSG:3035),
    6) Replaces ID with scenario names via 'id_to_name',
    7) Writes out a CSV with top3 frequencies.
    """
    # 1) open the netCDF
    ds = xr.open_dataset(netcdf_file)
    if varname not in ds:
        print(f"Variable '{varname}' not in {netcdf_file}.")
        return
    da = ds[varname]
    ds.close()

    # 2) Suppose lat/lon => EPSG:4326
    # If you're SURE it's lat/lon
    da = da.rio.write_crs("EPSG:4326", inplace=True)

    # Écrire en GeoTIFF
    out_tif = "CSF_selected_map.tif"
    da.rio.to_raster(out_tif)


    # 5) zonal stats
    gdf = gpd.read_file(shapefile)
    if gdf.crs is None:
        print("ATTENTION : le shapefile n'a pas de projection (crs).")

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

        # filter out ocean=0 if you wish:
        cat_items = [(k, v) for k, v in cat_dict.items() if k != 0]
        # sort descending by count
        cat_items.sort(key=lambda x: x[1], reverse=True)

        # rename ID => scenario name
        cat_items_named = []
        for k, v in cat_items:
            scenario_name = id_to_name.get(k, f"UnknownID_{k}")
            cat_items_named.append((scenario_name, v))

        top3 = cat_items_named[:3]
        while len(top3) < 3:
            top3.append(("None", 0))

        results.append({
            'zone_id'   : zone_key,
            'top1_csf'  : top3[0][0], 'top1_count': top3[0][1],
            'top2_csf'  : top3[1][0], 'top2_count': top3[1][1],
            'top3_csf'  : top3[2][0], 'top3_count': top3[2][1],
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_csv, index=False)
    print(f"=> Wrote zonal stats to {out_csv}")
    print(df_out.head(10))

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__=="__main__":
    if len(sys.argv)<3:
        print("Usage: python rebuild_out_optim_v2.py <simulation_input_list.txt> <robust_method>")
        sys.exit()

    sim_list_file = sys.argv[1]
    robust_method = sys.argv[2].lower()

    with open(sim_list_file,"r") as f:
        sim_list = [line.strip() for line in f if line.strip()]

    # A) Create Fout_CSF_selected_robust.nc
    mgmt_ids = create_fout_selected(sim_list, robust_method=robust_method)
    print("partie A done")
    # B) visualize
    id_to_name = invert_mgmt_ids(mgmt_ids)
    visualize_named_map(
        nc_file=OUTPUT_SELECTED_NC,
        id_to_name=id_to_name,
        png_file=OUTPUT_PNG
    )
    print("partie B done")
    # C) Perform zonal stats => top 3 CSFs
    do_zonal_stats(
        shapefile=SHAPEFILE,
        netcdf_file=OUTPUT_SELECTED_NC,
        varname=VAR_NAME,
        id_to_name=id_to_name,
        zone_id_field=SHAPE_ZONE_ID,
        out_csv=OUTPUT_CSV
    )
    print("partie C done")
    print("All done.")
