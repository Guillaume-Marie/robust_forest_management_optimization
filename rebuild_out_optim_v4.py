#!/usr/bin/env python3
"""
rebuild_out_optim_v4.py
=======================

Nouvelle version pour normaliser chaque variable (0..1)
avant le calcul de la fonction de coût.

Usage :
  python rebuild_out_optim_v4.py <simulation_input_list.txt> <robust_method> [improvement_threshold]

Méthodes possibles : classic, maximin, safetyfirst, classic_threshold
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
# Configuration des variables à prendre en compte
#  - "type": "benefit" ou "cost" (pour le calcul de la fonction de coût)
#  - "weight": poids dans la fonction de coût
#  - "agg": mode d'agrégation temporelle
#  - "year_min"/"year_max": bornes temporelles
#
# IMPORTANT : on va normaliser chaque variable dans [0..1] avant de la combiner.
# --------------------------------------------------------------------------------
VARIABLES_CONFIG = {
    "TOTAL_SOIL_c": {
        "type"     : "benefit",
        "weight"   : 0.0,
        "agg"      : "mean",
        "year_min" : 2089,
        "year_max" : 2099
    },
    "NBP_pool_c": {
        "type"     : "benefit",
        "weight"   : 0.0,
        "agg"      : "sum",
        "year_min" : 2010,
        "year_max" : 2099
    },
    "NPP": {
        "type"     : "benefit",
        "weight"   : 0.0,
        "agg"      : "sum",
        "year_min" : 2010,
        "year_max" : 2099
    },
    "TOTAL_BM_LITTER_c": {
        "type"     : "benefit",
        "weight"   : 0.0,
        "agg"      : "mean",
        "year_min" : 2089,
        "year_max" : 2099
    },
    "HARVEST_FOREST_c": {
        "type"     : "benefit",
        "weight"   : 1.0,
        "agg"      : "sum",
        "year_min" : 2010,
        "year_max" : 2099
    },
}

BAU_KEYWORD = "BAU"  # Identifie la simulation "Business As Usual"
BASE_PATH = "/home/mguill/Documents/output_CSF_v2/subset"
FILE_PATTERN = "{simulation}_{year}0101_{year}1231_1Y_stomate_history.nc"

OUTPUT_SELECTED_NC  = "Fout_CSF_selected_robust.nc"
OUTPUT_PNG          = "Fout_CSF_selected_robust.png"
OUTPUT_TIF          = "Fout_CSF_selected_robust.tif"
OUTPUT_TIF_4326     = "Fout_CSF_selected_robust_4326.tif"
OUTPUT_CSV          = "climate_zones_top3_csf.csv"
SHAPEFILE           = "enz_v8_4326.shp"
SHAPE_ZONE_ID       = "EnZ_name"
VAR_NAME            = "CSF_selected"

# --------------------------------------------------------------------------------
def parse_mgmt_and_ssp(sim_name):
    match = re.search(r"(.+)(SSP\d+)$", sim_name)
    if match:
        mgmt_name = match.group(1)
        ssp = match.group(2)
        return mgmt_name, ssp
    else:
        return sim_name, "NOSSP"


def get_global_time_range(variables_config):
    """
    Détermine l'année min et l'année max globales en parcourant VARIABLES_CONFIG.
    """
    mins = []
    maxs = []
    for var_name, cfg in variables_config.items():
        mins.append(cfg["year_min"])
        maxs.append(cfg["year_max"])
    return min(mins), max(maxs)


def load_and_aggregate(sim_name):
    """
    Charge tous les fichiers NetCDF de la plage [global_min..global_max].
    Pour chaque variable de VARIABLES_CONFIG, on fait l'agrégation
    temporelle dans sa propre plage (year_min..year_max).
    Retourne un Dataset 2D (lat, lon) agrégé.
    """
    global_min, global_max = get_global_time_range(VARIABLES_CONFIG)
    files_found = []
    for yr in range(global_min, global_max + 1):
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
    for var_name, cfg in VARIABLES_CONFIG.items():
        if var_name not in ds_concat:
            continue
        arr = ds_concat[var_name]
        # si dimension veget => on la réduit
        if "veget" in arr.dims:
            arr = arr.mean(dim="veget")

        var_min = cfg["year_min"]
        var_max = cfg["year_max"]
        agg_mode = cfg["agg"]

        # index de début et fin
        start_idx = var_min - global_min
        end_idx   = var_max - global_min
        time_dim_size = arr.sizes.get("time_counter", 0)
        start_idx = max(0, start_idx)
        end_idx   = min(time_dim_size - 1, end_idx)

        if start_idx > end_idx:
            # pas de données valides
            continue

        arr_sub = arr.isel(time_counter=slice(start_idx, end_idx + 1))

        if agg_mode == "sum":
            arr_agg = arr_sub.sum(dim="time_counter")
        else:
            arr_agg = arr_sub.mean(dim="time_counter")

        data_vars[var_name] = arr_agg

    ds_out = xr.Dataset(data_vars)
    return ds_out


def cost_function(var_dict):
    """
    Combine (avec les poids) les variables "benefit" (coût = -valeur) et "cost" (coût = +valeur).
    On suppose que var_dict[var_name] est déjà *normalisé* dans [0..1].
    """
    combined = None
    for var_name, arr in var_dict.items():
        cfg = VARIABLES_CONFIG[var_name]
        var_type = cfg["type"]
        weight   = cfg["weight"]
        if var_type == "benefit":
            this_cost = -weight * arr
        else:
            this_cost = weight * arr
        combined = this_cost if (combined is None) else (combined + this_cost)
    return combined


def create_selected_output(sim_list, robust_method="classic", improvement_threshold=0.0):
    """
    On charge toutes les simulations, on agrège leurs variables,
    puis on normalise chaque variable (0..1) globalement sur l'ensemble
    des simulations, avant le calcul du coût. Enfin on applique la méthode
    de sélection (classic, maximin, etc.) pour choisir la meilleure gestion pixel par pixel.
    """
    # 1) Chargement dans un dictionnaire mgmt_dict[mgmt][ssp] = Dataset (contient var1, var2, ...)
    mgmt_dict = {}
    global_mins = {}
    global_maxs = {}

    # ---- (a) Première passe : charger TOUTES les données, accumuler min/max
    # pour calculer plus tard la normalisation
    for sim in sim_list:
        mgmt, ssp = parse_mgmt_and_ssp(sim)
        ds_agg = load_and_aggregate(sim)
        if ds_agg is None:
            continue
        mgmt_dict.setdefault(mgmt, {})[ssp] = ds_agg

        # Mise à jour des global_mins / global_maxs pour chaque variable
        for v in ds_agg.data_vars:
            vmin = ds_agg[v].min().values
            vmax = ds_agg[v].max().values
            if v not in global_mins:
                global_mins[v] = vmin
                global_maxs[v] = vmax
            else:
                global_mins[v] = min(global_mins[v], vmin)
                global_maxs[v] = max(global_maxs[v], vmax)

    if not mgmt_dict:
        print("Aucune simulation valide trouvée. Abandon.")
        return {}

    # ---- (b) Appliquer la normalisation [0..1] sur chaque variable
    # pour tous mgmt/ssp
    for mg in mgmt_dict:
        for ssp in mgmt_dict[mg]:
            ds_tmp = mgmt_dict[mg][ssp]
            for v in ds_tmp.data_vars:
                vmin = global_mins[v]
                vmax = global_maxs[v]
                if np.isclose(vmin, vmax):
                    # aucune variation, on force à 0
                    ds_tmp[v] = 0.0
                else:
                    ds_tmp[v] = (ds_tmp[v] - vmin) / (vmax - vmin)
            mgmt_dict[mg][ssp] = ds_tmp  # on remplace par la version normalisée

    # 2) On calcule la fonction de coût sur ces données normalisées
    #    mgmt_dict[mg][ssp] => un DataSet (chaque variable). On en fait un DataArray de coût
    #    cost_map = cost_function(...)
    # On stocke cost_map dans mgmt_dict[mg][ssp] = cost_map.
    # (ou on peut séparer un second dict)

    for mg in mgmt_dict:
        for ssp in mgmt_dict[mg]:
            ds_tmp = mgmt_dict[mg][ssp]
            var_dict = {v: ds_tmp[v] for v in ds_tmp.data_vars}
            c_map = cost_function(var_dict)
            # on remplace le Dataset par la cost_map
            mgmt_dict[mg][ssp] = c_map

    # On prend lat/lon d'un cost_map au hasard
    sample_m = next(iter(mgmt_dict))
    sample_s = next(iter(mgmt_dict[sample_m]))
    ref_cost = mgmt_dict[sample_m][sample_s]
    latc = ref_cost.coords["lat"]
    lonc = ref_cost.coords["lon"]

    sel_da = xr.DataArray(
        np.zeros(ref_cost.shape, dtype=int),
        coords={"lat": latc, "lon": lonc},
        dims=["lat", "lon"]
    )
    final_cost = xr.full_like(sel_da, np.inf, dtype=float)

    mgmt_list_keys = sorted(mgmt_dict.keys())
    mgmt_id_map = {m: i + 1 for i, m in enumerate(mgmt_list_keys)}

    # On cherche si on a une gestion BAU
    bau_mg_list = [m for m in mgmt_dict if BAU_KEYWORD in m.upper()]
    bau_name = bau_mg_list[0] if bau_mg_list else None

    # 3) Sélection du meilleur scénario pixel par pixel
    for i in range(len(latc)):
        for j in range(len(lonc)):
            best_mgmt = None
            best_val  = np.inf

            for mg in mgmt_list_keys:
                ssp_vals = []
                for ssp, c_map in mgmt_dict[mg].items():
                    local_val = c_map.isel(lat=i, lon=j).values
                    ssp_vals.append(local_val)

                if not ssp_vals or np.any(np.isnan(ssp_vals)):
                    continue

                if robust_method == "maximin":
                    worst_case = np.max(ssp_vals)
                    if worst_case < best_val:
                        best_val = worst_case
                        best_mgmt = mg
                elif robust_method == "safetyfirst" and bau_name is not None:
                    # Compare le pire du scénario au pire du BAU
                    worst_case = np.max(ssp_vals)
                    bau_costs = list(mgmt_dict[bau_name].values())  # cost_map BAU pour chaque SSP
                    bau_stack = xr.concat(bau_costs, dim="ssp")
                    local_bau_worst = bau_stack.isel(lat=i, lon=j).max().values
                    if worst_case <= local_bau_worst:
                        # ensuite on compare la moyenne
                        avg_ = np.mean(ssp_vals)
                        if avg_ < best_val:
                            best_val = avg_
                            best_mgmt = mg
                elif robust_method == "classic_threshold":
                    avg_ = np.mean(ssp_vals)
                    if avg_ < best_val:
                        best_val = avg_
                        best_mgmt = mg
                else:
                    # "classic" : on minimise la moyenne
                    avg_ = np.mean(ssp_vals)
                    if avg_ < best_val:
                        best_val = avg_
                        best_mgmt = mg

            # post-traitement pour 'classic_threshold'
            if robust_method == "classic_threshold" and bau_name is not None and best_mgmt is not None:
                # comparer gain par rapport à BAU
                bau_cost_list = list(mgmt_dict[bau_name].values())
                stacked_bau = xr.concat(bau_cost_list, dim="ssp")
                bau_val = stacked_bau.isel(lat=i, lon=j).mean().values
                if not np.isnan(bau_val):
                    gain = (bau_val - best_val) / abs(bau_val) if bau_val != 0 else 0
                    if gain < improvement_threshold:
                        best_val  = bau_val
                        best_mgmt = bau_name

            if best_mgmt is None or np.isinf(best_val):
                sel_da[i, j] = 0
                final_cost[i, j] = np.inf
            else:
                sel_da[i, j] = mgmt_id_map[best_mgmt]
                final_cost[i, j] = best_val

    ds_out = xr.Dataset({"CSF_selected": sel_da, "Cost_selected": final_cost})
    ds_out.to_netcdf(OUTPUT_SELECTED_NC)
    ds_out.close()
    print(f"=> Fichier {OUTPUT_SELECTED_NC} créé (méthode={robust_method}, seuil={improvement_threshold})")

    return mgmt_id_map


def invert_mgmt_ids(mgmt_id_map):
    return {v: k for k, v in mgmt_id_map.items()}


def visualize_named_map(nc_file=OUTPUT_SELECTED_NC, id_to_name=None, png_file=OUTPUT_PNG):
    if not id_to_name or not os.path.exists(nc_file):
        return

    ds = xr.open_dataset(nc_file)
    if "CSF_selected" not in ds:
        ds.close()
        return

    arr = ds["CSF_selected"].where(ds["CSF_selected"] != 0, np.nan)
    ds.close()

    data_2d = arr.values
    used_ids = np.unique(data_2d[~np.isnan(data_2d)]).astype(int)
    if len(used_ids) == 0:
        return

    max_id = used_ids.max()
    ncat = max(max_id, 1)

    cmap = plt.get_cmap("Spectral", ncat)
    cmap.set_bad('white')
    boundaries = np.arange(0.5, ncat + 1.5, 1)
    norm = mcolors.BoundaryNorm(boundaries, ncat)

    fig, ax = plt.subplots(figsize=(8, 6))
    lats = arr.coords["lat"].values
    lons = arr.coords["lon"].values

    im = ax.pcolormesh(lons, lats, data_2d, cmap=cmap, norm=norm, shading='auto')
    ax.set_title("Forestry action selected", fontdict={'fontsize': 14})
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cb = fig.colorbar(im, ax=ax, boundaries=boundaries, ticks=range(1, ncat + 1))
    tick_labels = [id_to_name.get(i, f"ID_{i}") for i in range(1, ncat + 1)]
    cb.ax.set_yticklabels(tick_labels)

    plt.tight_layout()
    plt.savefig(png_file, dpi=150)
    plt.close()
    print(f"=> Carte sauvegardée dans {png_file}")


def do_zonal_stats(shapefile, netcdf_file, varname, id_to_name,
                   zone_id_field=SHAPE_ZONE_ID, out_csv=OUTPUT_CSV):
    if not os.path.exists(netcdf_file):
        return
    ds = xr.open_dataset(netcdf_file)
    if varname not in ds:
        ds.close()
        return
    da = ds[varname]
    ds.close()

    da = da.rio.write_crs("EPSG:4326", inplace=True)
    tmp_tif = "CSF_selected_map_tmp.tif"
    da.rio.to_raster(tmp_tif)

    if not os.path.exists(shapefile):
        return

    gdf = gpd.read_file(shapefile)
    if gdf.crs is None:
        print("ATTENTION : shapefile sans projection (crs).")

    stats = zonal_stats(
        gdf,
        tmp_tif,
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

        # on ignore la catégorie 0
        cat_items = [(k, v) for (k, v) in cat_dict.items() if k != 0]
        cat_items.sort(key=lambda x: x[1], reverse=True)
        cat_items_named = [(id_to_name.get(k, f"UnknownID_{k}"), v) for (k, v) in cat_items]

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
    df_out['top1_pct'] = df_out['top1_count'] / (df_out['top1_count'] + df_out['top2_count'] + df_out['top3_count']) * 100
    df_out['top2_pct'] = df_out['top2_count'] / (df_out['top1_count'] + df_out['top2_count'] + df_out['top3_count']) * 100
    df_out['top3_pct'] = df_out['top3_count'] / (df_out['top1_count'] + df_out['top2_count'] + df_out['top3_count']) * 100

    df_out.to_csv(out_csv, index=False)
    print(f"=> Statistiques zonales écrites dans {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rebuild_out_optim_v4.py <simulation_input_list.txt> <robust_method> [improvement_threshold]")
        sys.exit(1)

    sim_list_file = sys.argv[1]
    robust_method = sys.argv[2].lower()

    if not os.path.exists(sim_list_file):
        print(f"Fichier {sim_list_file} introuvable.")
        sys.exit(1)

    with open(sim_list_file, "r") as f:
        sim_list = [line.strip() for line in f if line.strip()]

    improvement_threshold = 0.0
    if robust_method == "classic_threshold":
        if len(sys.argv) >= 4:
            improvement_threshold = float(sys.argv[3])
        else:
            print("Aucun seuil fourni pour 'classic_threshold', on considère 0.0.")

    # A) Sélection du scénario optimal
    mgmt_ids = create_selected_output(
        sim_list,
        robust_method=robust_method,
        improvement_threshold=improvement_threshold
    )
    if not mgmt_ids:
        sys.exit(0)

    # B) Visualisation
    id_to_name = invert_mgmt_ids(mgmt_ids)
    visualize_named_map(
        nc_file=OUTPUT_SELECTED_NC,
        id_to_name=id_to_name,
        png_file=OUTPUT_PNG
    )

    # C) Statistiques zonales (optionnel)
    do_zonal_stats(
        shapefile=SHAPEFILE,
        netcdf_file=OUTPUT_SELECTED_NC,
        varname=VAR_NAME,
        id_to_name=id_to_name,
        zone_id_field=SHAPE_ZONE_ID,
        out_csv=OUTPUT_CSV
    )
    print("Terminé.")
