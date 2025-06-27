#!/usr/bin/env python3
"""
rebuild_out_optim_v5.py
=======================

Update to support two output file types (stomate or sechiba) and locate each variable in its proper file based on a new "pattern" field.
Usage :
  python3 rebuild_out_optim_v5.py <simulation_input_list.txt> <robust_method> [improvement_threshold]

Methods possible : classic, maximin, safetyfirst, classic_threshold
"""
import xarray as xr
import numpy as np
import sys
import os
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rasterstats import zonal_stats
import geopandas as gpd
import pandas as pd

# --------------------------------------------------------------------------------
# Configuration des variables à prendre en compte
#  - "type": "benefit" ou "cost"
#  - "weight": poids dans la fonction de coût
#  - "agg": mode d'agrégation temporelle
#  - "year_min"/"year_max": bornes temporelles
#  - "pattern": l'une des clés pour construire le nom de fichier (stomate, sechiba)
# IMPORTANT : on va normaliser chaque variable dans [0..1] avant de la combiner.
# --------------------------------------------------------------------------------
VARIABLES_CONFIG = {
    #"HARVEST_FOREST_c":  {"type":"benefit","weight":1.0,"agg":"sum", "year_min":2010,"year_max":2099,"pattern":"stomate"},      
    "NBP_pool_c":  {"type":"benefit","weight":1.0,"agg":"sum", "year_min":2010,"year_max":2099,"pattern":"stomate"},
}

BAU_KEYWORD = "BAU"  # identifie la simulation "Business As Usual"
BASE_PATH = "/home/mguill/Documents/output_CSF_v2/subset"
# Template pour générer le nom de fichier selon pattern de chaque variable
FILE_PATTERN_TEMPLATE = "{simulation}_{year}0101_{year}1231_1Y_{pattern}_history.nc"

OUTPUT_SELECTED_NC  = "Fout_CSF_NBP_robust.png"

if len(sys.argv) >= 4:
    thr = str(sys.argv[3])
else:
    thr = str("")

OUTPUT_PNG = (
    "Fout_CSF_SSPs_NBP_" + str(sys.argv[2].lower()) + thr + ".png"
)

OUTPUT_TIF      = "Fout_CSF_selected_robust.tif"
OUTPUT_TIF_4326 = "Fout_CSF_selected_robust_4326.tif"
OUTPUT_CSV      = "climate_zones_top3_csf.csv"
SHAPEFILE       = "enz_v8_4326.shp"
SHAPE_ZONE_ID   = "EnZ_name"
VAR_NAME        = "CSF_selected"

# --------------------------------------------------------------------------------
def parse_mgmt_and_ssp(sim_name):
    match = re.search(r"(.+)(SSP\d+)$", sim_name)
    if match:
        return match.group(1), match.group(2)
    return sim_name, "NOSSP"


def get_global_time_range(vars_cfg):
    mins, maxs = [], []
    for cfg in vars_cfg.values():
        mins.append(cfg["year_min"])
        maxs.append(cfg["year_max"])
    return min(mins), max(maxs)


def load_and_aggregate(sim_name):
    """
    Charge tous les fichiers nécessaires pour chaque pattern, extrait et agrège
    chaque variable selon sa configuration.
    Retourne un Dataset 2D (lat, lon) pour cette simulation.
    """
    global_min, global_max = get_global_time_range(VARIABLES_CONFIG)
    # Grouper variables par pattern de fichier
    pattern_groups = {}
    for var, cfg in VARIABLES_CONFIG.items():
        pattern_groups.setdefault(cfg["pattern"], []).append(var)

    data_vars = {}
    # Pour chaque pattern, charger et concatener
    for pattern, var_list in pattern_groups.items():
        # Chercher les fichiers existants
        files = []
        for yr in range(global_min, global_max+1):
            fn = FILE_PATTERN_TEMPLATE.format(
                simulation=sim_name, year=yr, pattern=pattern
            )
            fp = os.path.join(BASE_PATH, fn)
            if os.path.exists(fp):
                files.append(fp)
        if not files:
            continue
        # Charger et concat
        ds_list = [xr.open_dataset(f, decode_times=False) for f in files]
        ds_concat = xr.concat(ds_list, dim="time_counter") if len(ds_list)>1 else ds_list[0]
        for ds in ds_list:
            ds.close()

        # Extraire et aggréger chaque variable de ce pattern
        for var in var_list:
            if var not in ds_concat:
                continue
            arr = ds_concat[var]
            # réduction de dimension veget si présente
            if "veget" in arr.dims:
                arr = arr.mean(dim="veget")
            # découpage temporel selon config
            cfg = VARIABLES_CONFIG[var]
            start_idx = max(0, cfg["year_min"] - global_min)
            end_idx   = min(arr.sizes.get("time_counter",0)-1,
                              cfg["year_max"] - global_min)
            if start_idx > end_idx:
                continue
            sub = arr.isel(time_counter=slice(start_idx, end_idx+1))
            # on force skipna=False pour propager les NaN (océans)
            if cfg["agg"] == "sum":
                arr_agg = sub.sum(dim="time_counter")
            else:
                arr_agg = sub.mean(dim="time_counter")
            data_vars[var] = arr_agg

    if not data_vars:
        return None
    return xr.Dataset(data_vars)


def cost_function(var_dict):
    combined = None
    for var, arr in var_dict.items():
        cfg = VARIABLES_CONFIG[var]
        this_cost = (-cfg["weight"] * arr) if cfg["type"]=="benefit" else (cfg["weight"] * arr)
        combined = this_cost if combined is None else combined + this_cost
    return combined

def create_sorted_scenarios_output(
    mgmt_dict,
    output_nc="Fout_CSF_sorted_robust.nc"
):
    """
    mgmt_dict[mgmt][ssp] = DataArray 2D (lat, lon) contenant la fonction de coût
    pour la gestion 'mgmt' sous le SSP 'ssp'.

    1) On concatène tout dans un seul DataArray 'cost_3d' de dimension (scenario, lat, lon).
    2) On détecte les pixels "océans" = aucun coût valide => ID=0
    3) On trie en ORDRE DÉCROISSANT les coûts pour chaque (lat, lon) où il y a au moins un coût.
    4) On produit deux variables NetCDF :
       - CSF_sorted_id(csf_rank, lat, lon) : l'identifiant entier du scénario
       - Cost_sorted(csf_rank, lat, lon)   : la valeur du coût correspondante
    5) On génère un CSV "scenario_id_map.csv" associant scenario_id -> scenario_name
    6) Les pixels océans conservent CSF_sorted_id = 0 (tous les ranks)
       et Cost_sorted = NaN pour tous les ranks.

    Paramètres
    ----------
    mgmt_dict : dict
        Ex: mgmt_dict[mgmt][ssp] = xarray.DataArray( lat, lon )
    output_nc : str
        Nom du fichier NetCDF à générer

    Retourne
    --------
    ds_out : xarray.Dataset
        Dataset contenant "CSF_sorted_id" et "Cost_sorted".

    Écrit également un CSV "scenario_id_map.csv" listant l'association ID -> nom du scenario.
    """
    # 1) Construire la liste des noms de scénario, et collecter les DataArray de coût
    scenario_names = []
    scenario_arrays = []
    for mg in mgmt_dict:
        for ssp in mgmt_dict[mg]:
            scenario_names.append(f"{mg}_{ssp}")
            scenario_arrays.append(mgmt_dict[mg][ssp])
    
    if not scenario_arrays:
        print("Aucun scénario dans mgmt_dict => rien à classer.")
        return None

    # Concatène en dimension "scenario"
    cost_3d = xr.concat(scenario_arrays, dim="scenario")
    cost_3d = cost_3d.assign_coords({"scenario": ("scenario", scenario_names)})
    # Force l'ordre (scenario, lat, lon) si besoin
    cost_3d = cost_3d.transpose("scenario", "lat", "lon")

    n_scenarios = len(scenario_names)
    lat_vals = cost_3d.coords["lat"].values
    print(lat_vals)
    lon_vals = cost_3d.coords["lon"].values

    # 2) Détection des pixels océans : aucun coût valide => tout NaN sur la dimension scenario
    # => on repère la condition: cost_3d.isnull().all(dim="scenario")
    #    => un booléen 2D (lat, lon) = True si tout est NaN => océan
    ocean_mask_2d = cost_3d.isnull().all(dim="scenario") 
    
    # shape (lat, lon), True = océan
    # 3) Tri en ORDRE DÉCROISSANT
    # On convertit en numpy
    cost_np = cost_3d.values  # shape (Nscenario, Nlat, Nlon)

    # On fait un tri standard (ordre ascendant) => on inverse pour avoir DESC
    sorted_indices_asc = np.argsort(cost_np, axis=0)  # shape (Nscenario, Nlat, Nlon)
    # cost trié
    cost_sorted_np = np.take_along_axis(cost_np, sorted_indices_asc, axis=0)

    # 4) Construction d'un tableau d'IDs d'entiers
    scenario_ids = np.arange(n_scenarios, dtype=int) + 1  # +1 pour que le 1er ID=1
    # si vous préférez scenario_id=0..N-1, laissez tomber le +1

    # scenario_ids triés
    scenario_ids_sorted_np = scenario_ids[sorted_indices_asc]

    # 5) On enveloppe tout en DataArray
    Cost_sorted = xr.DataArray(
        cost_sorted_np,
        dims=("csf_rank","lat","lon"),
        coords={
            "csf_rank": np.arange(n_scenarios),
            "lat": lat_vals,
            "lon": lon_vals
        }
    )

    CSF_sorted_id = xr.DataArray(
        scenario_ids_sorted_np,
        dims=("csf_rank","lat","lon"),
        coords={
            "csf_rank": np.arange(n_scenarios),
            "lat": lat_vals,
            "lon": lon_vals
        }
    )

    # 6) Appliquer l'océan => ID=0 et cost=NaN
    # ocean_mask_2d = True => océan => on force CSF_sorted_id=0
    # => on broadcast ce masque sur la dimension csf_rank
    # => on fait .where(~mask, 0)
    # On veut : si ocean => ID=0, sinon garder la valeur calculée
    # pour Cost_sorted => si ocean => NaN
    ocean_mask_3d = ocean_mask_2d.broadcast_like(CSF_sorted_id.isel(csf_rank=0))
    # ocean_mask_3d a shape (lat, lon), on veut (csf_rank, lat, lon) => on "expand_dims"
    ocean_mask_3d = ocean_mask_3d.expand_dims(dim={"csf_rank": np.arange(n_scenarios)}, axis=0)

    # ID=0 en océan
    CSF_sorted_id = CSF_sorted_id.where(~ocean_mask_3d, 0)
    # cost=NaN en océan
    Cost_sorted = Cost_sorted.where(~ocean_mask_3d, np.nan)

    # 7) Construire le Dataset final
    ds_out = xr.Dataset({
        "CSF_sorted_id": CSF_sorted_id,  # dimension (csf_rank, lat, lon)
        "Cost_sorted":   Cost_sorted,    # dimension (csf_rank, lat, lon)
    })

    # 8) Table d'association ID -> Nom de scénario
    # scenario_id= i+1 => scenario_names[i]
    # +1 si vous avez fait scenario_ids = range(n_scenarios)+1
    scenario_map_csv = "scenario_id_map.csv"
    with open(scenario_map_csv, "w", encoding="utf-8") as f:
        f.write("scenario_id,scenario_name\n")
        # ID=0 => océan => On peut le mentionner (facultatif)
        f.write("0,OCEAN\n")
        for i, sname in enumerate(scenario_names):
            real_id = i + 1  # car on a +1 plus haut
            f.write(f"{real_id},{sname}\n")

    # Affichage
    print(f"Table ID -> Nom de scénario (total {len(scenario_names)}) + ID=0 pour océan:")
    print("  0 => OCEAN")
    for i, sname in enumerate(scenario_names):
        print(f"  ID={i+1} => {sname}")
    print(f"=> CSV d'association écrit dans {scenario_map_csv}")

    # 9) Sauvegarde NetCDF
    ds_out = ds_out.sortby("lat", ascending=True)
    ds_out.to_netcdf(output_nc)
    print(f"=> Fichier {output_nc} créé (csf_rank, lat, lon). ID=0 => océan.")

    return ds_out

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

    ds_sorted = create_sorted_scenarios_output(mgmt_dict)
    ds_sorted.to_netcdf("Fout_CSF_sorted_robust.nc")
    
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

    ocean_mask_2d = ds_sorted["Cost_sorted"].isnull().all(dim="csf_rank")
    
    # 3) Sélection du meilleur scénario pixel par pixel
    for i in range(len(latc)):
        for j in range(len(lonc)):

            if ocean_mask_2d.isel(lat=i, lon=j):
                sel_da[i, j]    = 0
                final_cost[i,j] = np.inf
                continue

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
    # --- Nouveau : exporter un CSV plat lat, lon, csf_id, cost ---
    df = xr.Dataset({
        "csf_id": ds_out["CSF_selected"],
        "cost": ds_out["Cost_selected"]
    }).to_dataframe().reset_index()

    # Supprimer les pixels océans où le coût est -0.0 ou 0.0
    df = df[~np.isclose(df["cost"], 0.0)]

    # Remplacer ID 0 (océan) par NaN pour plus de lisibilité
    df["csf_id"] = df["csf_id"].replace({0: pd.NA})

    # Ajouter le nom de scénario
    inv_map = {v: k for k, v in mgmt_id_map.items()}
    mgmt_id_df = pd.DataFrame([
        {"csf_id": id_, "scenario": name} for id_, name in inv_map.items()
    ])
    df = df.merge(mgmt_id_df, on="csf_id", how="left")

    # Écriture du CSV
    out_csv = f"Fout_CSF_NBP_{robust_method}.csv"
    df.to_csv(out_csv, index=False)
    print(f"→ CSV plat écrit dans {out_csv}")

    return mgmt_id_map


def invert_mgmt_ids(mgmt_id_map):
    return {v: k for k, v in mgmt_id_map.items()}


def visualize_named_map_from_csv(csv_file, png_file="map_selected_practice.png"):
    """
    Construit une carte 2D à partir d'un CSV contenant lat, lon, csf_id et scenario.
    Les valeurs 0 ou manquantes sont traitées comme océans (blanc).
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    import pandas as pd

    if not os.path.exists(csv_file):
        print(f"❌ Fichier CSV introuvable : {csv_file}")
        return

    df = pd.read_csv(csv_file)
    if not {"lat", "lon", "csf_id", "scenario"}.issubset(df.columns):
        print("❌ Colonnes manquantes dans le CSV.")
        return

    # Nettoyage : valeurs NaN ou csf_id==0 traitées comme océan
    df["csf_id"] = df["csf_id"].fillna(0).astype(int)
    ocean_mask = df["csf_id"] == 0

    # Création grille régulière
    lats = sorted(df["lat"].unique())
    lons = sorted(df["lon"].unique())
    id_grid = np.full((len(lats), len(lons)), np.nan)

    lat_idx = {lat: i for i, lat in enumerate(lats)}
    lon_idx = {lon: j for j, lon in enumerate(lons)}

    for _, row in df.iterrows():
        i = lat_idx[row["lat"]]
        j = lon_idx[row["lon"]]
        if row["csf_id"] > 0:
            id_grid[i, j] = row["csf_id"]

    # Récupération des labels
    id_to_name = df.dropna(subset=["csf_id", "scenario"]).drop_duplicates(subset=["csf_id"]).set_index("csf_id")["scenario"].to_dict()
    used_ids = sorted(id_to_name.keys())
    ncat = len(used_ids)

    # Préparation affichage
    cmap = plt.get_cmap("Spectral", ncat)
    cmap.set_bad('white')
    norm = mcolors.BoundaryNorm(np.arange(0.5, ncat + 1.5), ncat)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(lons, lats, id_grid, cmap=cmap, norm=norm, shading='auto')
    ax.set_title("Pratique forestière optimale sélectionnée", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cbar = fig.colorbar(im, ax=ax, ticks=used_ids)
    cbar.ax.set_yticklabels([id_to_name[i] for i in used_ids])

    plt.tight_layout()
    plt.savefig(png_file, dpi=150)
    plt.close()
    print(f"✅ Carte sauvegardée dans {png_file}")

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
        print("Usage: python rebuild_out_optim_v5.py <simulation_input_list.txt> <robust_method> [improvement_threshold]")
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
    csv_file = f"Fout_CSF_NBP_{robust_method}.csv"
    visualize_named_map_from_csv(csv_file, png_file=OUTPUT_PNG)

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