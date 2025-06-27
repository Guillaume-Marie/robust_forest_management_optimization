#!/usr/bin/env python3
"""
rebuild_out_optim_v7.py – version CSV
====================================

Cette version remplace l’ancien chargement depuis les NetCDF annuels
par l’utilisation directe du CSV agrégé généré par
``aggregate_cost_sep_v2.py`` (par défaut : ``Fout_aggreg.csv``).

Le CSV doit contenir les colonnes suivantes :
  * ``lat`` et ``lon`` (coordonnées des pixels)
  * ``mgmt`` (nom de la pratique forestière)
  * ``ssp``  (scénario SSP)
  * une colonne par variable énumérée dans ``VARIABLES_CONFIG``.

Le reste de la logique (normalisation, calcul du coût, sélection et
visualisations) est inchangé.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional


import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr

###############################################################################
# Configuration générale
###############################################################################
AGGREGATED_CSV_PATH = "data_aggregated.csv"  # <== change ici si besoin
BAU_KEYWORD = "BAU"

# Variables à combiner dans la fonction de coût.
# ``type`` = "benefit" => signe négatif dans le coût (on veut maximiser)
VARIABLES_CONFIG: Dict[str, Dict] = {
    "HARVEST_FOREST_c": {"type": "benefit", "weight": 0.5},
    #"NBP_pool_c": {"type": "benefit", "weight": 0.5},
    "WSTRESS_SEASON": {"type": "cost", "weight": 0.5},
}

# Output filenames
OUTPUT_MAP_PNG_TEMPLATE = 'Fout_CSF_SSPs_{variable}_{method}{threshold}.png'
SELECTED_VAR_NAME = 'HARVEST_FOREST_c'

# Masque océan (0 = terre, 1 = océan)
OCEAN_MASK_PATH = "/home/mguill/Documents/robust_forest_management_optimization/mask_océan.tif"
ocean_mask = rxr.open_rasterio(OCEAN_MASK_PATH)
if "band" in ocean_mask.dims:
    ocean_mask = ocean_mask.sel(band=1)
if {"x", "y"}.issubset(ocean_mask.dims):
    ocean_mask = ocean_mask.rename({"x": "lon", "y": "lat"})
ocean_mask = ocean_mask.rio.write_crs("EPSG:4326", inplace=True)

###############################################################################
# Fonctions utilitaires
###############################################################################

def parse_simulation_name(sim_entry: str) -> Tuple[str, str]:
    """Extrait (mgmt, ssp) du nom de simulation 'PracticeSSP*'."""
    m = re.search(r"(.+)(SSP\d+)$", sim_entry)
    return (m.group(1), m.group(2)) if m else (sim_entry, "NOSSP")


def load_csv_cached(path: str = AGGREGATED_CSV_PATH) -> pd.DataFrame:
    """Charge une seule fois le CSV agrégé."""
    if not hasattr(load_csv_cached, "_cache"):
        if not Path(path).exists():
            sys.exit(f"❌ CSV agrégé introuvable : {path}")
        load_csv_cached._cache = pd.read_csv(path)
    return load_csv_cached._cache


def dataframe_to_dataset(df: pd.DataFrame) -> Optional[xr.Dataset]:
    """Convertit un DataFrame (lat, lon, var¹…) en xarray.Dataset.

    ⚠️ Gère les doublons lat/lon en agrégeant par la moyenne.
    """
    vars_cols = [c for c in df.columns if c not in {"lat", "lon", "mgmt", "ssp"}]
    if df.empty or not vars_cols:
        return None

    # 1. Garantir unicité lat/lon (agrégation)
    df_uniq = (
        df.groupby(["lat", "lon"], as_index=False)[vars_cols]
        .mean()
        .sort_values(["lat", "lon"], ignore_index=True)
    )

    # 2. Conversion vers xarray
    ds = df_uniq.set_index(["lat", "lon"]).to_xarray()
    return ds


def compute_cost(dataset: xr.Dataset) -> xr.DataArray:
    """Combine les variables normalisées en un scalaire de coût."""
    cost = None
    for var, cfg in VARIABLES_CONFIG.items():
        if var not in dataset:
            continue
        arr = dataset[var]
        factor = -cfg["weight"] if cfg["type"] == "benefit" else cfg["weight"]
        term = factor * arr
        cost = term if cost is None else cost + term
    return cost

###############################################################################
# Étape principale : sélection de la pratique par pixel
###############################################################################

def create_selected_output(sim_list: List[str], selection_method: str = "classic", threshold_improvement: float = 0.0):
    csv_df = load_csv_cached()

    # Stocke : mgmt -> ssp -> Dataset ou DataArray (coût)
    mgmt_raw: Dict[str, Dict[str, xr.Dataset]] = {}
    var_min: Dict[str, float] = {}
    var_max: Dict[str, float] = {}

    # ---------------------------------------------------------------------
    # 1. Construction des Dataset agrégés pour chaque simulation
    # ---------------------------------------------------------------------
    for sim in sim_list:
        mgmt, ssp = parse_simulation_name(sim)
        sub = csv_df[(csv_df["mgmt"] == mgmt) & (csv_df["ssp"] == ssp)]
        ds = dataframe_to_dataset(sub)
        if ds is None:
            print(f"Avertissement : données manquantes pour {sim}")
            continue
        mgmt_raw.setdefault(mgmt, {})[ssp] = ds
        # min/max globaux (pour normalisation)
        for var in VARIABLES_CONFIG:
            if var in ds:
                vmin = float(ds[var].min())
                vmax = float(ds[var].max())
                var_min[var] = vmin if var not in var_min else min(var_min[var], vmin)
                var_max[var] = vmax if var not in var_max else max(var_max[var], vmax)

    if not mgmt_raw:
        sys.exit("❌ Aucune simulation valide trouvée dans le CSV.")

    # ---------------------------------------------------------------------
    # 2. Normalisation 0‑1 + calcul du coût
    # ---------------------------------------------------------------------
    mgmt_cost: Dict[str, Dict[str, xr.DataArray]] = {}
    for mgmt, ssp_map in mgmt_raw.items():
        for ssp, ds in ssp_map.items():
            for var in VARIABLES_CONFIG:
                if var in ds:
                    vmin, vmax = var_min[var], var_max[var]
                    ds[var] = 0.0 if np.isclose(vmin, vmax) else (ds[var] - vmin) / (vmax - vmin)
                    #ds[var] = ds[var]
            mgmt_cost.setdefault(mgmt, {})[ssp] = compute_cost(ds)

    ocean_bool = ocean_mask == 1
    # ---------------------------------------------------------------------
    # 4. Sélection finale (méthodes classic, maximin, safetyfirst, classic_threshold)
    # ---------------------------------------------------------------------
    sample_mgmt = next(iter(mgmt_cost))
    sample_ssp = next(iter(mgmt_cost[sample_mgmt]))
    ref_arr = mgmt_cost[sample_mgmt][sample_ssp]
    lats, lons = ref_arr["lat"], ref_arr["lon"]

    # tableau rempli de chaînes vides ""
    empty_str = np.full(ref_arr.shape, "", dtype=object)

    sel_id = xr.DataArray(
        empty_str,
        coords={"lat": lats, "lon": lons},
        dims=["lat", "lon"],
        name="sel_id"          # optionnel, juste pour avoir un nom
    )
    sel_cost = xr.full_like(sel_id, np.inf, dtype=float)
    mgmt_list = sorted(mgmt_cost.keys())
    bau_mgmt = next((m for m in mgmt_list if BAU_KEYWORD in m.upper()), None)

    for i_lat in range(len(lats)):
        for i_lon in range(len(lons)):
            best_mgmt = None
            best_val = np.inf
            for mgmt in mgmt_list:
                vals = np.array([mgmt_cost[mgmt][s].isel(lat=i_lat, lon=i_lon).values for s in mgmt_cost[mgmt]])
                if np.any(np.isnan(vals)):
                    continue
                if selection_method == "maximin":
                    worst = vals.max()
                    if worst < best_val:
                        best_val, best_mgmt = worst, mgmt
                elif selection_method == "safetyfirst" and bau_mgmt:
                    worst = vals.max()
                    bau_worst = np.array([mgmt_cost[bau_mgmt][s].isel(lat=i_lat, lon=i_lon).values for s in mgmt_cost[bau_mgmt]]).max()
                    if worst <= bau_worst:
                        avg = vals.mean()
                        if avg < best_val:
                            best_val, best_mgmt = avg, mgmt
                else:  # classic
                    avg = vals.mean()
                    if avg < best_val:
                        best_val, best_mgmt = avg, mgmt
            if selection_method == "classic_threshold" and bau_mgmt and best_mgmt:
                bau_avg = np.array([mgmt_cost[bau_mgmt][s].isel(lat=i_lat, lon=i_lon).values for s in mgmt_cost[bau_mgmt]]).mean()
                gain = (bau_avg - best_val) / abs(bau_avg) if bau_avg != 0 else 0
                if gain < threshold_improvement:
                    best_mgmt, best_val = bau_mgmt, bau_avg

            if best_mgmt:
                #print("BEST",best_mgmt,best_val, i_lat ,i_lon)
                sel_id[i_lat, i_lon] = best_mgmt
                sel_cost[i_lat, i_lon] = best_val

    final_ds = xr.Dataset({"CSF_selected": sel_id, "Cost_selected": sel_cost})
    out_csv = f"Fout_CSF_selected_HARVEST_FOREST_c_{selection_method}.csv"
    final_ds.to_dataframe().reset_index().to_csv(out_csv, index=False)
    return print("→ CSV sélection écrit :", out_csv)

def visualize_scenario_map_from_csv(csv_file,
                                    png_file,
                                    lat_col="lat",
                                    lon_col="lon",
                                    label_col="CSF_selected"):
    """
    Affiche la carte des scénarios CSF à partir d'un CSV « plat ».
    
    Le CSV doit au minimum contenir les colonnes : lat, lon, label_col.
    - `label_col` : chaîne identifiant le scénario (ex. « FM5pDIACUT »).
    - Les autres colonnes (Cost_selected…) sont ignorées pour l'affichage.
    """

    # ------------------------------------------------------------------
    # 1) Lecture + vérification
    # ------------------------------------------------------------------
    df = pd.read_csv(csv_file)
    needed = {lat_col, lon_col, label_col}
    if not needed.issubset(df.columns):
        raise ValueError(f"Colonnes manquantes : {needed - set(df.columns)}")

    # Nettoyage minimal
    df[label_col] = df[label_col].astype(str).str.strip()

    # ------------------------------------------------------------------
    # 2) Conversion “libellé → id_entier” (factorisation)
    # ------------------------------------------------------------------
    codes, uniques = pd.factorize(df[label_col].fillna("OCEAN"))
    df["id_plot"] = np.where(codes < 0, 0, codes + 1)   # 0 = pixel vide / océan
    id_to_label = {i + 1: lbl for i, lbl in enumerate(uniques)}

    # ------------------------------------------------------------------
    # 3) Construction de la grille 2D (lat × lon)
    # ------------------------------------------------------------------
    lats = np.sort(df[lat_col].unique())
    lons = np.sort(df[lon_col].unique())
    grid = np.full((len(lats), len(lons)), np.nan)

    lat_idx = {v: i for i, v in enumerate(lats)}
    lon_idx = {v: j for j, v in enumerate(lons)}

    for _, row in df.iterrows():
        i = lat_idx[row[lat_col]]
        j = lon_idx[row[lon_col]]
        grid[i, j] = row["id_plot"]

    # ------------------------------------------------------------------
    # 4) Colormap + légende
    # ------------------------------------------------------------------
    used_ids = np.sort(df["id_plot"].unique())
    # Colormap discrète avec autant de classes que de scénarios
    cmap = plt.get_cmap("tab20", len(used_ids))
    cmap.set_bad("white")          # pixels NaN = blanc
    norm = mcolors.BoundaryNorm(np.arange(0.5, len(used_ids) + 1.5), len(used_ids))

    # ------------------------------------------------------------------
    # 5) Affichage
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(lons, lats, grid, cmap=cmap, norm=norm, shading="auto")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Scénario CSF sélectionné")

    cbar = fig.colorbar(im, ax=ax, ticks=used_ids)
    cbar.ax.set_yticklabels([id_to_label.get(i, f"ID {i}") for i in used_ids])

    plt.tight_layout()
    plt.savefig(png_file, dpi=150)
    plt.close(fig)
    print(f"→ Carte sauvegardée : {png_file}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit("Usage: python rebuild_out_optim_v7.py <simulation_list.txt> <selection_method> [improvement_threshold]")

    sim_list_file = Path(sys.argv[1])
    method = sys.argv[2].lower()
    threshold = float(sys.argv[3]) if method == "classic_threshold" and len(sys.argv) >= 4 else 0.0

    if not sim_list_file.exists():
        sys.exit(f"Fichier liste simulations introuvable : {sim_list_file}")

    with sim_list_file.open() as f:
        simulations = [l.strip() for l in f if l.strip()]

    id_map = create_selected_output(simulations, selection_method=method, threshold_improvement=threshold)

    csv_name = f"Fout_CSF_selected_{SELECTED_VAR_NAME}_{method}.csv"
    png_name = OUTPUT_MAP_PNG_TEMPLATE.format(
        variable=SELECTED_VAR_NAME,
        method=method,
        threshold=str(threshold) if threshold else ''
    )
    visualize_scenario_map_from_csv(csv_name, png_name)

    print("Finished.")
