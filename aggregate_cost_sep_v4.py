#!/usr/bin/env python3
"""
aggregate_cost_sep_v4.py
========================

Version 4 – juin 2025
---------------------
* **Nouveau :**
  1. Prise en charge d’un **masque océans** (pixels mer) identique à celui
     utilisé dans `rebuild_out_optim_v6.py`. Les lignes dont le pixel est situé
     sur l’océan sont automatiquement supprimées du DataFrame final.
  2. **Moyenne pondérée** sur la dimension `veget` utilisant `veget_max` au
     lieu de la moyenne simple.
* L’API en ligne de commande reste inchangée :

```bash
python3 aggregate_cost_sep_v4.py <simulation_list.txt> [--output-csv data.csv]
```
"""

import xarray as xr
import numpy as np
import pandas as pd
import sys
import os
import rioxarray as rxr

# -----------------------------------------------------------------------------
# Configuration – à adapter
# -----------------------------------------------------------------------------
BASE_PATH              = "/home/mguill/Documents/output_CSF_v2/subset"
DEFAULT_YEAR_MIN       = 2010  # utilisé si year_min absent dans la cfg de la variable
DEFAULT_YEAR_MAX       = 2099  # utilisé si year_max absent dans la cfg de la variable
FILE_PATTERN_TEMPLATE  = "{simulation}_{year}0101_{year}1231_1Y_{pattern}_history.nc"

# Chemin vers le masque océans (GeoTIFF; 1=océan, 0=terre)
OCEAN_MASK_PATH = "/home/mguill/Documents/robust_forest_management_optimization/mask_océan.tif"
# -----------------------------------------------------------------------------
# Variables à extraire : pour chacune
#   * agg      : "sum" ou "mean"
#   * pattern  : "stomate" ou "sechiba"
#   * year_min : (optionnel) première année incluse
#   * year_max : (optionnel) dernière
# -----------------------------------------------------------------------------
VARIABLES_CONFIG = {
    # stomate
    "TOTAL_SOIL_c":       {"agg": "sum",  "pattern": "stomate", "year_min": 2099, "year_max": 2099},
    "NBP_pool_c":         {"agg": "sum",  "pattern": "stomate"},
    "NPP":                {"agg": "sum",  "pattern": "stomate"},
    "WSTRESS_SEASON":     {"agg": "mean", "pattern": "stomate"},
    "HET_RESP":           {"agg": "sum",  "pattern": "stomate"},
    "HARVEST_FOREST_c":   {"agg": "sum",  "pattern": "stomate"},
    # sechiba
    "netco2flux":         {"agg": "sum",  "pattern": "sechiba"},
    "humtot":             {"agg": "mean", "pattern": "sechiba"},
    "tsol_max":           {"agg": "mean", "pattern": "sechiba"},
    "tsol_min":           {"agg": "mean", "pattern": "sechiba"},
    "temp_sol":           {"agg": "mean", "pattern": "sechiba"},
    "alb_vis":            {"agg": "mean", "pattern": "sechiba"},
    "transpir":           {"agg": "sum",  "pattern": "sechiba"},
    "evap":               {"agg": "sum",  "pattern": "sechiba"},
    "evapot":             {"agg": "sum",  "pattern": "sechiba"},
    "transpot":           {"agg": "sum",  "pattern": "sechiba"},
}

OUTPUT_CSV = None  # sera défini par l’argument CLI

# -----------------------------------------------------------------------------
# Utilitaires
# -----------------------------------------------------------------------------

def parse_mgmt_and_ssp(sim_name: str):
    """Extrait les composantes mgmt et ssp du nom de simulation."""
    # Exemple : MGMT_A_SSP245 ➜ ("MGMT_A", "SSP245")
    parts = sim_name.split("SSP")
    mgmt = parts[0] 
    ssp  = "SSP"+ parts[1] if len(parts) >= 1 else "UNKNOWN"
    return mgmt, ssp


def collect_files(sim_name: str, pattern: str, year_min: int, year_max: int):
    """Retourne la liste des fichiers NetCDF correspondant aux critères."""
    files = []
    for year in range(year_min, year_max + 1):
        path = os.path.join(BASE_PATH, FILE_PATTERN_TEMPLATE.format(
            simulation=sim_name, year=year, pattern=pattern))
        if os.path.exists(path):
            files.append(path)
        else :
            print(path, "  do not exst!")
    return files


def load_ocean_mask():
    """Charge le masque océans en mémoire comme DataArray booléen (True = océan)."""
    if not os.path.exists(OCEAN_MASK_PATH):
        print(f"Warning: masque océans introuvable : {OCEAN_MASK_PATH}")
        return None
    mask = rxr.open_rasterio(OCEAN_MASK_PATH).sel(band=1)
    # Harmonise noms de dimensions avec xarray standards
    if "x" in mask.dims and "y" in mask.dims:
        mask = mask.rename({"x": "lon", "y": "lat"})
    # Réécrit le CRS si absent
    if mask.rio.crs is None:
        mask = mask.rio.write_crs("EPSG:4326", inplace=True)
    # Convertit en booléen : 1 = océan, 0 = terre
    mask = mask.astype(bool)
    return mask


OCEAN_MASK = load_ocean_mask()

# -----------------------------------------------------------------------------
# Agrégation d’une variable
# -----------------------------------------------------------------------------

def aggregate_variable(sim_name: str, var: str, cfg: dict):
    """Charge les fichiers nécessaires pour *une* variable et retourne un DataArray agrégé."""
    y_min = cfg.get("year_min", DEFAULT_YEAR_MIN)
    y_max = cfg.get("year_max", DEFAULT_YEAR_MAX)
    pattern = cfg["pattern"]
    files = collect_files(sim_name, pattern, y_min, y_max)
    if not files:
        return None

    ds_list = [xr.open_dataset(f, decode_times=False) for f in files]
    ds_concat = xr.concat(ds_list, dim="time_counter") if len(ds_list) > 1 else ds_list[0]
    for ds in ds_list:
        ds.close()

    if var not in ds_concat:
        return None
    arr = ds_concat[var]

    # --- moyenne pondérée sur la dimension "veget" --------------------------
    if "veget" in arr.dims:
        if "VEGET_MAX" in ds_concat:
            weights = ds_concat["VEGET_MAX"]
            # Protection contre division par zéro
            total = weights.sum(dim="veget")
            # Evite RuntimeWarning invalid/NaN
            total = xr.where(total == 0, np.nan, total)
            weights = weights / total
            arr = (arr * weights).sum(dim="veget")
        else:
            print("WARNING: veget_max not reconized or badly written",var )
            # Fallback : moyenne simple si veget_max absent
            arr = arr.mean(dim="veget")

    # --- agrégation temporelle ---------------------------------------------
    if cfg["agg"] == "sum":
        result = arr.sum(dim="time_counter")
    else:
        result = arr.mean(dim="time_counter")
    return result

# -----------------------------------------------------------------------------
# Dataset complet par simulation
# -----------------------------------------------------------------------------

def load_and_aggregate(sim_name: str):
    """Retourne un `xr.Dataset` agrégé pour une simulation."""
    data_vars = {}
    for var, cfg in VARIABLES_CONFIG.items():
        da = aggregate_variable(sim_name, var, cfg)
        if da is not None:
            data_vars[var] = da
        else :
            print("Warning something wrong with ",sim_name, var, cfg)
            print(da)

    return xr.Dataset(data_vars) if data_vars else None

# -----------------------------------------------------------------------------
# Filtrage océan & fusion DataFrames
# -----------------------------------------------------------------------------

def filter_ocean(df: pd.DataFrame):
    """Supprime les lignes correspondant à des pixels oceaniques."""
    if OCEAN_MASK is None:
        return df
    mask_df = OCEAN_MASK.to_dataframe(name="is_ocean").reset_index()
    mask_df["lat"] = mask_df["lat"].astype(df["lat"].dtype)
    mask_df["lon"] = mask_df["lon"].astype(df["lon"].dtype)
    df = df.merge(mask_df, on=["lat", "lon"], how="left")
    return df[df["is_ocean"] == False].drop(columns=["is_ocean"])

# -----------------------------------------------------------------------------
# Programme principal
# -----------------------------------------------------------------------------

def main():
    global OUTPUT_CSV

    if len(sys.argv) < 2:
        print("Usage: python3 aggregate_cost_sep_v4.py <simulation_list.txt> [--output-csv data.csv]")
        sys.exit(1)

    sim_list_file = sys.argv[1]
    if "--output-csv" in sys.argv:
        idx = sys.argv.index("--output-csv")
        if idx + 1 < len(sys.argv):
            OUTPUT_CSV = sys.argv[idx + 1]
        else:
            print("--output-csv exige un chemin de fichier.")
            sys.exit(1)

    # Lecture de la liste des simulations
    with open(sim_list_file) as f:
        sims = [line.strip() for line in f if line.strip()]

    if not sims:
        print("Aucune simulation dans la liste.")
        sys.exit(0)

    all_frames = []
    for sim in sims:
        mgmt, ssp = parse_mgmt_and_ssp(sim)
        ds = load_and_aggregate(sim)
        if ds is None:
            print(f"Warning: pas de données pour {sim}")
            continue
        df = ds.to_dataframe().reset_index()
        df["mgmt"] = mgmt
        df["ssp"] = ssp
        df = filter_ocean(df)
        all_frames.append(df)

    if not all_frames:
        print("Aucune donnée agrégée.")
        sys.exit(0)

    big_df = pd.concat(all_frames, ignore_index=True)

    if OUTPUT_CSV:
        big_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Données agrégées sauvegardées dans {OUTPUT_CSV}")
    else:
        print(big_df.head())

    print("Terminé.")


if __name__ == "__main__":
    main()