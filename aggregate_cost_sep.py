#!/usr/bin/env python3
"""
aggregate_cost_sep.py
=====================

Nouvelle version avec normalisation [0,1] des variables avant sauvegarde.

Usage:
  python aggregate_cost_sep.py <simulation_list.txt> [--output-csv data.csv]
"""

import xarray as xr
import numpy as np
import pandas as pd
import sys
import os
import re

BASE_PATH   = "/home/mguill/Documents/output_CSF_v2/subset"  # répertoire des NetCDF
YEAR_MIN    = 2039
YEAR_MAX    = 2099
FILE_PATTERN = "{simulation}_{year}0101_{year}1231_1Y_stomate_history.nc"

# Variables à extraire et leur mode d'agrégation
VARIABLES_INFO = {
    "TOTAL_SOIL_c": "mean",
    "NBP_pool_c"   : "sum",
    "NPP"          : "sum",
    # Ajoutez ici d'autres variables si besoin
}

def parse_mgmt_and_ssp(sim_name):
    """
    Pour un nom ex: 'BAUSSP126', renvoie (mgmt='BAU', ssp='SSP126').
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
    Concatène les fichiers NetCDF de YEAR_MIN..YEAR_MAX pour la simulation `sim_name`.
    Retourne un xarray.Dataset agrégé suivant VARIABLES_INFO.
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
    for var_name, agg_mode in VARIABLES_INFO.items():
        if var_name not in ds_concat:
            continue
        arr = ds_concat[var_name]
        # Si dimension veget -> on moyenner par ex.
        if "veget" in arr.dims:
            arr = arr.mean(dim="veget")
        # Agrégation temporelle
        if agg_mode == "sum":
            arr = arr.sum(dim="time_counter")
        else:
            arr = arr.mean(dim="time_counter")
        data_vars[var_name] = arr

    return xr.Dataset(data_vars)


def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_cost_sep.py <simulation_list.txt> [--output-csv data.csv]")
        sys.exit(1)

    sim_list_file = sys.argv[1]
    output_csv = None
    if len(sys.argv) >= 4 and sys.argv[2] == "--output-csv":
        output_csv = sys.argv[3]
    elif len(sys.argv) == 3 and sys.argv[2].startswith("--output-csv"):
        output_csv = sys.argv[2].split("=")[1]

    with open(sim_list_file, "r") as f:
        sims = [line.strip() for line in f if line.strip()]

    all_dfs = []
    # On charge toutes les simulations et on concatène
    for sim_name in sims:
        mgmt_name, ssp = parse_mgmt_and_ssp(sim_name)
        ds_agg = load_and_aggregate(sim_name)
        if ds_agg is None:
            print(f"Warning: no files found for simulation {sim_name}")
            continue

        df = ds_agg.to_dataframe().reset_index()
        df["mgmt"] = mgmt_name
        df["ssp"]  = ssp
        all_dfs.append(df)

    if not all_dfs:
        print("No data found for any simulation. Exiting.")
        sys.exit(0)

    big_df = pd.concat(all_dfs, axis=0, ignore_index=True)

    # -------------------------------
    # Étape de NORMALISATION 0..1
    # -------------------------------
    # Pour chaque variable de VARIABLES_INFO, on calcule min et max
    for var_name in VARIABLES_INFO.keys():
        if var_name not in big_df.columns:
            continue
        min_val = big_df[var_name].min()
        max_val = big_df[var_name].max()
        if np.isclose(min_val, max_val):
            # Si aucune variation, on force la valeur à 0 (ou 0.5, selon convenance)
            big_df[var_name] = 0.0
        else:
            big_df[var_name] = (big_df[var_name] - min_val) / (max_val - min_val)

    # Export CSV si demandé
    if output_csv:
        big_df.to_csv(output_csv, index=False)
        print(f"Saved normalized aggregated data to {output_csv}")

    # Export NetCDF
    ds_final = big_df.set_index(["mgmt","ssp","lat","lon"]).to_xarray()
    ds_final.to_netcdf("multicriteria_data_normalized.nc")
    print("Saved normalized data to multicriteria_data_normalized.nc")


if __name__ == "__main__":
    main()
