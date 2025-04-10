#!/usr/bin/env python3
"""
plot_and_check_v2.py
====================

Version modifiée pour différencier certains types d'expériences (BAU, FM1, FM5)
en utilisant des styles de lignes distincts.

Usage:
  python plot_and_check_v2.py list_of_sims.txt
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os

# ---------------------
# PARAMÈTRES GLOBAUX
# ---------------------

BASE_PATH = "/home/mguill/Documents/output_CSF_v2"
ANNEE_MIN = 2030
ANNEE_MAX = 2080
FILE_PATTERN = "{simulation}_{year}0101_{year}1231_1Y_stomate_history.nc"

# Variables à tracer, et leur mode d'agrégation
VARIABLES_INFO = {
    "NBP_pool_c"   : ("benefit", 1.0),
    "TOTAL_SOIL_c" : ("benefit", 1.0),
    "GPP"          : ("benefit", 1.0),
    "NPP"          : ("benefit", 1.0),    
    "LAI_MAX"      : ("benefit", 1.0),
    "RDI"          : ("benefit", 1.0)
}
AGGREGATION_MODES = {
    "NBP_pool_c"   : "mean",
    "TOTAL_SOIL_c" : "mean",
    "GPP"          : "mean",
    "NPP"          : "mean",
    "LAI_MAX"      : "mean",
    "RDI"          : "mean",
}

TEMPORAL_MODES = {
    "NBP_pool_c"   : "cumsum",
    "TOTAL_SOIL_c" : "none",
    "GPP"          : "cumsum",
    "NPP"          : "cumsum",
    "LAI_MAX"      : "none",
    "RDI"          : "none",
}

SPATIAL_DIMS = ["lat","lon"]
VEGET_ID = 8
TIME_DIM = "time_counter"
BASE_YEAR = ANNEE_MIN

OUTPUT_PNG = (
    "plot_and_check_legend_PFT"
    + str(VEGET_ID) + "_"
    + str(ANNEE_MIN) + "_"
    + str(ANNEE_MAX)
    + ".png"
)

# ---------------------
# NOUVELLE FONCTION
# ---------------------
def get_line_style(sim_name):
    """
    Détermine le style de ligne en fonction du nom de la simulation.
    Vous pouvez ajuster les conditions et les styles ci-dessous
    (solid='-', dashed='--', dotted=':', dashdot='-.').
    """
    name_upper = sim_name.upper()
    if "BAU" in name_upper:
        return "-"   # trait plein
    elif "FM1" in name_upper:
        return "--"  # tirets
    elif "FM5" in name_upper:
        return ":"   # pointillés
    else:
        return "-."  # tirets-pointillés (par défaut pour les autres)

# ---------------------
def load_and_aggregate_timeseries(sim_name):
    """
    Pour la simulation `sim_name` (ex: "BAUSSP126"), on concatène
    les fichiers annuels de ANNEE_MIN à ANNEE_MAX et on agrège en
    moyennant (ou sommant) les dims spatiales (SPATIAL_DIMS).
    On conserve la dimension TIME_DIM pour tracer l’évolution dans le temps.
    Retourne un Dataset contenant chaque variable de VARIABLES_INFO
    sous forme 1D (time_counter).
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
    ds_concat = xr.concat(ds_list, dim=TIME_DIM)
    for ds in ds_list:
        ds.close()

    data_vars = {}
    for var_name, (var_type, weight) in VARIABLES_INFO.items():
        if var_name not in ds_concat:
            continue
        arr = ds_concat[var_name]

        # Sélection sur veget, si nécessaire
        if "veget" in arr.dims:
            arr = arr.sel(veget=VEGET_ID)

        # Agrégation spatiale
        agg_mode = AGGREGATION_MODES.get(var_name, "mean")
        dims_to_agg = [d for d in SPATIAL_DIMS if d in arr.dims]

        if agg_mode == "sum":
            arr_agg = arr.sum(dim=dims_to_agg)
        elif agg_mode == "mean":
            arr_agg = arr.mean(dim=dims_to_agg)
        elif agg_mode == "max":
            arr_agg = arr.max(dim=dims_to_agg)
        elif agg_mode == "min":
            arr_agg = arr.min(dim=dims_to_agg)
        else:
            arr_agg = arr.mean(dim=dims_to_agg)

        data_vars[var_name] = arr_agg

    ds_out = xr.Dataset(data_vars)
    return ds_out

# ---------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_and_check_v2.py list_of_sims.txt")
        sys.exit(1)

    list_file = sys.argv[1]
    if not os.path.exists(list_file):
        print(f"Erreur : {list_file} n’existe pas.")
        sys.exit(1)

    # Lecture de la liste de noms de simulation
    with open(list_file, "r") as f:
        sims = [line.strip() for line in f if line.strip()]

    scenario_datasets = {}
    for sim_name in sims:
        ds_agg = load_and_aggregate_timeseries(sim_name)
        if ds_agg is not None:
            scenario_datasets[sim_name] = ds_agg
        else:
            print(f"  Aucune donnée pour la simulation '{sim_name}' !")

    if not scenario_datasets:
        print("Aucune simulation valide chargée. Abandon.")
        sys.exit(1)

    # Liste des variables qu’on veut tracer
    all_vars = list(VARIABLES_INFO.keys())
    nvars = len(all_vars)

    # Création d’une figure avec un gridspec => 2 colonnes :
    #  1) la gauche pour nos n sous-graphiques
    #  2) la droite pour la légende
    fig = plt.figure(figsize=(10, 3*nvars))
    gs = gridspec.GridSpec(nrows=nvars, ncols=2, width_ratios=[4,1])

    axes = []
    for i in range(nvars):
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)

    # Axe pour la légende (prend toutes les lignes, mais la 2e colonne)
    legend_ax = fig.add_subplot(gs[:, 1])
    legend_ax.axis('off')  # pas de cadre ni graduation

    all_handles = []
    all_labels = []

    # --------------------------------------------------------------------
    # On parcourt les variables => chaque variable -> un subplot
    # --------------------------------------------------------------------
    for i, var_name in enumerate(all_vars):
        ax = axes[i]
        ax.set_title(var_name)

        # Pour chaque simulation => tracer la courbe
        for sim_name, ds_agg in scenario_datasets.items():
            if var_name not in ds_agg:
                continue

            arr_var = ds_agg[var_name].values
            agg_mode = TEMPORAL_MODES.get(var_name, "none")
            N = ds_agg[var_name].sizes.get(TIME_DIM, 0)
            time_index = np.arange(N)
            years = BASE_YEAR + time_index

            # Calcul cumsum si besoin
            if agg_mode == "cumsum":
                arr_var = np.cumsum(arr_var)

            # Choix du style de ligne
            ls = get_line_style(sim_name)

            line, = ax.plot(years, arr_var, label=sim_name, linestyle=ls)
        
        ax.set_ylabel(var_name)

    # --------------------------------------------------------------------
    # Récupérer les handles/labels pour la légende commune
    # --------------------------------------------------------------------
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)

    # Éventuellement on peut supprimer les doublons
    combined = {}
    for h, l in zip(all_handles, all_labels):
        combined[l] = h
    unique_handles = list(combined.values())
    unique_labels  = list(combined.keys())

    # Légende unique
    legend_ax.legend(unique_handles, unique_labels, loc='center')

    axes[-1].set_xlabel("Années (BASE_YEAR + index)")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.close()
    print(f"=> Graphique multipanels + légende séparée : {OUTPUT_PNG}")

# ---------------------
if __name__ == "__main__":
    main()
