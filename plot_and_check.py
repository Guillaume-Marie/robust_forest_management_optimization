#!/usr/bin/env python3
"""
plot_and_check.py
=================

Ce script lit une liste de simulations dans un fichier texte,
charge pour chacune un Dataset NetCDF (concaténation annuelle + agrégation sur lat/lon...),
puis produit un graphe à n panneaux (un par variable),
ET un panneau (subplot) séparé à droite pour la légende.

Inspiré de check_outputs_with_agg.py, 
avec un gridspec pour afficher la légende dans un subplot distinct.

Usage:
  python plot_and_check.py list_of_sims.txt

Exemple de "list_of_sims.txt":
  BAUSSP126
  FM5SSP126
  FM1SSP126

Le script suppose que tu disposes d'un ensemble de fichiers NetCDF
pour chaque simulation, nommés ainsi:
  {simulation}_{year}0101_{year}1231_1Y_stomate_history.nc
entre ANNEE_MIN..ANNEE_MAX.
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
ANNEE_MIN = 2010
ANNEE_MAX = 2099
FILE_PATTERN = "{simulation}_{year}0101_{year}1231_1Y_stomate_history.nc"

# Variables à tracer, et leur mode d'agrégation
VARIABLES_INFO = {
    "NBP_pool_c"   : ("benefit", 1.0),
    "TOTAL_SOIL_c" : ("benefit", 1.0),
    "GPP"          : ("benefit", 1.0),
    "NPP"          : ("benefit", 1.0),    
    "LAI_MAX"      : ("benefit", 1.0),
    "FOREST_MANAGED": ("benefit", 1.0),
    "RDI"          : ("benefit", 1.0)
}
AGGREGATION_MODES = {
    "NBP_pool_c"   : "mean",
    "TOTAL_SOIL_c" : "mean",
    "GPP"          : "mean",
    "NPP"          : "mean",
    "LAI_MAX"      : "mean",
    "FOREST_MANAGED":"max",
    "RDI"          : "mean",
}

# Dimensions spatiales qu'on veut moyenner
SPATIAL_DIMS = ["lat","lon"]
VEGET_ID = 4
TIME_DIM = "time_counter"
BASE_YEAR = ANNEE_MIN

OUTPUT_PNG = "plot_and_check_legend_separated.png"

# ---------------------
# FONCTION D'AGRÉGATION
# ---------------------
def load_and_aggregate_timeseries(sim_name):
    """
    Pour la simulation `sim_name` (ex: "BAUSSP126"), on concatène
    les fichiers annuels de ANNEE_MIN à ANNEE_MAX et on agrège en
    moyennant (ou sommant) les dims spatiales (SPATIAL_DIMS).
    On conserve la dimension TIME_DIM pour tracer l'évolution dans le temps.
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

                # 1) Sélection sur veget
        if "veget" in arr.dims:
            # On suppose que la coord veget va de 1 à 17
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
        data_vars[var_name] = arr_agg

    ds_out = xr.Dataset(data_vars)
    return ds_out

# ---------------------
# SCRIPT PRINCIPAL
# ---------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_and_check.py list_of_sims.txt")
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

    # Liste des variables qu'on veut tracer
    all_vars = list(VARIABLES_INFO.keys())
    nvars = len(all_vars)

    # Création d'une figure avec un gridspec => 2 colonnes :
    #  1) la gauche pour nos n sous-graphiques
    #  2) la droite pour la légende
    fig = plt.figure(figsize=(10, 3*nvars))
    gs = gridspec.GridSpec(nrows=nvars, ncols=2, width_ratios=[4,1])

    axes = []
    for i in range(nvars):
        # Sous-graphe i en colonne 0
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)

    # Axe pour la légende (prend toutes les lignes, mais la 2e colonne)
    legend_ax = fig.add_subplot(gs[:, 1])
    legend_ax.axis('off')  # pas de cadre ni graduation

    # On va stocker tous les handles/labels pour la légende
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

            arr_var = ds_agg[var_name]

            # Récupération du temps (dimension TIME_DIM)
            # Suppose qu'on indexe juste 0..N-1
            N = arr_var.sizes.get(TIME_DIM, 0)
            time_index = np.arange(N)
            # ou si tu veux : years = BASE_YEAR + time_index
            # c'est toi qui vois
            years = BASE_YEAR + time_index

            yvals = arr_var.values
            # Vérifier la cohérence
            if len(years) != len(yvals):
                print(f"Attention, mismatch x={len(years)} y={len(yvals)} pour {sim_name} - {var_name}.")
                continue

            line, = ax.plot(years, yvals, label=sim_name)
        
        ax.set_ylabel(var_name)

    # --------------------------------------------------------------------
    # RÉCUPÉRER LES HANDLES/LABELS DE TOUS LES AXES
    # pour pouvoir afficher UNE légende commune
    # --------------------------------------------------------------------
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)

    # On crée un dict (label -> handle) pour éliminer doublons si besoin
    # (ex: si la même simulation s'affiche sur plusieurs subplots)
    combined = {}
    for h, l in zip(all_handles, all_labels):
        combined[l] = h
    # On re-sépare
    unique_handles = list(combined.values())
    unique_labels  = list(combined.keys())

    # --------------------------------------------------------------------
    # AFFICHE LA LÉGENDE DANS legend_ax
    # --------------------------------------------------------------------
    legend_ax.legend(unique_handles, unique_labels, loc='center')

    axes[-1].set_xlabel("Index du temps (time_counter + BASE_YEAR)")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.close()
    print(f"=> Graphique multipanels + légende séparée : {OUTPUT_PNG}")

# ---------------------
if __name__ == "__main__":
    main()
