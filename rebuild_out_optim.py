"""
NetCDF Processing Script for Simulation Data
------------------------------------------------
This script processes NetCDF files from different simulations, applies a cost function to select the best simulation,
and reconstructs an optimized output file based on the selected simulation data.

Features:
- Reads a list of simulation names from `simulation_input_list.txt`
- Dynamically constructs file paths based on a predefined base directory
- Loads multiple years of NetCDF data per simulation and aggregates them
- Supports both summation (`sum`) and averaging (`mean`) for variable aggregation
- Computes a customizable cost function based on multiple variables
- Selects the best simulation per grid cell based on the minimum cost function value
- Generates two output files:
  - `Fout_CSF_selected.nc`: Contains the selected simulation index per pixel
  - `Fout_rebuilded.nc`: Reconstructed dataset using the best simulation's values

Usage:
  python extract_max_python.py <simulation_input_list.txt>

Configuration:
- `base_path`: Root directory containing all simulation folders
- `variables_utilisees`: List of variables used in the cost function
- `coefficients`: Weights assigned to each variable in the cost function
- `aggregation_modes`: Defines whether each variable is aggregated using `sum` or `mean`
- `annee_min`, `annee_max`: Defines the range of years to process (single year = no aggregation)

"""

import xarray as xr
import numpy as np
import sys
import os
import shutil
from netCDF4 import Dataset
import glob  

# Base path où sont stockées toutes les simulations
base_path = "/home/scratch01/sluys//IGCM_OUT/OL2/TEST/test/"

# Liste des variables utilisées dans la fonction de coût
variables_utilisees = ["NPP", "GPP"]

# Coefficients de pondération pour chaque variable
coefficients = {
    "NPP": 1.0,
    "GPP": 0.5
}

# Modes d'agrégation spécifiques à chaque variable ("sum" ou "mean")
aggregation_modes = {
    "NPP": "sum",
    "GPP": "mean"
}

# Plage d'années à considérer
annee_min = 1701
annee_max = 1701

# Modèle de nom des fichiers NetCDF
file_pattern = "{simulation}_{year}0101_{year}1231_1Y_stomate_history.nc"

# Fonction de coût dynamique
def cost_function(variables):
    """ Fonction de coût combinant plusieurs variables avec des coefficients """
    return sum(coefficients[var] * variables[var] for var in variables_utilisees if var in variables)

# Variables à modifier dans `Fout_rebuilded.nc`
variables_a_modifier = variables_utilisees  

# 🎯Fonction pour supprimer un fichier s'il existe
def supprimer_fichier(fichier):
    if os.path.exists(fichier):
        print(f"🗑️ Suppression de {fichier}...")
        os.remove(fichier)

# Fonction pour trouver et charger les fichiers NetCDF d'une simulation
def charger_et_aggreg_simulation(simulation_name):
    """ Trouve, charge et agrège les fichiers NetCDF d'une simulation sur plusieurs années. """
    
    simulation_path = os.path.join(base_path, simulation_name, "SBG/Output/YE")
    fichiers_selectionnes = []

    if annee_min == annee_max:
        file_name = file_pattern.format(simulation=simulation_name, year=annee_min)
        file_path = os.path.join(simulation_path, file_name)
        if os.path.exists(file_path):
            fichiers_selectionnes.append(file_path)
    else:
        for annee in range(annee_min, annee_max + 1):
            file_name = file_pattern.format(simulation=simulation_name, year=annee)
            file_path = os.path.join(simulation_path, file_name)
            if os.path.exists(file_path):
                fichiers_selectionnes.append(file_path)

    if not fichiers_selectionnes:
        print(f"⚠ Aucun fichier NetCDF trouvé entre {annee_min}-{annee_max} pour {simulation_name}")
        return None

    ds_list = [xr.open_dataset(f) for f in fichiers_selectionnes]

    if annee_min == annee_max:
        return ds_list[0]
    else:
        ds_concat = xr.concat(ds_list, dim="time_counter")  
        ds_aggreg = {var: ds_concat[var].sum(dim="time_counter") if aggregation_modes.get(var, "mean") == "sum" 
                     else ds_concat[var].mean(dim="time_counter") for var in variables_utilisees if var in ds_concat}
        return xr.Dataset(ds_aggreg)

# Etape 1 : Génération de `Fout_CSF_selected.nc`
def create_fout_selected(fichier_liste, output_selected):
    print(f"🟢 Étape 1 : Création de `{output_selected}` avec fonction de coût sur {annee_min}-{annee_max}")

    supprimer_fichier(output_selected)

    with open(fichier_liste, "r") as f:
        simulations = [line.strip() for line in f.readlines()]

    if not simulations:
        print("❌ Erreur : Aucun nom de simulation trouvé dans la liste.")
        sys.exit(1)

    ds_base = charger_et_aggreg_simulation(simulations[0])
    simulation_dict = {sim: i+1 for i, sim in enumerate(simulations)}

    csf_selected = xr.DataArray(
        np.ones((ds_base.dims["lat"], ds_base.dims["lon"]), dtype=int),
        coords={"lat": ds_base["lat"], "lon": ds_base["lon"]},
        dims=["lat", "lon"]
    )

    cost_map = xr.full_like(ds_base["NPP"].isel(veget=0), fill_value=np.inf)

    for sim in simulations:
        simulation_id = simulation_dict[sim]

        print(f"  📂 Traitement de la simulation : {sim} (ID: {simulation_id})")

        ds = charger_et_aggreg_simulation(sim)
        if ds is None:
            continue

        if any(var not in ds for var in variables_utilisees):
            print(f"⚠ Variables manquantes dans {sim}, passage à la suivante.")
            continue

        variables_data = {var: ds[var].mean(dim="veget") for var in variables_utilisees}
        cost_value = cost_function(variables_data)

        mask = cost_value < cost_map
        cost_map = xr.where(mask, cost_value, cost_map)
        csf_selected = xr.where(mask, simulation_id, csf_selected)

        ds.close()

    ds_selected = xr.Dataset({"CSF_selected": csf_selected})
    ds_selected.to_netcdf(output_selected)
    ds_selected.close()

    print(f"✅ Fichier généré : {output_selected}")

    return simulations, output_selected, simulation_dict

# Etape 2 : Génération de `Fout_rebuilded.nc`
def create_fout_rebuilded(simulations, fichier_selected, output_rebuilded, simulation_dict):
    print(f"🟢 Étape 2 : Création de `{output_rebuilded}`")

    supprimer_fichier(output_rebuilded)

    shutil.copy(glob.glob(os.path.join(base_path, simulations[0], "SBG/Output/YE", "*.nc"))[0], output_rebuilded)

    ds_selected = xr.open_dataset(fichier_selected)
    csf_selected = ds_selected["CSF_selected"]

    with Dataset(output_rebuilded, mode="a") as ds_rebuilded:
        for sim in simulations[1:]:
            simulation_id = simulation_dict.get(sim, -1)

            if simulation_id not in np.unique(csf_selected.values):
                print(f"⚠ Simulation {sim} ignorée.")
                continue

            print(f"  📂 Application des valeurs depuis : {sim}")

            ds = charger_et_aggreg_simulation(sim)
            if ds is None:
                continue

            base_mask = (csf_selected == simulation_id).values

            for var in variables_a_modifier:
                if var in ds.variables and var in ds_rebuilded.variables and ds[var].shape == ds_rebuilded[var].shape:
                    mask = np.broadcast_to(base_mask, ds[var].shape)
                    np.copyto(ds_rebuilded[var][:], ds[var].values, where=mask)

            ds.close()

    ds_selected.close()
    print(f"✅ Fichier généré : {output_rebuilded}")

# Exécution principale
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_max_python.py <fichier_liste>")
        sys.exit(1)

    fichier_liste = sys.argv[1]

    output_selected = f"Fout_CSF_selected_cost_{annee_min}-{annee_max}.nc"
    output_rebuilded = f"Fout_rebuilded_cost_{annee_min}-{annee_max}.nc"

    simulations, fichier_selected, simulation_dict = create_fout_selected(fichier_liste, output_selected)
    create_fout_rebuilded(simulations, fichier_selected, output_rebuilded, simulation_dict)

    print("🎯 Tout est terminé avec succès !")

