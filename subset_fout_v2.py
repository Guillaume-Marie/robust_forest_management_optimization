#!/usr/bin/env python3
"""
subset_fout_v2.py
================

Mise à jour :
  - Support de deux types de fichiers (stomate, sechiba) via FILE_PATTERN_TEMPLATE.
  - Passage de la liste variables_to_extract à un dictionnaire VARIABLES_CONFIG incluant le champ 'pattern'.

Usage :
  python3 subset_fout_v2.py
"""
import os
import sys
import xarray as xr

# Dossier où se trouvent les fichiers NetCDF
input_dir = "/home/mguill/Documents/output_CSF_v2"

# Dossier de sortie (peut être identique à input_dir si souhaité)
output_dir = "/home/mguill/Documents/output_CSF_v2/subset"

# -----------------------------------------------------------------------------
# Configuration : variables et pattern de fichier associés
# -----------------------------------------------------------------------------
# Les variables listées ici seront extraites uniquement des fichiers dont le pattern correspond.
VARIABLES_CONFIG = {
    # stomate
    "TOTAL_SOIL_c":       {"pattern":"stomate"},
    "NBP_pool_c":         {"pattern":"stomate"},
    "VEGET_MAX":          {"pattern":"stomate"},   
    "NPP":                {"pattern":"stomate"},
    "HARVEST_FOREST_c":   {"pattern":"stomate"},
    "TOTAL_BM_LITTER_c":  {"pattern":"stomate"},
    "TOTAL_M_c":          {"pattern":"stomate"},
    "ALPHA_SELF_THINNING":{"pattern":"stomate"},
    "KF":                 {"pattern":"stomate"},
    "SAP_M_AB_c":         {"pattern":"stomate"},
    "WOODMASS_IND":       {"pattern":"stomate"},
    "IND":                {"pattern":"stomate"},
    "RDI":                {"pattern":"stomate"},
    "LAI_MAX":            {"pattern":"stomate"},
    "HET_RESP":           {"pattern":"stomate"},
    "WSTRESS_SEASON":     {"pattern":"stomate"},
    # sechiba 
    "netco2flux":         {"pattern":"sechiba"},
    "humtot":             {"pattern":"sechiba"},
    "tsol_max":           {"pattern":"sechiba"},
    "tsol_min":           {"pattern":"sechiba"},
    "temp_sol":           {"pattern":"sechiba"},
    "alb_vis":            {"pattern":"sechiba"},
    "transpir":           {"pattern":"sechiba"},
    "evap":               {"pattern":"sechiba"},
    "evapot":             {"pattern":"sechiba"},
    "transpot":           {"pattern":"sechiba"}
}

# Définir ici les opérations arithmétiques simples à appliquer après extraction
# Format : "nouveau_nom": {"vars": ["var1", "var2"], "op": "+" | "-" | "*" | "/" }
ARITHMETIC_CONFIG = {
    "diff_transpir": {
        "vars": ["transpir", "transpot"],
        "op": "-",
        "pattern": "sechiba"
    },
    "range_tsol": {
        "vars": ["tsol_max", "tsol_min"],
        "op": "-",
        "pattern": "sechiba"
    },
    # Exemple d'opération stomate
    "mineral_soil_C": {
        "vars": ["TOTAL_SOIL_c", "TOTAL_BM_LITTER_c"],
        "op": "-",
        "pattern": "stomate"
    },
    # Ajoutez ici vos propres formules avec le champ "pattern"
}

# Template pour détecter les fichiers à traiter
FILE_PATTERN_TEMPLATE = "{simulation}_{year}0101_{year}1231_1Y_{pattern}_history.nc"
PATTERNS = sorted({cfg['pattern'] for cfg in VARIABLES_CONFIG.values()})

# -----------------------------------------------------------------------------
# Extraction et création de variables dérivées
# -----------------------------------------------------------------------------

def extraire_variables_netcdf(input_dir, output_dir):
    """
    Parcourt les fichiers *_{pattern}_history.nc, extrait les variables
    définies dans VARIABLES_CONFIG pour chaque pattern,
    puis calcule les nouvelles variables arithmétiques limitativement au même pattern.
    """
    if not os.path.isdir(input_dir):
        print(f"Erreur: input_dir '{input_dir}' invalide.")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        for pattern in PATTERNS:
            suffix = f"_{pattern}_history.nc"
            if not fname.endswith(suffix):
                continue

            input_path = os.path.join(input_dir, fname)
            print(f"Chargement : {input_path}")
            ds = xr.open_dataset(input_path, decode_times=False)

            # Sélection des variables extraites pour ce pattern
            vars_to_extract = [var for var, cfg in VARIABLES_CONFIG.items()
                               if cfg['pattern'] == pattern and var in ds]
            if not vars_to_extract:
                print(f"  > Aucune variable à extraire pour le pattern '{pattern}'.")
                ds.close()
                break

            ds_subset = ds[vars_to_extract].copy()

            # Calcul des variables dérivées pour ce pattern
            for new_var, cfg in ARITHMETIC_CONFIG.items():
                if cfg.get('pattern') != pattern:
                    continue  # n’applique que sur le bon pattern
                v1, v2 = cfg['vars']
                op = cfg['op']
                if v1 in ds_subset and v2 in ds_subset:
                    if op == '+':
                        ds_subset[new_var] = ds_subset[v1] + ds_subset[v2]
                    elif op == '-':
                        ds_subset[new_var] = ds_subset[v1] - ds_subset[v2]
                    elif op == '*':
                        ds_subset[new_var] = ds_subset[v1] * ds_subset[v2]
                    elif op == '/':
                        ds_subset[new_var] = ds_subset[v1] / ds_subset[v2]
                    else:
                        print(f"  ! Opérateur '{op}' non supporté pour '{new_var}'.")
                        continue
                    print(f"  > Création de '{new_var}' = {v1} {op} {v2}")
                else:
                    print(f"  ! Impos. de créer '{new_var}': variables {v1}, {v2} absentes.")

            # Sauvegarde du subset modifié
            output_path = os.path.join(output_dir, fname)
            print(f"  > Écriture du subset vers {output_path}")
            ds_subset.to_netcdf(output_path)

            ds_subset.close()
            ds.close()
            break

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    extraire_variables_netcdf(input_dir, output_dir)
    print("Extraction et création des variables terminée !")

if __name__ == "__main__":
    main()
