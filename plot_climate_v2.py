#!/usr/bin/env python3
"""
plot_climate_v2.py – spatial aggregation by climate regions *or* global
====================================================================

Mise à jour :
  - Support de deux types de fichiers (stomate, sechiba) via un template `FILE_PATTERN_TEMPLATE`.
  - Fusion de VARIABLES_INFO et TEMPORAL_MODES en un unique `VARIABLES_CONFIG` avec un champ `mode` et un champ `pattern`.
  - Chargement et fusion par année des fichiers de chaque pattern avant concaténation.

Usage :
  python plot_climate_v2.py list_of_sims.txt [--agg=regions|global]

Requirements: geopandas, shapely, rioxarray, matplotlib, pandas, xarray, numpy.
"""
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import re

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SUBTRT_COLOR_DICT = {
    "PN"        : "green",
    "NDIACUT"   : "red",
    "PDIACUT"   : "orange",
    "PPRESIDUAL": "grey",
    "PRESIDUAL" : "blue",
    "NTHININT"  : "purple",
    "PTHININT"  : "pink",
    ""          : "black",
}

BASE_PATH         = "/home/mguill/Documents/output_CSF_v2/subset"
ANNEE_MIN         = 2030
ANNEE_MAX         = 2099
FILE_PATTERN_TEMPLATE = "{simulation}_{year}0101_{year}1231_1Y_{pattern}_history.nc"

SPATIAL_DIMS      = ["lat", "lon"]
VEGET_ID          = 4  # ID de PFT/VEGET à sélectionner
TIME_DIM          = "time_counter"
SSP               = "SSP370"

# Fusion de VARIABLES_INFO et TEMPORAL_MODES, avec pattern
VARIABLES_CONFIG = {
    "TOTAL_SOIL_c":       {"mode":"none",   "pattern":"stomate"},
    "mineral_soil_C":     {"mode":"none",   "pattern":"sechiba"},
    "NBP_pool_c":         {"mode":"cumsum", "pattern":"stomate"},
    "NPP":                {"mode":"cumsum", "pattern":"stomate"},
    "HARVEST_FOREST_c":   {"mode":"cumsum", "pattern":"stomate"},
    "TOTAL_BM_LITTER_c":  {"mode":"cumsum", "pattern":"stomate"},
    "TOTAL_M_c":          {"mode":"none",   "pattern":"stomate"},
    "ALPHA_SELF_THINNING":{"mode":"none",   "pattern":"stomate"},
    "KF":                 {"mode":"none",   "pattern":"stomate"},
    "SAP_M_AB_c":         {"mode":"none",   "pattern":"stomate"},
    "WOODMASS_IND":       {"mode":"none",   "pattern":"stomate"},
    "CO2_TAKEN":          {"mode":"cumsum", "pattern":"stomate"},
    "IND":                {"mode":"none",   "pattern":"stomate"},
    "RDI":                {"mode":"none",   "pattern":"stomate"},
    "LAI_MAX":            {"mode":"none",   "pattern":"stomate"},
    "HET_RESP":           {"mode":"cumsum", "pattern":"stomate"},
    "WSTRESS_SEASON":     {"mode":"cumsum", "pattern":"stomate"},
    "netco2flux":         {"mode":"cumsum", "pattern":"sechiba"},
    "humtot":             {"mode":"none",   "pattern":"sechiba"},
    "evapot":             {"mode":"none",   "pattern":"sechiba"},
    "transpot":           {"mode":"none",   "pattern":"sechiba"},
    "diff_transpir":      {"mode":"cumsum",   "pattern":"sechiba"},
    "range_tsol":         {"mode":"none",   "pattern":"sechiba"},
    "WaterTableD":        {"mode":"none",   "pattern":"sechiba"}
}

SHAPEFILE         = "enz_v8_4326.shp"
REGION_COL        = "EnZ_name"

# -----------------------------------------------------------------------------
# Fonctions utilitaires
# -----------------------------------------------------------------------------
def parse_sub_treatment(sim_name: str) -> str:
    m = re.match(r"^(BAU|FM\d+)(.*)(SSP\d+)$", sim_name.upper())
    return "UNKNOWN" if not m else m.group(2)


def get_linecolor(sim_name: str) -> str:
    return SUBTRT_COLOR_DICT.get(parse_sub_treatment(sim_name).upper(), "black")


def get_linestyle(sim_name: str) -> str:
    s = sim_name.upper()
    if "BAU" in s:
        return "-"
    if "FM1" in s:
        return "--"
    if "FM5" in s:
        return ":"
    return "-"

# -----------------------------------------------------------------------------
# Chargement et concaténation multi-pattern par année
# -----------------------------------------------------------------------------
def load_concat_dataset(sim_name: str):
    """
    Pour chaque année, charge les fichiers correspondant à chaque pattern,
    les fusionne, puis concatène tous les DS annuels le long de TIME_DIM.
    Retourne un Dataset avec tous les variables de VARIABLES_CONFIG.
    """
    # Identifier patterns utilisées
    patterns = sorted({cfg['pattern'] for cfg in VARIABLES_CONFIG.values()})
    annual_ds = []
    for year in range(ANNEE_MIN, ANNEE_MAX + 1):
        ds_list = []
        for pat in patterns:
            fname = FILE_PATTERN_TEMPLATE.format(simulation=sim_name, year=year, pattern=pat)
            fpath = os.path.join(BASE_PATH, fname)
            if os.path.exists(fpath):
                ds_list.append(xr.open_dataset(fpath, decode_times=False))
        if not ds_list:
            continue
        if len(ds_list) > 1:
            ds_year = xr.merge(ds_list)
            for ds in ds_list:
                ds.close()
        else:
            ds_year = ds_list[0]
        annual_ds.append(ds_year)
    if not annual_ds:
        return None
    ds_full = xr.concat(annual_ds, dim=TIME_DIM)
    ds_full = ds_full.rio.write_crs("EPSG:4326", inplace=True)
    return ds_full


def clip_and_aggregate_timeseries(ds, var, polygons_gdf):
    if var not in ds:
        return None
    arr = ds[var]
    if "veget" in arr.dims:
        arr = arr.sel(veget=VEGET_ID)

    nt = ds.sizes[TIME_DIM]
    regions = sorted(polygons_gdf[REGION_COL].unique())
    out = pd.DataFrame(index=range(nt), columns=regions)

    for reg in regions:
        geom = polygons_gdf[polygons_gdf[REGION_COL] == reg].unary_union
        ds_clip = ds.rio.clip([geom.__geo_interface__], crs=ds.rio.crs, drop=True)
        if var not in ds_clip or ds_clip[var].size == 0:
            out[reg] = np.nan
        else:
            arr_clip = ds_clip[var]
            if "veget" in arr_clip.dims:
                arr_clip = arr_clip.sel(veget=VEGET_ID)
            out[reg] = arr_clip.mean(dim=SPATIAL_DIMS, skipna=True).values
    return out


def global_aggregate_timeseries(ds, var):
    if var not in ds:
        return None
    arr = ds[var]
    if "veget" in arr.dims:
        arr = arr.sel(veget=VEGET_ID)
    return pd.DataFrame({"GLOBAL": arr.mean(dim=SPATIAL_DIMS, skipna=True).values})

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} list_of_sims.txt [--agg=regions|global]")
        sys.exit(1)

    list_file = sys.argv[1]
    if not os.path.exists(list_file):
        print("Fichier liste inexistant.")
        sys.exit(1)

    agg_mode = "regions"
    if len(sys.argv) >= 3 and sys.argv[2].startswith("--agg="):
        agg_mode = sys.argv[2].split("=", 1)[1].lower()
        if agg_mode not in ("regions", "global"):
            print("--agg doit être 'regions' ou 'global'.")
            sys.exit(1)

    output_pdf = f"plot_climate_{agg_mode.upper()}_PFT{VEGET_ID}_{ANNEE_MIN}_{ANNEE_MAX}_{SSP}.pdf"

    with open(list_file) as f:
        sims = [l.strip() for l in f if l.strip()]
    if not sims:
        print("Liste vide.")
        sys.exit(1)

    polygons = None
    if agg_mode == "regions":
        if not os.path.exists(SHAPEFILE):
            print("Shapefile manquant.")
            sys.exit(1)
        polygons = gpd.read_file(SHAPEFILE)
        polygons = polygons.to_crs("EPSG:4326") if polygons.crs else polygons.set_crs("EPSG:4326", inplace=True)

    # Extraction des données
    scenario_data = {}
    for sim in sims:
        ds = load_concat_dataset(sim)
        if ds is None:
            print(f"Aucune donnée pour {sim}")
            continue
        scenario_data[sim] = {}
        for var, cfg in VARIABLES_CONFIG.items():
            if agg_mode == "regions":
                df = clip_and_aggregate_timeseries(ds, var, polygons)
            else:
                df = global_aggregate_timeseries(ds, var)
            if df is not None:
                # Appliquer mode temporel (cumsum ou none)
                if cfg['mode'] == 'cumsum':
                    df = df.cumsum()
                scenario_data[sim][var] = df

    if not scenario_data:
        print("Pas de simulations exploitables.")
        sys.exit(1)

    years = np.arange(ANNEE_MIN, ANNEE_MAX + 1)

    # Création du PDF
    with PdfPages(output_pdf) as pdf:
        for var in VARIABLES_CONFIG:
            # Détermination des régions ou GLOBAL
            if agg_mode == "regions":
                region_vals = sorted(polygons[REGION_COL].unique())
                ncols, nrows = 3, (len(region_vals) + 2) // 3
                fig = plt.figure(figsize=(10 * ncols, 3 * nrows))
                gs = gridspec.GridSpec(nrows, ncols + 1,
                                       width_ratios=[4] * ncols + [1],
                                       wspace=0.2, hspace=0.4)
                legend_ax = fig.add_subplot(gs[:, -1]); legend_ax.axis("off")
                axes = [fig.add_subplot(gs[r, c]) for r in range(nrows) for c in range(ncols)][:len(region_vals)]
            else:
                region_vals = ["GLOBAL"]
                fig, ax = plt.subplots(figsize=(16, 10))
                axes = [ax]
                legend_ax = None

            fig.suptitle(f"{var} (VEGET={VEGET_ID}) – {agg_mode.upper()}")
            handles = {}
            for idx, reg in enumerate(region_vals):
                ax = axes[idx]
                if agg_mode == "regions":
                    ax.set_title(f"Région: {reg}")
                for sim, data_vars in scenario_data.items():
                    df = data_vars.get(var)
                    if df is None or reg not in df.columns:
                        continue
                    y = df[reg].values if agg_mode == "regions" else df["GLOBAL"].values
                    line, = ax.plot(years, y,
                                    linestyle=get_linestyle(sim),
                                    color=get_linecolor(sim),
                                    label=sim)
                    handles[sim] = line
                ax.set_xlabel("Année")
                ax.set_ylabel(var)

            # Masquer axes non utilisés
            if agg_mode == "regions":
                for extra in axes[len(region_vals):]:
                    extra.set_visible(False)
                legend_ax.legend(handles.values(), handles.keys(), loc="center")
            else:
                fig.legend(handles.values(), handles.keys(), loc="lower center",
                           ncol=min(3, len(handles)), bbox_to_anchor=(0.5, -0.05))
                fig.subplots_adjust(bottom=0.25)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"=> PDF généré : {output_pdf}")

if __name__ == "__main__":
    main()
