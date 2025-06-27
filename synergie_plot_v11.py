#!/usr/bin/env python3
"""
synergie_plot_v15.py – Impact d’une pratique optimale unique
===========================================================

* **Contrôle** : impression du *top‑4* des pratiques optimales (en %).
* **Plot** : barre + nom en gras pour la variable optimisée, agrégation
  par SSP ou moyenne SSP.
* **Tables** : export CSV **et** Markdown (colonne optimisée en gras).
* **Robuste** : tous les tests d’erreurs et chemins de sortie créés au
  besoin.

Usage minimal
-------------
```bash
python synergie_plot_v11.py \
  --input Fout_agregg.csv \
  --vars HARVEST_FOREST_c NPP WSTRESS_SEASON \
  --practice-file Fout_CSF_HARVEST_maximin.csv \
  --practice-var CSF_selected \
  --id-map-file scenario_id_map.csv \
  --optim-var HARVEST_FOREST_c
```
"""

################################################################################
# IMPORTS & CONSTANTES
################################################################################
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import cm

# Réglage matplotlib par défaut (polices lisibles)
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

################################################################################
# OUTILS DIVERS
################################################################################

def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI."""
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Calcule l’impact (Δ% vs BAU) d’une pratique optimale unique – "
            "choisie pour une variable cible – sur divers indicateurs. "
            "Sort: barplot groupé, table CSV + Markdown et stats best‑practice."
        ),
    )

    # --- Fichiers / entrées ---
    p.add_argument("--input", required=True, help="CSV agrégé (pixels × variables)")
    p.add_argument("--practice-file", required=True, help="NetCDF best‑practice (IDs)")
    p.add_argument("--practice-var", default="CSF_selected", help="Nom de la variable NetCDF")

    # --- Variables & identifiants ---
    p.add_argument("--vars", nargs="+", required=True, help="Variables à analyser")
    p.add_argument("--optim-var", required=True, help="Variable ayant servi à optimiser la pratique")
    p.add_argument("--group", nargs="+", default=["lat", "lon"], help="Colonnes identifiant un pixel")
    p.add_argument("--ssp-col", default="ssp", help="Colonne SSP dans le CSV")
    p.add_argument("--practice-col", default="mgmt", help="Colonne pratique dans le CSV")
    p.add_argument("--bau-col", default="mgmt", help="Colonne marquant le BAU dans le CSV")
    p.add_argument("--bau-val", default="BAU", help="Valeur identifiant le BAU")

    # --- Mode / sorties ---
    p.add_argument("--mode", choices=["per-ssp", "robust"], default="per-ssp", help="Agrégation des SSP")
    p.add_argument("--table-out", default=None, help="Chemin du CSV Δ %; défaut = dossier de sortie")
    p.add_argument("--outdir", default="rowplots_delta_ssp", help="Dossier pour les figures & tables")

    return p.parse_args()

def load_best_practice(path: str | Path, varname: str) -> pd.DataFrame:
    """Charge le fichier des meilleures pratiques (CSV ou NetCDF)."""
    if str(path).endswith(".csv"):
        df = pd.read_csv(path)
        if "lat" not in df or "lon" not in df or varname not in df:
            sys.exit(f"❌ Le fichier CSV {path} doit contenir 'lat', 'lon' et '{varname}'")
        return df.rename(columns={varname: "BEST_PRACTICE_ID"})[["lat", "lon", "BEST_PRACTICE_ID"]]
    else:
        ds = xr.open_dataset(path)
        if varname not in ds:
            sys.exit(f"❌ Variable {varname} absente dans {path}")
        return ds[varname].to_dataframe(name="BEST_PRACTICE_ID").reset_index()

def load_mapping(path: str | Path, id_col: str, name_col: str) -> Dict[int, str]:
    df = pd.read_csv(path)
    if id_col not in df.columns or name_col not in df.columns:
        sys.exit("❌ Le mapping doit contenir les colonnes indiquées")
    return dict(zip(df[id_col], df[name_col]))


def delta_percent(opt: float, bau: float) -> float:
    """Δ % vs BAU (NaN si BAU = 0)."""
    return (opt - bau) / bau * 100.0 if bau != 0 else np.nan

################################################################################
# FONCTIONS PRINCIPALES
################################################################################

def compute_deltas(
    df: pd.DataFrame,
    best_df: pd.DataFrame,
    args: argparse.Namespace,
    var_order: List[str],
) -> Dict[str, tuple[pd.Series, pd.Series]]:
    """
    Retourne un dict {SSP: (mean_series, std_series)} :
      • *mean_series* : Δ % moyen par variable
      • *std_series*  : écart-type inter-pixels du Δ %
    """
    results: Dict[str, tuple[pd.Series, pd.Series]] = {}
    ssp_values = sorted(df[args.ssp_col].unique())

    for ssp in ssp_values:
        sub = df[df[args.ssp_col] == ssp]
        merged = pd.merge(sub, best_df, on=args.group, how="inner")

        # Sélection des lignes optimales / BAU
        match_simple = merged[args.practice_col] == merged["BEST_PRACTICE"]
        match_combo  = merged[args.practice_col] + "_" + merged[args.ssp_col] == merged["BEST_PRACTICE"]
        opt_rows = merged[match_simple | match_combo]
        bau_rows = sub[sub[args.bau_col] == args.bau_val]

        # Moyennes par pixel
        opt_pix = opt_rows.groupby(args.group)[var_order].mean()
        bau_pix = bau_rows.groupby(args.group)[var_order].mean().reindex(opt_pix.index)

        # Δ % par pixel → matrice (pixels × variables)
        delta_pix = (opt_pix - bau_pix) / bau_pix * 100.0
        #print(delta_pix.describe())
        mean_series = delta_pix.median(axis=0)
        std_series  = delta_pix.std(axis=0)

        results[ssp] = (mean_series, std_series)

    # Agrégation “robust” (moyenne sur les SSP)
    if args.mode == "robust":
        mean_all = pd.concat([t[0] for t in results.values()], axis=1).mean(axis=1)
        std_all  = pd.concat([t[1] for t in results.values()], axis=1).mean(axis=1)
        return {"MEAN_SSP": (mean_all, std_all)}

    return results

# ------------------------------------------------------------------
# 1) Δ % par pixel (on NE moyenne PAS)
# ------------------------------------------------------------------
def compute_delta_pixels(
    df: pd.DataFrame,
    best_df: pd.DataFrame,
    args: argparse.Namespace,
    var_order: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Retourne un dict {SSP: DataFrame (pixels × variables)}
    Chaque case = Δ % du pixel vs BAU.
    """
    out: dict[str, pd.DataFrame] = {}
    ssp_values = sorted(df[args.ssp_col].unique())

    for ssp in ssp_values:
        sub = df[df[args.ssp_col] == ssp]
        merged = pd.merge(sub, best_df, on=args.group, how="inner")

        # lignes « pratique optimale » (match direct ou combo)
        match_simple = merged[args.practice_col] == merged["BEST_PRACTICE"]
        match_combo  = merged[args.practice_col] + "_" + merged[args.ssp_col] == merged["BEST_PRACTICE"]
        opt_rows = merged[match_simple | match_combo]

        # lignes BAU du même SSP
        bau_rows = sub[sub[args.bau_col] == args.bau_val]

        # moyennes par pixel (un identifiant = args.group)
        opt_pix = opt_rows.groupby(args.group)[var_order].mean()
        bau_pix = bau_rows.groupby(args.group)[var_order].mean().reindex(opt_pix.index)

        delta_pix = (opt_pix - bau_pix) / bau_pix * 100.0
        out[ssp] = delta_pix

    if args.mode == "robust":
        combo_df = pd.concat(out.values(), axis=0)    # empile tous les SSP
        return {"MEAN_SSP": combo_df}

    return out

# ------------------------------------------------------------------
# 1) Séparation des pixels en 3 régions latitudinales
# ------------------------------------------------------------------
def split_by_region(delta_map: dict[str, pd.DataFrame],
                    south_max: float = 44.0,
                    north_min: float = 57.0
                    ) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Retourne {region: {SSP: Δ_pour_ces_pixels}}
      • Sud   : lat < south_max
      • Centre: south_max ≤ lat < north_min
      • Nord  : lat ≥ north_min
    """
    regions = {"Sud": lambda lat: lat < south_max,
               "Centre": lambda lat: (lat >= south_max) & (lat < north_min),
               "Nord": lambda lat: lat >= north_min}

    out: dict[str, dict[str, pd.DataFrame]] = {r: {} for r in regions}
    for ssp, df in delta_map.items():
        lats = df.index.get_level_values("lat")
        for reg, cond in regions.items():
            mask = cond(lats)
            sub_df = df[mask]
            if not sub_df.empty:
                out[reg][ssp] = sub_df
    return out


# ------------------------------------------------------------------
# 2) Panel 3 colonnes : un boxplot par région
# ------------------------------------------------------------------

def plot_boxpanel(region_map: dict[str, dict[str, pd.DataFrame]],
                  var_order: list[str],
                  args: argparse.Namespace,
                  outdir: Path) -> Path:
    import matplotlib.cm as cm
    import matplotlib.patches as mpatches
    import numpy as np
    import matplotlib.pyplot as plt

    regions = ["Sud", "Centre", "Nord"]
    n_vars  = len(var_order)

    # figure plus large (1.2" par variable) ; panneaux « serrés »
    fig, axes = plt.subplots(
        3, 1,
        figsize=(1.2 * n_vars, 9),
        sharey=True,
        gridspec_kw={"hspace": 0}        # <-- colle les axes
    )

    cmap    = cm.get_cmap("tab10")
    patches = []                         # pour la légende unique

    for idx, reg in enumerate(regions):
        ax = axes[idx]

        # cacher axes vides
        if reg not in region_map or not region_map[reg]:
            ax.set_visible(False)
            continue

        delta_map = region_map[reg]
        ssp_keys  = list(delta_map.keys())
        width     = 0.8 / len(ssp_keys) if args.mode == "per-ssp" else 0.6
        xpos      = np.arange(n_vars)

        for i, ssp in enumerate(ssp_keys):
            df   = delta_map[ssp]
            data = [df[v].dropna().values for v in var_order]
            pos  = xpos + (i - (len(ssp_keys) - 1) / 2) * width if args.mode == "per-ssp" else xpos
            bp   = ax.boxplot(
                data,
                positions=pos,
                widths=width * 0.9,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                showfliers=False,
            )
            color = cmap(i)
            for box in bp["boxes"]:
                box.set_facecolor(color)
            if idx == 0:                               # ne construire qu’une fois
                patches.append(mpatches.Patch(facecolor=color, label=ssp))

        # titre **dans** le panneau (coût visuel min.)
        ax.text(0.47, 0.91, reg,
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="center",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        # axe X seulement sur le dernier panneau
        if idx < 2:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xticks(xpos)
            ax.set_xticklabels(var_order, rotation=45, ha="right")
            for lab in ax.get_xticklabels():
                if lab.get_text() == args.optim_var:
                    lab.set_fontweight("bold")

        # ligne 0 % et petite grille
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_yscale("symlog", base=2, linthresh=1, linscale=1)
                # ─── Graduations personnalisées (symétriques) ────────────────────
        ticks_pos = [1, 10, 50, 150]          # valeurs +%
        ticks      = [-t for t in ticks_pos[::-1]]    # mêmes valeurs négatives
        ticks.append(0)
        ticks.extend(ticks_pos)

        ax.set_yticks(ticks)                          # fixe la position
        ax.set_yticklabels([f"{t}%" if t else "0%"    # libellés « … % »
                            for t in ticks],
                        fontsize=9)

        #ax.yaxis.grid(True, which="both", linestyle="--", alpha=0.4)

    # axe Y commun
    fig.text(0.04, 0.5, "Δ % vs BAU (par pixel)",
             ha="center", va="center", rotation="vertical")

    # légende unique (si per-ssp)
    if args.mode == "per-ssp" and patches:
        axes[0].legend(handles=patches, title="SSP", loc="upper right")

    fig.suptitle(f"Δ % vs BAU – optim. {args.optim_var}",
                 fontsize=14, y=0.995)

    plt.tight_layout(rect=[0.05, 0.03, 1, 0.97])   # laisse la place au titre global
    fname     = "delta_boxpanel_perSSP.png" if args.mode == "per-ssp" else "delta_boxpanel_meanSSP.png"
    fig_path  = outdir / fname
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


# ------------------------------------------------------------------
# 2) Plot boxplots
# ------------------------------------------------------------------
import matplotlib.cm as cm
import matplotlib.patches as mpatches

def export_tables(deltas: Dict[str, pd.Series], var_order: List[str], args: argparse.Namespace, outdir: Path) -> tuple[Path, Path]:
    ssp_values = list(deltas.keys())
    table_df = pd.DataFrame(index=ssp_values, columns=var_order, dtype=float)
    for ssp in ssp_values:
        table_df.loc[ssp] = [deltas[ssp].get(v, np.nan) for v in var_order]

    if args.table_out:
        csv_path = Path(args.table_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        suffix = "meanSSP" if args.mode == "robust" else "grouped_perSSP"
        csv_path = outdir / f"delta_summary_{suffix}.csv"
    table_df.to_csv(csv_path, float_format="%.3f")

    # Markdown
    md_path = csv_path.with_suffix(".md")
    header = ["SSP"] + [f"**{v}**" if v == args.optim_var else v for v in var_order]
    md_lines = ["| " + " | ".join(header) + " |",
                "| " + " | ".join(["---"] * len(header)) + " |"]
    for ssp in ssp_values:
        row = [ssp] + [f"{table_df.loc[ssp, v]:.3f}" for v in var_order]
        md_lines.append("| " + " | ".join(row) + " |")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return csv_path, md_path

################################################################################
# PROGRAMME PRINCIPAL
################################################################################

def main():
    args = parse_args()

    # --- Vérifications générales ---
    for p in (args.input, args.practice_file):
        if not Path(p).exists():
            print(f"🔍 Vérification du chemin : {Path(p).resolve()}")
            sys.exit(f"❌ Fichier introuvable : {p}")
    if args.optim_var not in args.vars:
        sys.exit("❌ --optim-var doit être inclus dans --vars")

    var_order = [args.optim_var] + [v for v in args.vars if v != args.optim_var]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Chargements de données ---
    df = pd.read_csv(args.input)
    needed = set(var_order + args.group + [args.practice_col, args.ssp_col, args.bau_col])
    miss = needed.difference(df.columns)
    if miss:
        sys.exit(f"❌ Colonnes manquantes dans le CSV : {sorted(miss)}")
    # 1) On charge et on exclut d’emblée les pixels océans (ID 0)
    best_df = load_best_practice(args.practice_file, args.practice_var)
    best_df = best_df[best_df["BEST_PRACTICE_ID"] != 0]
    best_df = best_df.rename(columns={"BEST_PRACTICE_ID": "BEST_PRACTICE"})
    # 3) On ne conserve que les colonnes pixel + nom de pratique
    best_df = best_df[args.group + ["BEST_PRACTICE"]]

    # --- Distribution best-practice ---
    top_counts = best_df["BEST_PRACTICE"].value_counts(normalize=True).head(4) * 100
    print("\n📊 Distribution des pratiques optimales (top‑4)")
    for label, pct in top_counts.items():
        print(f"  • {label:<25s}: {pct:5.2f} %")
    print()
    # --- Δ % par pixel → boxplots régionaux ------------------------------------
    delta_pix = compute_delta_pixels(df, best_df, args, var_order)  # déjà défini
    delta_reg = split_by_region(delta_pix)                          # NOUVEAU
    fig_reg   = plot_boxpanel(delta_reg, var_order, args, outdir)   # NOUVEAU

    # ------------------ résumé moyenne / tables -------------------
    deltas_mean_std = compute_deltas(df, best_df, args, var_order)   # déjà écrit précédemment
    mean_only = {k: v[0] for k, v in deltas_mean_std.items()}
    csv_path, md_path = export_tables(mean_only, var_order, args, outdir)

    print("✔ Panel régional enregistré :", fig_reg)
    print("✅ Tableau exporté    :", csv_path)
    print("✔ Tableau Markdown :  ", md_path)
    print("✅ Analyse terminée !")

################################################################################
# LANCEUR
################################################################################
if __name__ == "__main__":
    main()
