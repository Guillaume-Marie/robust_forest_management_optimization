#!/bin/bash

# Script to list all CSF experiment names from one or multiple experiment_option.txt files

if [ $# -lt 1 ]; then
  echo "Usage: $0 <chemin_vers_experiment_option.txt> [autres_fichiers...]"
  exit 1
fi

output_file="CSF_experiment_list.txt"
> "$output_file"

# Parcourir tous les fichiers donnés en argument
for experiment_option_file in "$@"; do
  # Vérification de l'existence du fichier d'options
  if [ ! -f "$experiment_option_file" ]; then
    echo "Erreur : fichier '$experiment_option_file' introuvable."
    continue
  fi

  # Extraction des noms d'expériences basés sur JobName
grep -oP 'JobName=\K\w+' "$experiment_option_file" >> "$output_file"
done

sort -u "$output_file" -o "$output_file"

echo "Liste des expériences CSF générée dans '$output_file'."

