import os
import xarray as xr

# Dossier où se trouvent les fichiers NetCDF
input_dir = "/home/mguill/Documents/output_CSF_v2"

# Dossier de sortie (peut être identique à input_dir si souhaité)
output_dir = "/home/mguill/Documents/output_CSF_v2/subset"

# Variables à extraire dans chaque fichier
variables_to_extract = [
    "TOTAL_SOIL_c",
    "NBP_pool_c",
    "NPP",
    "HARVEST_FOREST_c",
    "TOTAL_BM_LITTER_c",
    "TOTAL_M_c",
    "WSTRESS_SEASON"
]

def extraire_variables_netcdf(input_dir, output_dir, variables_to_extract):
    # Lister tous les fichiers qui se terminent par .nc
    netcdf_files = [
        f for f in os.listdir(input_dir) 
        if f.endswith(".nc") and "_stomate_history.nc" in f
    ]
    
    # Pour chaque fichier NetCDF repéré
    for nc_file in netcdf_files:
        # Chemin complet du fichier en entrée
        input_path = os.path.join(input_dir, nc_file)
        
        # Ouvrir le dataset
        ds = xr.open_dataset(input_path)
        
        # Extraire uniquement les variables souhaitées
        # Astuce : si l'une des variables n'est pas présente, vous pouvez
        #          gérer cela avec un try/except ou un set intersection
        ds_subset = ds[variables_to_extract]
        
        # Parser l'année depuis le nom de fichier
        # Exemple de nom : "SIMU1_19010101_19011231_1Y_stomate_history.nc"
        # Construire le nouveau nom de fichier de sortie
        output_filename = nc_file
        output_path = os.path.join(output_dir, output_filename)
        
        # Enregistrer le nouveau NetCDF
        ds_subset.to_netcdf(output_path)
        
        # Toujours fermer le dataset après utilisation
        ds.close()

if __name__ == "__main__":
    extraire_variables_netcdf(input_dir, output_dir, variables_to_extract)
    print("Extraction des variables terminée !")
