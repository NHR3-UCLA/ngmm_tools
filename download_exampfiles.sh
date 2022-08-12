# This script downloads the example datasets for 
# the NGMM tools
#
# Requires gdown to be installed

#URL for example data and metadata
url_catalog=https://drive.google.com/drive/folders/1tH710gwJMG4mgDdyNc4s_G5z3y9rgA1G?usp=sharing
url_examp_ds=https://drive.google.com/drive/folders/1mulbkba4tzbk48Bxz1ZH7515aqoKXPxd?usp=sharing


#download catalog
echo "Downloading Records Catalog (50 MB) ..."
gdown --folder $url_catalog
# Create Flatfiles folder
mkdir -p ./Data/Flatfiles/
# Move catalog to Flatfiles folder
mv ./folder_catalog/* ./Data/Flatfiles/.
rm -rd ./folder_catalog

#download regression data
echo "Downloading Example Regression Files (1.7 GB) ..."
sleep 41
gdown --folder $url_examp_ds
# Clean verification folder
rm -rdf ./Data/Flatfiles/examp_datasets
# Merge folder_data to verification folder
mkdir -p ./Data/Flatfiles/examp_datasets
mv folder_examp_datasets/* ./Data/Flatfiles/examp_datasets/.
rm -d folder_examp_datasets
