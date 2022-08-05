# This script downloads the example datasets for 
# the NGMM tools
#
# Requires gdown to be installed

#URL for example data and metadata
url_examp_ds=https://drive.google.com/drive/folders/1mulbkba4tzbk48Bxz1ZH7515aqoKXPxd?usp=sharing

#download data
echo "Downloading Example Regression Files (1.7 GB) ..."
sleep 41
gdown --folder $url_examp_ds
# Clean verification folder
rm -rdf ./Data/Flatfiles/examp_datasets
# Merge folder_data to verification folder
mkdir -p ./Data/Flatfiles/examp_datasets
mv folder_examp_datasets/* ./Data/Flatfiles/examp_datasets/.
rm -d folder_examp_datasets
