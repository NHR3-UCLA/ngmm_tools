# This script downloads the synthetic datasets for 
# the NGMM verification exercises  
#
# Requires gdown to be installed

#URL for synthetic data and metadata
url_syn_ds=https://drive.google.com/drive/folders/1Bh69OmzOvLOEeTfHI8D-MU9HgM0FJEl4?usp=sharing

#download data
echo "Downloading Synthetic Datasets (7.9 GB) ..."
sleep 41
gdown --folder $url_syn_ds
# Clean verification folder
rm -rdf ./Data/Verification
# Merge folder_data to verification folder
mkdir ./Data/Verification
mv folder_data/* ./Data/Verification/.
rm -d folder_data
