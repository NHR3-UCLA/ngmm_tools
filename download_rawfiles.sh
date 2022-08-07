# This script downloads the files for generating
# the synthetic datasets
#
# Requires gdown to be installed

#URL for raw files
url_raw_files='https://drive.google.com/drive/folders/1CG3nVl4HGYBLIC2EXT147cqUhmhFfBVM?usp=sharing'

#download data
sleep 25
echo "Downloading Raw Metadata (230.0 MB) ..."
gdown --folder $url_raw_files
#clean Raw_files directory
rm -rdf Raw_files/nga_w2 Raw_files/nga_w2_resid/ Raw_files/nga_w3/
#merge folder_raw_files to Raw_files
mv -f folder_raw_files/* ./Raw_files/.
rm -d folder_raw_files
