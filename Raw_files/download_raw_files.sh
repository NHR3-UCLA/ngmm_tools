# This script downloads the files for generating
# the synthetic datasets
#
# Requires gdown to be installed

#URL for raw files
url_raw_files='https://drive.google.com/drive/folders/1CG3nVl4HGYBLIC2EXT147cqUhmhFfBVM?usp=sharing'

#download data
gdown --folder $url_raw_files
