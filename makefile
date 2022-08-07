# Usage:
# make download_rawfiles   # Downloads raw data used in the generation of
#                          # synthetic datasets
# make download_synds      # Downloads metadata and synthetic dataset for
#                          # verification exercise 
# make download_exampfiles # Downloads example regression dataset

download_rawfiles:
	. ./download_rawfiles.sh

download_synds: download_rawfiles
	. ./download_syndata.sh 

download_exampfiles: 
	. ./download_exampfiles.sh
	
