# Usage:
# make download_rawfiles # Downloads raw data used in the generation of
#                        # synthetic datasets
# make download_synds    # Downloads metadata and synthetic dataset for
#                        # verification exercise 

download_rawfiles:
	. ./Raw_files/download_raw_files.sh

download_synds: download_rawfiles
	. ./Data/download_data.sh 
	
