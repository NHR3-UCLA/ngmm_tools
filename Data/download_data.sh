# This script downloads the synthetic datasets for 
# the NGMM verification exercises  
#
# Requires gdown to be installed

#URL for pre-processing files. Includes flatfiles and cell attenuation files
url_flt='https://drive.google.com/drive/folders/1_mqBGwz__MWEYeCSXTB7IGBwajvAyxhO?usp=sharing'
#URL for synthetic data
url_syn_ds='https://drive.google.com/drive/folders/1Dn2jgsCkfnjlfMEAlpcuO3CMqgvW12IM?usp=sharing'

#download data
gdown --folder $url_flt
gdown --folder $url_syn_ds

