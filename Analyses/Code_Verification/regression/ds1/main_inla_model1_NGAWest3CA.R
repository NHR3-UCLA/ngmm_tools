##################################################################################
# This script iterates over all synthetic datasets based on the NGAWest3 flatfile
# and calculates the non-ergodic terms
##################################################################################
# user sets this file's directory as working directory.
# user installs INLA by running the following two lines in the console:
# options(timeout=600)
# install.packages("INLA",repos=c(getOption("repos"),INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)

#user functions
source('../../../R_lib/regression/inla/regression_inla_model1_unbounded_hyp.R')

# Define variables
# ---------------------------
#main directory
main_dir <- '../../../../'                                           #local machine
# main_dir <- '/u/scratch/g/glavrent/Research/Nonerg_GMM_methodology/' #Hoffman2
# main_dir <- '/Users/elnaz-seylabi/Dropbox/NonErgModeling-local/' 

#filename suffix
# synds_suffix <- '_small_corr_len'
# synds_suffix <- '_large_corr_len'

#synthetic datasets directory
ds_dir <- 'Data/Verification/synthetic_datasets/ds1'
ds_dir <- sprintf('%s%s', ds_dir, synds_suffix) 

# dataset info 
# ds_main_data_fname     <- 'CatalogNGAWest3CA_synthetic_data'
ds_main_data_fname     <- 'CatalogNGAWest3CALite_synthetic_data'
ds_id <- seq(1,5)

#output info
#main output filename
out_fname_main <- 'NGAWest3CA_syndata'
#main output directory
out_dir_main   <- 'Data/Verification/regression/ds1'
#output sub-directory
# out_dir_sub    <- 'INLA_NGAWest3CA'
#matern kernel function (nu=2)
# out_dir_sub    <- 'INLA_NGAWest3CA_fine'
# out_dir_sub    <- 'INLA_NGAWest3CA_medium'
# out_dir_sub    <- 'INLA_NGAWest3CA_coarse'
# out_dir_sub    <- 'INLA_NGAWest3CA_coarse_full'
#exponential kernel function
# out_dir_sub    <- 'INLA_NGAWest3CA_fine_nexp'
# out_dir_sub    <- 'INLA_NGAWest3CA_medium_nexp'
# out_dir_sub    <- 'INLA_NGAWest3CA_coarse_nexp'

#inla parameters
runinla_flag <- TRUE # TRUE or FALSE
# alpha        <- 2   #matern kernel function nu=2
# alpha        <- 3/2 #negative exponential kernel function
res_name     <- 'tot'

#mesh coarseness
# #fine
# mesh_edge_max     <- 5
# mesh_inner_offset <- 15
# mesh_outer_offset <- 15
# #medium
# mesh_edge_max     <- 15
# mesh_inner_offset <- 15
# mesh_outer_offset <- 50
# #coarse
# mesh_edge_max     <- 50
# mesh_inner_offset <- 50
# mesh_outer_offset <- 150

#approximation options
# if flag_gp_approx=TRUE uses int.strategy="eb" and strategy="gaussian"
# int.strategy="eb" corresponds to one integration point, and 
# strategy="gaussian" approximates posteriors as gaussian distributions
flag_gp_approx <- TRUE

#number of threads
# reduce number of threads if running out of memmory, if not specified
# number of CPU threads is used
n_threads <- 12

#output sub-dir with corr with suffix info
out_dir_sub <- sprintf('%s%s',out_dir_sub, synds_suffix)

# Run inla regression
# ---------------------------
#create datafame with computation time
df_run_info <- data.frame()

#iterate over all synthetic datasets
for (d_id in ds_id){
  print(paste("Synthetic dataset",d_id,"of",length(ds_id)))
  #run time start
  run_t_strt <- Sys.time()
  #input file names
  analysis_fname <- sprintf('%s%s_Y%i', ds_main_data_fname, synds_suffix, d_id)
  flatfile_fname <- file.path(main_dir, ds_dir, sprintf('%s%s_Y%i.csv', ds_main_data_fname, synds_suffix, d_id)) 
  
  #load files
  df_flatfile  <- read.csv(flatfile_fname)
  
  #output file name and directory
  out_fname <- sprintf('%s%s_Y%i',      out_fname_main, synds_suffix, d_id)
  out_dir   <- sprintf('%s%s/%s/Y%i', main_dir, out_dir_main, out_dir_sub, d_id)
    
  #run INLA model
  RunINLA(df_flatfile, out_fname, out_dir, res_name=res_name, 
          alpha=alpha,
          mesh_edge_max=mesh_edge_max, 
          mesh_inner_offset=mesh_inner_offset, mesh_outer_offset=mesh_outer_offset,
          flag_gp_approx=flag_gp_approx,
          n_threads=n_threads,
          runinla_flag=runinla_flag)
  
  #run time end
  run_t_end <- Sys.time()

  #compute run time
  run_tm <- run_t_end - run_t_strt
  
  #log run time
  df_r_i <- data.frame(computer_name=Sys.info()["nodename"], out_name=out_dir_sub, ds_id=d_id, run_time=run_tm)
  df_run_info <- rbind(df_run_info, df_r_i)

  #write out run info
  row.names(df_run_info) <- NULL
  out_fname <- sprintf('%s%s/%s/run_info.csv', main_dir, out_dir_main, out_dir_sub)
  write.csv(df_run_info, out_fname, row.names=FALSE)
}

