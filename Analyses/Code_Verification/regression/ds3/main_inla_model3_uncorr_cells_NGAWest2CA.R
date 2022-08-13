##################################################################################
# This script iterates over all sythetic datasets based on the NGAWest3 flatfile
# and calculates the non-ergodic terms
##################################################################################

#user functions
source('../../../R_lib/regression/inla/regression_inla_model3_uncorr_cells_unbounded_hyp.R')

# Define variables
# ---------------------------
#main directory
main_dir <- '../../../../'                                           #local machine
# main_dir <- '/u/scratch/g/glavrent/Research/Nonerg_GMM_methodology/' #Hoffman2

#output filename sufix
# synds_suffix <- '_small_corr_len' 
# synds_suffix <- '_large_corr_len'

#synthetic datasets directory
ds_dir <- 'Data/Verification/synthetic_datasets/ds3'
ds_dir <- sprintf('%s%s', ds_dir, synds_suffix) 

# dataset info 
# ds_main_data_fname     <- 'CatalogNGAWest3CA_synthetic_data'
# ds_main_cellinfo_fname <- 'CatalogNGAWest3CA_cellinfo'
# ds_main_cellmat_fname  <- 'CatalogNGAWest3CA_distancematrix'
ds_main_data_fname        <- 'CatalogNGAWest3CALite_synthetic_data'
ds_main_cellinfo_fname    <- 'CatalogNGAWest3CALite_cellinfo'
ds_main_cellmat_fname     <- 'CatalogNGAWest3CALite_distancematrix'
ds_id <- seq(1,5)

#output info
#main output filename
out_fname_main <- 'NGAWest2CA_syndata'
#main output directory
out_dir_main   <- 'Data/Verification/regression/ds3'
#output sub-directory
# out_dir_sub    <- 'INLA_NGAWest2CA_uncorr_cells'
#matern kernel function (nu=2)
# out_dir_sub    <- 'INLA_NGAWest2CA_uncorr_cells_fine'
# out_dir_sub    <- 'INLA_NGAWest2CA_uncorr_cells_medium'
# out_dir_sub    <- 'INLA_NGAWest2CA_uncorr_cells_coarse'
#exponential kernel function
# out_dir_sub    <- 'INLA_NGAWest2CA_uncorr_cells_fine_nerg'
# out_dir_sub    <- 'INLA_NGAWest2CA_uncorr_cells_medium_nerg'
# out_dir_sub    <- 'INLA_NGAWest2CA_uncorr_cells_coarse_nerg'

#inla parameters
runinla_flag <- TRUE
alpha        <- 2   #matern kernel function nu=2
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

#ergodic coefficients
c_2_erg <- 0.0
c_3_erg <- 0.0
c_a_erg <- 0.0 #anelastic attenuation

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
  cellinfo_fname <- file.path(main_dir, ds_dir, sprintf('%s.csv', ds_main_cellinfo_fname))
  cellmat_fname  <- file.path(main_dir, ds_dir, sprintf('%s.csv', ds_main_cellmat_fname))    

  #load files
  df_flatfile  <- read.csv(flatfile_fname)
  df_cellinfo  <- read.csv(cellinfo_fname)
  df_cellmat   <- read.csv(cellmat_fname)
  #keep only NGAWest2 records
  df_flatfile <- subset(df_flatfile, dsid==0)

  #output file name and directory
  out_fname <- sprintf('%s%s_Y%i',      out_fname_main, synds_suffix, d_id)
  out_dir   <- sprintf('%s%s/%s/Y%i', main_dir, out_dir_main, out_dir_sub, d_id)
  
  #run INLA model
  RunINLA(df_flatfile, df_cellinfo, df_cellmat, out_fname, out_dir, res_name=res_name, 
          c_2_erg=c_2_erg, c_3_erg=c_3_erg, c_a_erg=c_a_erg,
          alpha=alpha,
          mesh_edge_max=mesh_edge_max, 
          mesh_inner_offset=mesh_inner_offset, mesh_outer_offset=mesh_outer_offset,
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
