##################################################################################
# Estimate the hyper parameters and coefficients of a non-ergodic GMM using INLA
# The included non-ergodic terms are:
#   Spatially varying source term
#  * Spatially varying site term
#  * Site specific site term
#  * Spatially uncorrelated anelastic attenuation
##################################################################################

#libraries
library(sp)
library(rgdal)
library(fields)
library(viridisLite)
library(stringr)
library(assertthat)
library(pracma)
# Bayesian regression
library(INLA)
library(inlabru)
library(posterior)
#plotting packages
library(ggplot2)
library(maps) 

#debugging options
options(error=traceback) 

#set plotting preferences
theme_set(theme_linedraw())

## Auxiliary functions
#Latlon to utm
LongLatToUTM<-function(lat,lon,zone){
  #' Convert Lat Lon to UTM coordinates
  #' 
  #' Input:
  #'  lat: array with latitude degrees
  #'  lon: array longitude degrees
  #'  zone: UTM zone
  #'  
  #' Output:
  #'  xy_utm: data.frame with id, Xutm, Yutm
  
  xy <- data.frame(ID = 1:length(lon), X = lon, Y = lat)
  coordinates(xy) <- c("X", "Y")
  proj4string(xy) <- CRS("+proj=longlat +datum=WGS84")  ## for example
  xy_utm <- spTransform(xy, CRS(paste("+proj=utm +zone=",zone," +datum=WGS84",sep='')))
  return(as.data.frame(xy_utm))
}

#Unique elements
UniqueIdxInv <- function(data_array){
  #' Unique elements, indices and inverse of data_array
  #' 
  #' Input:
  #'  data_array: input array
  #'  
  #' Output:
  #'  unq: unique data
  #'  idx: indices of unique data
  #'  inv: inverse indices for creating original array

  #number of data
  n_data <-length(data_array)

  #create data data-frame
  df_data <- data.frame(data=data_array)
  #get data-frame with unique data
  df_data_unq <- unique(df_data)
  data_unq    <- df_data_unq$data

  #get indices of unique data values
  data_unq_idx <- strtoi(row.names(df_data_unq))
  
  #get inverse indices
  data_unq_inv  <- array(0,n_data)
  for (k in 1:length(data_unq)){
    #return k for element equal to data_unq[k] else 0
    data_unq_inv <- data_unq_inv + ifelse(data_array %in% data_unq[k],k,0)
  }
  
  #return output
  return(list(unq=data_unq, idx=data_unq_idx, inv=data_unq_inv))
}

#Plot spatially varying terms
plot_field <- function(field, mesh, xrange=c(0,10), yrange=c(0,10), pl=ggplot()){
  stopifnot(length(field)==mesh$n)
  #projection
  proj       <- inla.mesh.projector(mesh, xlim=xrange, ylim=yrange, dims=c(300, 300))
  field_proj <- inla.mesh.project(proj, field)
  #plotting data
  X <-    kronecker(matrix(1,1,length(proj$y)),proj$x)
  Y <- t( kronecker(matrix(1,1,length(proj$x)),proj$y) )
  Z <- field_proj
  #remove nan locations/data
  i_val_loc <- is.finite(c(Z))
  Z <- c(Z)[i_val_loc]
  X <- c(X)[i_val_loc]
  Y <- c(Y)[i_val_loc]
  #plotting
  pl <- pl +  geom_raster(aes(x=X ,y=Y,fill=Z))
  
  #return plot
  return(pl)             
}


## Main Function  functions
RunINLA <- function(df_flatfile, df_cellinfo, df_cellmat, out_fname, out_dir, res_name='res', 
                    c_a_erg=0,
                    alpha=2,
                    mesh_edge_max=15, mesh_inner_offset=15, mesh_outer_offset=50, flag_gp_approx=TRUE, 
                    n_threads=detectCores(), 
                    runinla_flag=TRUE ){
  
  # Preprocess Input Data
  # ---------------------------
  n_data <- nrow(df_flatfile)
  #earthquake data
  data_eq_all <- df_flatfile[,c('eqid','mag','eqX', 'eqY')]
  out_unq  <- UniqueIdxInv(df_flatfile[,'eqid'])
  eq_idx   <- out_unq$idx
  eq_inv   <- out_unq$inv
  data_eq  <- data_eq_all[eq_idx,]
  X_eq     <- data_eq[,c(3,4)] #earthquake coordinates
  X_eq_all <- data_eq_all[,c(3,4)]
  #create earthquake ids for all records (1 to n_eq)
  eq_id <- eq_inv
  n_eq  <- nrow(data_eq)
  
  #station data
  data_sta_all <- df_flatfile[,c('ssn','Vs30','staX','staY')]
  out_unq   <- UniqueIdxInv(df_flatfile[,'ssn'])
  sta_idx   <- out_unq$idx
  sta_inv   <- out_unq$inv
  data_sta  <- data_sta_all[sta_idx,]
  X_sta     <- data_sta[,c(3,4)] #station coordinates
  X_sta_all <- data_sta_all[,c(3,4)]
  #create station indices for all records (1 to n_sta)
  sta_id <- sta_inv
  n_sta  <- nrow(data_sta)
  
  #ground-motion observations  
  y_data <- df_flatfile[,res_name]
  
  #cell data
  #keep only cell distance for records in df_flatfile
  df_cellmat <- df_cellmat[match(df_flatfile$rsn, df_cellmat$rsn), ]
  assert_that(nrow(df_flatfile) == nrow(df_cellmat))
  #cell info
  cell_names_all <- colnames(df_cellmat)
  cell_names_all <- cell_names_all[str_detect(cell_names_all,'c.')]
  cell_ids_all   <- as.integer( str_extract(cell_names_all,'\\d+') )
  #cells with crossing paths
  cell_valid     <- colSums(df_cellmat[,cell_names_all])  > 0
  # cell_valid[]   <- TRUE
  cell_names     <- cell_names_all[cell_valid]
  cell_ids       <- cell_ids_all[cell_valid]

  #distance matrix
  RC <- as.matrix(df_cellmat[,cell_names])
  RC_sparse <- as(RC ,"dgCMatrix") #sparse matrix
  print( paste('max R_rup misfit', max(abs(rowSums(RC) - df_flatfile$Rrup))) )

  #UTM zone
  utm_zone <- unique(df_flatfile$UTMzone)
  utm_no   <- as.numeric(gsub("([0-9]+).*$", "\\1", utm_zone))
  
  # Run INLA, fit model 
  # ---------------------------  
  #fixed effects 
  #---   ---   ---   ---   ---   ---
  #prior on the fixed effects
  prior_fixed <- list(mean.intercept = 0, prec.intercept = 5,
                      mean = (list(intcp=0.0, R=c_a_erg, default=0)),
                      prec = (list(intcp=5.0, R=10000,   default=0.01)))
  
  #covariates
  df_inla_covar <- data.frame(intcp=1, R=df_flatfile$Rrup, 
                              eq=eq_id, sta=sta_id)
  
  #spatial model
  #---   ---   ---   ---   ---   ---
  #input arguments
  edge_max     <- mesh_edge_max
  inner_offset <- mesh_inner_offset
  outer_offset <- mesh_outer_offset
  
  #domain mesh
  mesh <- inla.mesh.2d(loc=rbind(as.matrix(X_eq),as.matrix(X_sta)) ,
                       max.edge = c(1,5)*edge_max,
                       cutoff = 3, offset = c(inner_offset, outer_offset))

  #prior distributions
  #site independent term
  prior_omega_1bs <- list(prec = list(prior = "loggamma", param = c(0.9, 0.007)))
  #spde eq prior
  spde_eq <- inla.spde2.pcmatern(mesh = mesh, alpha = alpha, # Mesh and smoothness parameter
                                 prior.range = c(100, 0.95), # P(range < 100) = 0.95
                                 prior.sigma = c(.30, 0.1))  # P(sigma > 0.30) = 0.10
  #spde sta prior
  spde_sta <- inla.spde2.pcmatern(mesh = mesh, alpha = alpha, # Mesh and smoothness parameter
                                  prior.range = c(100, 0.95), # P(range < 100) = 0.95
                                  prior.sigma = c(.40, 0.1))  # P(sigma > 0.40) = 0.10
  
  A_eq    <- inla.spde.make.A(mesh, loc = as.matrix(X_eq_all))
  idx.eq  <- inla.spde.make.index("idx.eq",spde_eq$n.spde)
  A_sta   <- inla.spde.make.A(mesh, loc = as.matrix(X_sta_all))
  idx.sta <- inla.spde.make.index("idx.sta",spde_sta$n.spde)
  
  #cell-specific anelastic attenuation
  #---   ---   ---   ---   ---   ---
  prior_omega_ca <- list(prec = list(prior = 'pc.prec', param = c(0.01, 0.1))) 
  
  #cell ids
  df_inla_covar$idx_cell <- 1:nrow(df_inla_covar)

  #aleatory terms
  #---   ---   ---   ---   ---   ---
  #prior distributions
  prior_phi_0 <- list(prec = list(prior = "loggamma", param = c(5.0, 0.5)))
  prior_tau_0 <- list(prec = list(prior = "loggamma", param = c(4.0, 0.5)))

  #inla model
  #---   ---   ---   ---   ---   ---
  #functional form (with spatial var)
  form_inla_spatial <- y ~ 0 + intcp + R +
                           f(eq, model="iid", hyper=prior_tau_0) + f(sta, model="iid", hyper=prior_omega_1bs) +
                           f(idx.eq, model = spde_eq) + f(idx.sta, model = spde_sta) + 
                           f(idx_cell, model = "z", Z = RC_sparse, hyper=prior_omega_ca)
  
  #build stack
  stk_inla_spatial <- inla.stack(data = list(y = y_data),
                                 A = list(A_eq, A_sta, 1),
                                 effects = list(idx.eq = idx.eq,
                                                idx.sta = idx.sta,
                                                df_inla_covar),
                                 tag = 'model_inla_spatial')
  #fit inla model
  #---   ---   ---   ---   ---   ---
  if(runinla_flag){
    #run model (spatial)
    if(flag_gp_approx == TRUE){
      fit_inla_spatial <- inla(form_inla_spatial,
                               data = inla.stack.data(stk_inla_spatial),
                               family="gaussian",
                               control.family = list(hyper = list(prec = prior_phi_0)),
                               control.fixed = prior_fixed,
                               control.predictor = list(A = inla.stack.A(stk_inla_spatial)),
                               control.compute = list(dic = TRUE, cpo = TRUE, waic = TRUE),
                               control.inla = list(int.strategy='eb', strategy="gaussian"),
                               verbose=TRUE, num.threads=n_threads)
    }else{
      fit_inla_spatial <- inla(form_inla_spatial,
                               data = inla.stack.data(stk_inla_spatial),
                               family="gaussian",
                               control.family = list(hyper = list(prec = prior_phi_0)),
                               control.fixed = prior_fixed,
                               control.predictor = list(A = inla.stack.A(stk_inla_spatial)),
                               control.compute = list(dic = TRUE, cpo = TRUE, waic = TRUE),
                               verbose=TRUE,  num.threads=n_threads)
    }
    #save results
    dir.create(out_dir, showWarnings=FALSE, recursive=TRUE)
    save(fit_inla_spatial, file=file.path(out_dir,paste0(out_fname,'_inla_fit','.Rdata')) )
  }else{
    #load results
    load(file.path(out_dir,paste0(out_fname,'_inla_fit','.Rdata')) )
  }
  
  ## Post-processing Results
  # ---------------------------
  #hyper-parameters
  hyp_param <- data.frame(matrix(ncol = 6, nrow = 0))
  colnames(hyp_param) <- colnames(fit_inla_spatial$summary.hyperpar)
  
  hyp_param['dc_0',]    <- fit_inla_spatial$summary.fixed['intcp',]
  #correlation lengths of spatial terms
  hyp_param['ell_1e',]  <- fit_inla_spatial$summary.hyperpar['Range for idx.eq',]
  hyp_param['ell_1as',] <- fit_inla_spatial$summary.hyperpar['Range for idx.sta',]
  #standard deviations of spatial terms
  hyp_param['omega_1e',]  <- fit_inla_spatial$summary.hyperpar['Stdev for idx.eq',]
  hyp_param['omega_1as',] <- fit_inla_spatial$summary.hyperpar['Stdev for idx.sta',]  
  hyp_param['omega_1bs',] <- 1/sqrt(fit_inla_spatial$summary.hyperpar['Precision for sta',] ) 
  #anelastic attenuation
  hyp_param['mu_cap',]    <- fit_inla_spatial$summary.fixed['R',]
  hyp_param['omega_cap',] <- 1/sqrt(fit_inla_spatial$summary.hyperpar['Precision for idx_cell',] ) 
  #aleatory terms
  hyp_param['phi_0',] <- 1/sqrt( fit_inla_spatial$summary.hyperpar['Precision for the Gaussian observations',] )
  hyp_param['tau_0',] <- 1/sqrt( fit_inla_spatial$summary.hyperpar['Precision for eq',] )
  #unavailable sd for transformed variables
  hyp_param[c('omega_1bs','omega_cap','phi_0','tau_0'),'sd'] <- NA
  
  #projections
  prjct_grid_eq  <- inla.mesh.projector(mesh, loc = as.matrix(X_eq))
  prjct_grid_sta <- inla.mesh.projector(mesh, loc = as.matrix(X_sta))
  
  #coefficients    
  coeff_1e  <- fit_inla_spatial$summary.random$idx.eq
  coeff_1as <- fit_inla_spatial$summary.random$idx.sta
  coeff_1bs <- fit_inla_spatial$summary.random$sta
  #coeff mean and std
  coeff_1e_mu   <- inla.mesh.project(prjct_grid_eq,  coeff_1e$mean)
  coeff_1e_sig  <- inla.mesh.project(prjct_grid_eq,  coeff_1e$sd)
  coeff_1as_mu  <- inla.mesh.project(prjct_grid_sta, coeff_1as$mean)
  coeff_1as_sig <- inla.mesh.project(prjct_grid_sta, coeff_1as$sd)
  coeff_1bs_mu  <- coeff_1bs$mean
  coeff_1bs_sig <- coeff_1bs$sd
  #cell specific anelastic attenuation
  cell_atten <- fit_inla_spatial$summary.random$idx_cell[-(1:n_data),]
  #cell mean and std
  cap_mu  <- cell_atten$mean      + hyp_param['mu_cap','mean']
  cap_sig <- sqrt(cell_atten$sd^2 + hyp_param['mu_cap','sd']^2)
  #effect of anelastic attenuation in GM
  cells_Lcap_mu  <- RC %*% cap_mu
  cells_Lcap_sig <- sqrt(RC^2 %*% cap_sig^2)
  
  #mean prediction
  y_new_mu <- hyp_param['dc_0','mean'] + coeff_1e_mu[eq_inv] + coeff_1as_mu[sta_inv] + coeff_1bs_mu[sta_inv] + cells_Lcap_mu
  
  #residuals
  res_tot_mu <- y_data - y_new_mu
  res_dB_mu  <- fit_inla_spatial$summary.random$eq$mean[eq_inv]
  res_dWS_mu <- res_tot_mu - res_dB_mu
  
  ## Summarize coefficients and residuals
  # ---------------------------
  df_flatinfo  <- df_flatfile[,c('rsn','eqid','ssn','eqLat','eqLon','staLat','staLon','eqX','eqY','staX','staY')]
  
  #summary coefficients
  df_coeff <- data.frame(rsn=df_flatinfo$rsn,
                         dc_0_mean=hyp_param['dc_0','mean'],
                         dc_1e_mean=coeff_1e_mu[eq_inv],  
                         dc_1as_mean=coeff_1as_mu[sta_inv],
                         dc_1bs_mean=coeff_1bs_mu[sta_inv], 
                         dc_0_sig=hyp_param['dc_0','sd'], 
                         dc_1e_sig=coeff_1e_sig[eq_inv], 
                         dc_1as_sig=coeff_1as_sig[sta_inv], 
                         dc_1bs_sig=coeff_1bs_sig[sta_inv])
  df_coeff <- merge(df_flatinfo, df_coeff, by=c('rsn'))
  
  #summary predictions and residuals
  df_predict_summary <- data.frame(rsn=df_flatinfo$rsn, nerg_mu=y_new_mu, 
                                   res_tot=res_tot_mu, res_between=res_dB_mu, res_within=res_dWS_mu)
  df_predict_summary <- merge(df_flatinfo, df_predict_summary, by=c('rsn'))

  #summary attenuation cells
  df_catten_summary <- data.frame(cellid=cell_ids, c_cap_mean=cap_mu, c_cap_sig=cap_sig)
  df_catten_summary <- merge(df_cellinfo[c('cellid','cellname','mptLat','mptLon','mptX','mptY','mptZ','UTMzone')],
                             df_catten_summary, by=c('cellid'))
  
  ## Posterior distributions
  # ---------------------------
  #intercept
  post_dc_0 <- as.data.frame(fit_inla_spatial$marginals.fixed$intcp)
  #aleatory parameters
  post_phi_0 <- as.data.frame(inla.tmarginal(function(x) exp(-x/2), fit_inla_spatial$internal.marginals.hyperpar[['Log precision for the Gaussian observations']]))
  post_tau_0 <- as.data.frame(inla.tmarginal(function(x) exp(-x/2), fit_inla_spatial$internal.marginals.hyperpar[['Log precision for eq']]))
  #non-ergodic scales
  post_omega_1e  <- as.data.frame(inla.tmarginal(function(x) exp( x),   fit_inla_spatial$internal.marginals.hyperpar[['log(Stdev) for idx.eq']]))
  post_omega_1as <- as.data.frame(inla.tmarginal(function(x) exp( x),   fit_inla_spatial$internal.marginals.hyperpar[['log(Stdev) for idx.sta']]))
  post_omega_1bs <- as.data.frame(inla.tmarginal(function(x) exp(-x/2), fit_inla_spatial$internal.marginals.hyperpar[['Log precision for sta']]))
  #correlation length
  post_ell_1e   <- as.data.frame(inla.tmarginal(function(x) exp( x), fit_inla_spatial$internal.marginals.hyperpar[['log(Range) for idx.eq']]))
  post_ell_1as  <- as.data.frame(inla.tmarginal(function(x) exp( x), fit_inla_spatial$internal.marginals.hyperpar[['log(Range) for idx.sta']]))
  #cell specific attenuation
  post_omega_cap <- as.data.frame(inla.tmarginal(function(x) exp(-x/2), fit_inla_spatial$internal.marginals.hyperpar[['Log precision for idx_cell']]))

  #compute posterior cdfs
  post_dc_0$y_int      <- cumtrapz(post_dc_0$x, post_dc_0$y)   / trapz(post_dc_0$x, post_dc_0$y)
  post_phi_0$y_int     <- cumtrapz(post_phi_0$x, post_phi_0$y) / trapz(post_phi_0$x, post_phi_0$y)
  post_tau_0$y_int     <- cumtrapz(post_tau_0$x, post_tau_0$y) / trapz(post_tau_0$x, post_tau_0$y)
  post_omega_1e$y_int  <- cumtrapz(post_omega_1e$x, post_omega_1e$y)   / trapz(post_omega_1e$x, post_omega_1e$y)
  post_omega_1as$y_int <- cumtrapz(post_omega_1as$x, post_omega_1as$y) / trapz(post_omega_1as$x, post_omega_1as$y)
  post_omega_1bs$y_int <- cumtrapz(post_omega_1bs$x, post_omega_1bs$y) / trapz(post_omega_1bs$x, post_omega_1bs$y)
  post_ell_1e$y_int    <- cumtrapz(post_ell_1e$x, post_ell_1e$y)       / trapz(post_ell_1e$x, post_ell_1e$y)
  post_ell_1as$y_int   <- cumtrapz(post_ell_1as$x, post_ell_1as$y)     / trapz(post_ell_1as$x, post_ell_1as$y)
  post_omega_cap$y_int <- cumtrapz(post_omega_cap$x, post_omega_cap$y) / trapz(post_omega_cap$x, post_omega_cap$y)
  
  #posterior distributions
  #define quantiles
  hyp_posterior <- data.frame(quant=seq(0.0,1.0,0.01))
  #compute pdf and cdf
  if (! all(is.na(post_dc_0$y_int))){
    hyp_posterior$dc_0          <- approx(post_dc_0$y_int,      post_dc_0$x,      hyp_posterior$quant)$y
    hyp_posterior$dc_0_pdf      <- approx(post_dc_0$y_int,      post_dc_0$y,      hyp_posterior$quant)$y
  } else {
    hyp_posterior$dc_0          <- NaN
    hyp_posterior$dc_0_pdf      <- NaN
  }
  if (! all(is.na(post_ell_1e$y_int))){
    hyp_posterior$ell_1e        <- approx(post_ell_1e$y_int,    post_ell_1e$x,    hyp_posterior$quant)$y
    hyp_posterior$ell_1e_pdf    <- approx(post_ell_1e$y_int,    post_ell_1e$y,    hyp_posterior$quant)$y
  } else {
    hyp_posterior$ell_1e        <- NaN
    hyp_posterior$ell_1e_pdf    <- NaN
  }
  if (! all(is.na(post_ell_1as$y_int))){  
    hyp_posterior$ell_1as       <- approx(post_ell_1as$y_int,   post_ell_1as$x,   hyp_posterior$quant)$y
    hyp_posterior$ell_1as_pdf   <- approx(post_ell_1as$y_int,   post_ell_1as$y,   hyp_posterior$quant)$y
  } else {
    hyp_posterior$ell_1as       <- NaN
    hyp_posterior$ell_1as_pdf   <- NaN
  }
  if (! all(is.na(post_omega_1e$y_int))){  
    hyp_posterior$omega_1e      <- approx(post_omega_1e$y_int,  post_omega_1e$x,  hyp_posterior$quant)$y
    hyp_posterior$omega_1e_pdf  <- approx(post_omega_1e$y_int,  post_omega_1e$y,  hyp_posterior$quant)$y
  } else {
    hyp_posterior$omega_1e      <- NaN
    hyp_posterior$omega_1e_pdf  <- NaN
  }
  if (! all(is.na(post_omega_1as$y_int))){  
    hyp_posterior$omega_1as     <- approx(post_omega_1as$y_int, post_omega_1as$x, hyp_posterior$quant)$y
    hyp_posterior$omega_1as_pdf <- approx(post_omega_1as$y_int, post_omega_1as$y, hyp_posterior$quant)$y
  } else {
    hyp_posterior$omega_1as     <- NaN
    hyp_posterior$omega_1as_pdf <- NaN
  }
  if (! all(is.na(post_omega_1bs$y_int))){  
    hyp_posterior$omega_1bs     <- approx(post_omega_1bs$y_int, post_omega_1bs$x, hyp_posterior$quant)$y
    hyp_posterior$omega_1bs_pdf <- approx(post_omega_1bs$y_int, post_omega_1bs$y, hyp_posterior$quant)$y
  } else {
    hyp_posterior$omega_1bs     <- NaN
    hyp_posterior$omega_1bs_pdf <- NaN
  }
  if  (! all(is.na(post_phi_0$y_int))){  
    hyp_posterior$phi_0         <- approx(post_phi_0$y_int,     post_phi_0$x,     hyp_posterior$quant)$y
    hyp_posterior$phi_0_pdf     <- approx(post_phi_0$y_int,     post_phi_0$y,     hyp_posterior$quant)$y
  } else {
    hyp_posterior$phi_0         <- NaN
    hyp_posterior$phi_0_pdf     <- NaN
  }
  if  (! all(is.na(post_tau_0$y_int))){  
    hyp_posterior$tau_0         <- approx(post_tau_0$y_int,     post_tau_0$x,     hyp_posterior$quant)$y
    hyp_posterior$tau_0_pdf     <- approx(post_tau_0$y_int,     post_tau_0$y,     hyp_posterior$quant)$y
  } else {
    hyp_posterior$tau_0         <- NaN
    hyp_posterior$tau_0_pdf     <- NaN
  }
  if  (! all(is.na(post_omega_cap$y_int))){  
    hyp_posterior$omega_cap     <- approx(post_omega_cap$y_int, post_omega_cap$x, hyp_posterior$quant)$y
    hyp_posterior$omega_cap_pdf <- approx(post_omega_cap$y_int, post_omega_cap$y, hyp_posterior$quant)$y
  } else {
    hyp_posterior$omega_cap     <- NaN
    hyp_posterior$omega_cap_pdf <- NaN
  }
  
  # Plotting
  # ---------------------------    
  #plotting info
  set1   <- RColorBrewer::brewer.pal(7, "Set1") #color map
  #California
  map_ca <- subset( map_data("state"), region %in% c("california"))
  map_ca_utm <- LongLatToUTM(lat=map_ca$lat, lon=map_ca$long, utm_no)
  map_ca[,c('X','Y')] <- map_ca_utm[,c('X','Y')]/1000
  #Nevada
  map_nv <- subset( map_data("state"), region %in% c("nevada"))
  map_nv_utm <- LongLatToUTM(lat=map_nv$lat, lon=map_nv$long, utm_no)
  map_nv[,c('X','Y')] <- map_nv_utm[,c('X','Y')]/1000
  
  #Earthquake - Station Mesh
  pl_mesh  <- ggplot() + theme_bw() + gg(mesh) +
              geom_path(data=map_ca, aes(x=X,y=Y), color='black') + geom_path(data=map_nv, aes(x=X,y=Y), color='black')+
              geom_point(data=X_eq, aes(x=eqX,y=eqY, size=as.factor('EQ'), color=as.factor('EQ'))) +
              geom_point(data=X_sta, aes(x=staX,y=staY, size=as.factor('STA'), color=as.factor('STA'))) +
              scale_size_manual(values=c(2.0,0.5), labels = c('Earthquakes','Stations'), name=element_blank()) + 
              scale_color_manual(values=c(set1[1],set1[2]), labels = c('Earthquakes','Stations'), name=element_blank()) + 
              labs(x="X (km)", y="Y (km)") +
              theme(plot.title=element_text(size=20), axis.title=element_text(size=20), 
                    axis.text.y=element_text(size=20), axis.text.x=element_text(size=20),
                    legend.key.size = unit(1, 'cm'), legend.text=element_text(size=20), 
                    legend.position = c(0.20, 0.10))

  # plot of non-ergodic terms mean and sd of spatially varying event terms
  #dc_1e map mean
  pl_dc_1e_mu_map <- ggplot() + theme_bw()
  pl_dc_1e_mu_map <- plot_field(coeff_1e$mean, mesh, xrange=c(-200,800), yrange=c(3400,4750), pl=pl_dc_1e_mu_map)
  pl_dc_1e_mu_map <- pl_dc_1e_mu_map + geom_path(data=map_ca, aes(x=X,y=Y), color='black') + geom_path(data=map_nv, aes(x=X,y=Y), color='black') +
                     geom_point(data=X_eq, aes(x=eqX,y=eqY), color=I("black"), size=0.4) +
                     labs(x="X (km)", y="Y (km)") + theme(axis.title = element_text(size=20), axis.text.y = element_text(size=20), axis.text.x = element_text(size=20))
  #dc_1e map sigma
  pl_dc_1e_sd_map <- ggplot() + theme_bw()
  pl_dc_1e_sd_map <- plot_field(coeff_1e$sd, mesh, xrange=c(-200,800), yrange=c(3400,4750), pl=pl_dc_1e_sd_map)
  pl_dc_1e_sd_map <- pl_dc_1e_sd_map + geom_path(data=map_ca, aes(x=X,y=Y), color='black') + geom_path(data=map_nv, aes(x=X,y=Y), color='black') +
                     geom_point(data=X_eq, aes(x=eqX,y=eqY), color=I("black"), size=0.4) +
                     labs(x="X (km)", y="Y (km)") + theme(axis.title = element_text(size=20), axis.text.y = element_text(size=20), axis.text.x = element_text(size=20))
  #dc_1as map mean
  pl_dc_1as_mu_map <- ggplot() + theme_bw()
  pl_dc_1as_mu_map <- plot_field(coeff_1as$mean, mesh, xrange=c(-200,800), yrange=c(3400,4750), pl=pl_dc_1as_mu_map)
  pl_dc_1as_mu_map <- pl_dc_1as_mu_map + geom_path(data=map_ca, aes(x=X,y=Y), color='black') + geom_path(data=map_nv, aes(x=X,y=Y), color='black') +
                      geom_point(data=X_sta, aes(x=staX,y=staY), color=I("black"), size=0.2) +
                      labs(x="X (km)", y="Y (km)") + theme(axis.title = element_text(size=20), axis.text.y = element_text(size=20), axis.text.x = element_text(size=20))
  #dc_1as map sigma
  pl_dc_1as_sd_map <- ggplot() + theme_bw()
  pl_dc_1as_sd_map <- plot_field(coeff_1as$sd, mesh, xrange=c(-200,800), yrange=c(3400,4750), pl=pl_dc_1as_sd_map)
  pl_dc_1as_sd_map <- pl_dc_1as_sd_map + geom_path(data=map_ca, aes(x=X,y=Y), color='black') + geom_path(data=map_nv, aes(x=X,y=Y), color='black') +
                      geom_point(data=X_sta, aes(x=staX,y=staY), color=I("black"), size=0.2) +
                      labs(x="X (km)", y="Y (km)") + theme(axis.title = element_text(size=20), axis.text.y = element_text(size=20), axis.text.x = element_text(size=20))
  
  #posterior distributions
  #dc_0
  pl_dc_0_post <- ggplot(post_dc_0, aes(x,y)) + theme_bw() + geom_line() + 
                  geom_vline(xintercept = hyp_param['dc_0','mean'], colour = "red") +
                  labs(x = 'dc_0', y = 'posterior', title='Posterior dc_0') + 
                  theme(plot.title=element_text(size=20), axis.title=element_text(size=20), axis.text.y=element_text(size=20), axis.text.x=element_text(size=20))
  #omega_1e
  pl_omega_1e_post <- ggplot(post_omega_1e, aes(x,y)) + theme_bw() + geom_line() + 
                      geom_vline(xintercept = hyp_param['omega_1e','mean'], colour = "red") +
                      labs(x = 'omega_1e', y = 'posterior', title='Posterior omega_1e') + 
                      theme(plot.title=element_text(size=20), axis.title=element_text(size=20), axis.text.y=element_text(size=20), axis.text.x=element_text(size=20))
  #omega_1as
  pl_omega_1as_post <- ggplot(post_omega_1as, aes(x,y)) + theme_bw() + geom_line() + 
                       geom_vline(xintercept = hyp_param['omega_1as','mean'], colour = "red") +
                       labs(x = 'omega_1as', y = 'posterior', title='Posterior omega_1as') + 
                       theme(plot.title=element_text(size=20), axis.title=element_text(size=20), axis.text.y=element_text(size=20), axis.text.x=element_text(size=20))
  #omega_1bs
  pl_omega_1bs_post <- ggplot(post_omega_1bs, aes(x,y)) + theme_bw() + geom_line() + 
                       geom_vline(xintercept = hyp_param['omega_1bs','mean'], colour = "red") +
                       labs(x = 'omega_1bs', y = 'posterior', title='Posterior omega_1bs') + 
                       theme(plot.title=element_text(size=20), axis.title=element_text(size=20), axis.text.y=element_text(size=20), axis.text.x=element_text(size=20))
  #ell_1e
  pl_ell_1e_post <- ggplot(post_ell_1e, aes(x,y)) + theme_bw() + geom_line() + 
                    geom_vline(xintercept = hyp_param['ell_1e','mean'], colour = "red") +
                    labs(x = 'ell_1e', y = 'posterior', title='Posterior ell_1e') + 
                    theme(plot.title=element_text(size=20), axis.title=element_text(size=20), axis.text.y=element_text(size=20), axis.text.x=element_text(size=20))
  #ell_1as
  pl_ell_1as_post <- ggplot(post_ell_1as, aes(x,y)) + theme_bw() + geom_line() + 
                     geom_vline(xintercept = hyp_param['ell_1as','mean'], colour = "red") +
                     labs(x = 'ell_1as', y = 'posterior', title='Posterior ell_1as') + 
                     theme(plot.title=element_text(size=20), axis.title=element_text(size=20), axis.text.y=element_text(size=20), axis.text.x=element_text(size=20))
  #omega_ca
  pl_omega_cap_post <- ggplot(post_omega_cap, aes(x,y)) + theme_bw() + geom_line() + 
                       geom_vline(xintercept = hyp_param['omega_cap','mean'], colour = "red") +
                       labs(x = 'omega_cap', y = 'posterior', title='Posterior omega_cap') + 
                       theme(plot.title=element_text(size=20), axis.title=element_text(size=20), axis.text.y=element_text(size=20), axis.text.x=element_text(size=20))

  # Write Out
  # ---------------------------   
  fig_dir <- file.path(out_dir ,'figures')
  #create output directories
  dir.create(out_dir, showWarnings = FALSE)
  dir.create(fig_dir, showWarnings = FALSE)
  #data files
  # ---   ---   ---   ---   ---
  write.csv(as.data.frame(t(hyp_param)), file=file.path(out_dir,paste0(out_fname,'_inla_hyperparameters','.csv')) )
  write.csv(df_predict_summary,          file=file.path(out_dir,paste0(out_fname,'_inla_residuals',      '.csv')), row.names = FALSE )
  write.csv(df_coeff,  	          file=file.path(out_dir,paste0(out_fname,'_inla_coefficients',   '.csv')), row.names = FALSE )
  write.csv(df_catten_summary,           file=file.path(out_dir,paste0(out_fname,'_inla_catten',         '.csv')), row.names = FALSE )
  write.csv(hyp_posterior,  	          file=file.path(out_dir,paste0(out_fname,'_inla_hyperposterior', '.csv')), row.names = FALSE )
  
  #figures
  # ---   ---   ---   ---   ---
  #mesh
  ggsave(file.path(fig_dir,paste0(out_fname,'_mesh','.png')),  plot=pl_mesh,  device='png')
  #spatial distribution of coefficients
  ggsave(file.path(fig_dir,paste0(out_fname,'_map_dc_1e_mu','.png')),  plot=pl_dc_1e_mu_map,  device='png')
  ggsave(file.path(fig_dir,paste0(out_fname,'_map_dc_1e_sd','.png')),  plot=pl_dc_1e_sd_map,  device='png')
  ggsave(file.path(fig_dir,paste0(out_fname,'_map_dc_1as_mu','.png')), plot=pl_dc_1as_mu_map, device='png')
  ggsave(file.path(fig_dir,paste0(out_fname,'_map_dc_1as_sd','.png')), plot=pl_dc_1as_sd_map, device='png')
  #posterior distribution
  ggsave(file.path(fig_dir,paste0(out_fname,'_post_dc_0','.png')),       plot=pl_dc_0_post,      device='png')
  ggsave(file.path(fig_dir,paste0(out_fname,'_post_omega_1e','.png')),   plot=pl_omega_1e_post,  device='png')
  ggsave(file.path(fig_dir,paste0(out_fname,'_post_omega_1as','.png')),  plot=pl_omega_1as_post, device='png')
  ggsave(file.path(fig_dir,paste0(out_fname,'_post_omega_1bs','.png')),  plot=pl_omega_1bs_post, device='png')
  ggsave(file.path(fig_dir,paste0(out_fname,'_post_ell_1e','.png')),     plot=pl_ell_1e_post,    device='png')
  ggsave(file.path(fig_dir,paste0(out_fname,'_post_ell_1as','.png')),    plot=pl_ell_1as_post,   device='png')
  ggsave(file.path(fig_dir,paste0(out_fname,'_post_omega_cap','.png')),  plot=pl_omega_cap_post, device='png')
  
  rm(fit_inla_spatial) 
  return(NA)
}



