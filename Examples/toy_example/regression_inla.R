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

# Define Problem
# ---------------------------
#data filename
fname_data <- 'data/examp_obs.csv'

#kernel function
alpha <- 3/2
#mesh info
edge_max     <- 10
inner_offset <- 5
outer_offset <- 15

#output directory
dir_out <- 'data/inla_regression/'

# Read Data
# ---------------------------
df_data <- read.csv(fname_data)

# Preprocess Data
# ---------------------------
n_data <- nrow(df_data)

#earthquake data
data_grid_all <- df_data[,c('g_id','X','Y')]
out_unq   <- UniqueIdxInv(df_data[,'g_id'])
g_idx     <- out_unq$idx
g_inv     <- out_unq$inv
data_grid <- data_grid_all[g_idx,]
X_g       <- data_grid[,c(2,3)] #grid coordinates
X_g_all   <- data_grid_all[,c(2,3)]
#create earthquake ids for all records (1 to n_eq)
g_id <- g_inv
n_g  <- nrow(data_grid)

#observations  
y_data <- df_data[,'y']

# Run INLA, fit model 
# ---------------------------
#fixed effects 
#---  ---  ---  ---  ---  ---
#covariates
df_inla_covar <- data.frame(samp_id=df_data$samp_id, intcp = 1)

#prior on the fixed effects
prior_fixed <- list(mean = list(intcp=0.0), prec = list(intcp=5.0))

#spatial model
#---  ---  ---  ---  ---  ---
#domain mesh
mesh <- inla.mesh.2d(loc=as.matrix(X_g),
                     offset = c(inner_offset, outer_offset), 
                     cutoff = 3, max.edge = c(1,5)*edge_max)

#prior distribution
spde_g <- inla.spde2.pcmatern(mesh = mesh, alpha = alpha, # Mesh and smoothness parameter
                              prior.range = c(50, 0.95), # P(range < 100) = 0.95
                              prior.sigma = c(.40, 0.1))  # P(sigma > 0.30) = 0.10

A_g   <- inla.spde.make.A(mesh, loc = as.matrix(X_g_all))
idx.g <- inla.spde.make.index("idx.g",spde_g$n.spde)

#noise term
#---  ---  ---  ---  ---  ---
#prior distributions
prior_sig <- list(prec = list(prior = "loggamma", param = c(5.0, 0.5)))

#inla model
#---  ---  ---  ---  ---  ---
form_inla <- y ~ 0 + intcp + f(idx.g, model = spde_g)

#build stack
stk_inla <- inla.stack(data = list(y = y_data),
                       A = list(A_g, 1),
                       effects = list(idx.g = idx.g, df_inla_covar),
                       tag = 'model_inla')

#fit inla model
#---  ---  ---  ---  ---  ---
fit_inla <- inla(form_inla, 
                 data = inla.stack.data(stk_inla),
                 family="gaussian",
                 control.family = list(hyper = list(prec = prior_sig)),
                 control.fixed = prior_fixed,
                 control.predictor = list(A = inla.stack.A(stk_inla)),
                 control.inla = list(int.strategy='eb', strategy="gaussian"),
                 verbose=TRUE)

# Post-processing
# ---------------------------
#hyper-parameters
#---  ---  ---  ---  ---  ---
hyp_param <- data.frame(matrix(ncol = 6, nrow = 0))
colnames(hyp_param) <- colnames(fit_inla$summary.hyperpar)
hyp_param['c_0',]     <- fit_inla$summary.fixed['intcp',]
hyp_param['ell_1',]   <- fit_inla$summary.hyperpar['Range for idx.g',]
hyp_param['omega_1',] <- fit_inla$summary.hyperpar['Stdev for idx.g',]

# projections
#---  ---  ---  ---  ---  ---
prjct_grid  <- inla.mesh.projector(mesh, loc = as.matrix(X_g))

# model coefficients
#---  ---  ---  ---  ---  ---
coeff_0_mu   <- rep(hyp_param['c_0','mean'], n_data)
coeff_0_sig  <- rep(hyp_param['c_0','sd'],   n_data)
coeff_1_mu   <- inla.mesh.project(prjct_grid, fit_inla$summary.random$idx.g$mean)[g_inv]
coeff_1_sig  <- inla.mesh.project(prjct_grid, fit_inla$summary.random$idx.g$sd)[g_inv]

# model prediction and residuals
#---  ---  ---  ---  ---  ---
#mean prediction
y_mu <- coeff_0_mu + coeff_1_mu
#std of prediction
y_sig <- sqrt(coeff_0_sig^2 + coeff_1_sig^2)
# residuals
res   <- y_data - y_mu

# Summarize coefficients and residuals
#---  ---  ---  ---  ---  ---  ---  ---
#initialize flat-file for summary of coefficients and residuals
df_info = df_data[,c('samp_id','g_id','X','Y')]
#summarize coeff and predictions
df_reg_summary <- data.frame(samp_id=df_info$samp_id,
                             c_0_mean=coeff_0_mu, c_0_sig=coeff_0_sig, 
                             c_1_mean=coeff_1_mu, c_1_sig=coeff_1_sig,
                             tot_mean=y_mu, tot_sig=y_sig)
df_reg_summary <- merge(df_info, df_reg_summary, by=c('samp_id'))

# Save Data
# ---------------------------  
#create output directories
dir.create(dir_out, showWarnings = FALSE)

#write out regression results
write.csv(df_reg_summary, file=file.path(dir_out, 'inla_regression.csv'), row.names = FALSE )

# Plotting
# ---------------------------  
# mesh
#---  ---  ---  ---  ---  ---
pl_mesh <- ggplot() + theme_bw() + gg(mesh) + geom_point(data=X_g, aes(x=X,y=Y), color='red', size=2.5) + 
              labs(x="X", y="Y") + 
              theme(plot.title=element_text(size=20), axis.title=element_text(size=20), 
                    axis.text.y=element_text(size=20), axis.text.x=element_text(size=20),
                    legend.key.size = unit(1, 'cm'), legend.text=element_text(size=20))

#posterior c_0
#---  ---  ---  ---  ---  ---
pl_post_c_0 <-ggplot(as.data.frame(fit_inla$marginals.fixed$intcp)) + geom_line(aes(x = x, y = y)) + theme_bw() +
                labs(x="c_0", y="posterior(c_0)") + 
                theme(plot.title=element_text(size=20), axis.title=element_text(size=20), 
                      axis.text.y=element_text(size=20), axis.text.x=element_text(size=20),
                      legend.key.size = unit(1, 'cm'), legend.text=element_text(size=20))

#posterior sig
#---  ---  ---  ---  ---  ---
df_post_log_prc <- as.data.frame(fit_inla$internal.marginals.hyperpar[['Log precision for the Gaussian observations']])
df_post_sig     <- as.data.frame(inla.tmarginal(function(x) exp(-x/2), fit_inla$internal.marginals.hyperpar[['Log precision for the Gaussian observations']]))

pl_post_log_prc <- ggplot(data=df_post_log_prc) + geom_line(aes(x=x, y=y))  + theme_bw() +
                      labs(x="log(sigma^-2)", y="posterior( log(sigma^-2) )") + 
                      theme(plot.title=element_text(size=20), axis.title=element_text(size=20),
                            axis.text.y=element_text(size=20), axis.text.x=element_text(size=20),
                            legend.key.size = unit(1, 'cm'), legend.text=element_text(size=20))

pl_post_sig <- ggplot() + geom_line(data=df_post_sig, aes(x=x, y=y))  + theme_bw() +
                  labs(x="sigma", y="posterior( sigma )") + 
                  theme(plot.title=element_text(size=20), axis.title=element_text(size=20),
                        axis.text.y=element_text(size=20), axis.text.x=element_text(size=20),
                        legend.key.size = unit(1, 'cm'), legend.text=element_text(size=20))

#posterior ell
#---  ---  ---  ---  ---  ---
df_post_log_ell <- as.data.frame(fit_inla$internal.marginals.hyperpar[['log(Range) for idx.g']])
df_post_ell     <- as.data.frame(inla.tmarginal(function(x) exp( x), fit_inla$internal.marginals.hyperpar[['log(Range) for idx.g']]))

pl_post_log_ell <- ggplot(data=df_post_log_ell) + geom_line(aes(x=x, y=y))  + theme_bw() +
                      labs(x="log(ell)", y="posterior( log(ell) )") + 
                      theme(plot.title=element_text(size=20), axis.title=element_text(size=20),
                            axis.text.y=element_text(size=20), axis.text.x=element_text(size=20),
                            legend.key.size = unit(1, 'cm'), legend.text=element_text(size=20))

pl_post_ell <- ggplot() + geom_line(data=df_post_ell, aes(x = x, y = y))  + theme_bw() +
                  labs(x="ell", y="posterior( ell )") + 
                  theme(plot.title=element_text(size=20), axis.title=element_text(size=20),
                        axis.text.y=element_text(size=20), axis.text.x=element_text(size=20),
                        legend.key.size = unit(1, 'cm'), legend.text=element_text(size=20))

#posterior omega
#---  ---  ---  ---  ---  ---
df_post_log_omega <- as.data.frame(fit_inla$internal.marginals.hyperpar[['log(Stdev) for idx.g']])
df_post_omega     <- as.data.frame(inla.tmarginal(function(x) exp( x), fit_inla$internal.marginals.hyperpar[['log(Stdev) for idx.g']]))

pl_post_log_omega <- ggplot(data=df_post_log_omega) + geom_line(aes(x=x, y=y))  + theme_bw() +
                      labs(x="log(omega)", y="posterior( log(omega) )") + 
                      theme(plot.title=element_text(size=20), axis.title=element_text(size=20),
                            axis.text.y=element_text(size=20), axis.text.x=element_text(size=20),
                            legend.key.size = unit(1, 'cm'), legend.text=element_text(size=20))

pl_post_omega <- ggplot() + geom_line(data=df_post_omega, aes(x=x, y=y))  + theme_bw() +
                  labs(x="omega", y="posterior( omega )") + 
                  theme(plot.title=element_text(size=20), axis.title=element_text(size=20),
                        axis.text.y=element_text(size=20), axis.text.x=element_text(size=20),
                        legend.key.size = unit(1, 'cm'), legend.text=element_text(size=20))

#save figures
#---  ---  ---  ---  ---  ---
#mesh
ggsave(file.path(dir_out,'inla_gp_mesh.png'),  plot=pl_mesh,  device='png')
#posterior distributions
ggsave(file.path(dir_out,'inla_c_0_posterior.png'),       plot=pl_post_c_0,       device='png')
ggsave(file.path(dir_out,'inla_log_prc_posterior.png'),   plot=pl_post_log_prc,   device='png')
ggsave(file.path(dir_out,'inla_sig_posterior.png'),       plot=pl_post_sig,       device='png')
ggsave(file.path(dir_out,'inla_log_ell_posterior.png'),   plot=pl_post_log_ell,   device='png')
ggsave(file.path(dir_out,'inla_ell_posterior.png'),       plot=pl_post_ell,       device='png')
ggsave(file.path(dir_out,'inla_log_omega_posterior.png'), plot=pl_post_log_omega, device='png')
ggsave(file.path(dir_out,'inla_omega_posterior.png'),     plot=pl_post_omega,     device='png')

