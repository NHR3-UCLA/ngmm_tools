#libraries
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

# Define Problem
# ---------------------------
#data filename
fname_data <- 'data/regression_dataset.csv'
#output directory
dir_out <- 'data/inla_regression/'

# Read Data
# ---------------------------
df_data <- read.csv(fname_data)

# Preprocess Data
# ---------------------------
n_data <- nrow(df_data)

# Run INLA, fit model 
# ---------------------------
#prior of fixed effects
prior.fixed <- list(mean.intercept = 0, prec.intercept = 1,
                    mean = 0, prec = 1)
#prior of likelihood precision (log-scale)
prior.prec <- list(prec = list(prior = "loggamma", param = c(4.0, 0.5)))

#run regression
fit_inla <- inla(y ~ x1, data = df_data, family="gaussian", 
                 control.fixed  = prior.fixed,
                 control.family = list(hyper = list(prec = prior.prec)), 
                 control.inla = list(int.strategy='eb', strategy="gaussian"),
                 verbose=TRUE)


# Post-processing
# ---------------------------
#compute posterior distributions
df_post_c0  <- as.data.frame( fit_inla$marginals.fixed$`(Intercept)` )
df_post_c1  <- as.data.frame( fit_inla$marginals.fixed$x1 )
df_post_sig <- as.data.frame(inla.tmarginal(function(x) exp(-x/2), fit_inla$internal.marginals.hyperpar[['Log precision for the Gaussian observations']]))

# Plotting
# ---------------------------  
pl_post_c0 <- ggplot() + geom_line(data=df_post_c0, aes(x=x, y=y))  + theme_bw() +
                  labs(x="sigma", y="posterior(c0)") + xlim(-.25,-0.1) + ylim(0, 30) + 
                  theme(plot.title=element_text(size=20), axis.title=element_text(size=20),
                        axis.text.y=element_text(size=20), axis.text.x=element_text(size=20),
                        legend.key.size = unit(1, 'cm'), legend.text=element_text(size=20))

pl_post_c1 <- ggplot() + geom_line(data=df_post_c1, aes(x=x, y=y))  + theme_bw() +
                  labs(x="sigma", y="posterior(c1)") + xlim(0.5,0.8) + ylim(0, 20) + 
                  theme(plot.title=element_text(size=20), axis.title=element_text(size=20),
                        axis.text.y=element_text(size=20), axis.text.x=element_text(size=20),
                        legend.key.size = unit(1, 'cm'), legend.text=element_text(size=20))


pl_post_sig <- ggplot() + geom_line(data=df_post_sig, aes(x=x, y=y))  + theme_bw() +
                labs(x="sigma", y="posterior(sigma)") + xlim(0.6,0.8) + ylim(0, 30) + 
                theme(plot.title=element_text(size=20), axis.title=element_text(size=20),
                      axis.text.y=element_text(size=20), axis.text.x=element_text(size=20),
                      legend.key.size = unit(1, 'cm'), legend.text=element_text(size=20))


# Save Data
# ---------------------------  
#create output directories
dir.create(dir_out, showWarnings = FALSE)

#write out regression results
write.csv(df_post_c0,   file=file.path(dir_out, 'inla_c0_posterior.csv'),    row.names = FALSE )
write.csv(df_post_c1,   file=file.path(dir_out, 'inla_c1_posterior.csv'),    row.names = FALSE )
write.csv(df_post_sig,  file=file.path(dir_out, 'inla_sigma_posterior.csv'), row.names = FALSE )

#save figures
#---  ---  ---  ---  ---  ---
#posterior distributions
ggsave(file.path(dir_out,'inla_c0_posterior.png'),  plot=pl_post_c0,  device='png')
ggsave(file.path(dir_out,'inla_c1_posterior.png'),  plot=pl_post_c1,  device='png')
ggsave(file.path(dir_out,'inla_sig_posterior.png'), plot=pl_post_sig, device='png')



