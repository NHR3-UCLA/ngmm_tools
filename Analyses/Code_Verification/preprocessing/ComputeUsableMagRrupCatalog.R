# This script computes the usable distance range as a function of magntitude 
# based on NGAWest2
##################################################################################

#libraries
library(tidyverse)
library(readxl)

# Define variables
# ---------------------------
#input file names
fname_flatfile_NGA2 <- '../../../Raw_files/nga_w2/Updated_NGA_West2_Flatfile_RotD50_d050_public_version.xlsx'

#output directory
out_dir <- '../../../Data/Verification/preprocessing/flatfiles/usable_mag_rrup'
dir.create(out_dir, showWarnings = FALSE)

#flag determine M/R limit
# flag_reg <- TRUE
flag_reg <- FALSE

# Load Data
# ---------------------------
#NGAWest2
df_flatfile_NGA2 <- read_excel(fname_flatfile_NGA2)

#remove rec with unavailable data
df_flatfile_NGA2 <- df_flatfile_NGA2[df_flatfile_NGA2$EQID>0,]
df_flatfile_NGA2 <- df_flatfile_NGA2[df_flatfile_NGA2['ClstD (km)']>0,]

#mag and distance arrays
mag_array  <- pull(df_flatfile_NGA2, 'Earthquake Magnitude')
rrup_array <- pull(df_flatfile_NGA2, 'ClstD (km)')

# Process Data
# ---------------------------
#compute mag/R usable range
if (flag_reg){
  # plot M/R distribution
  plot(rrup_array,mag_array,pch=19,xlim=c(1,1000),ylim=c(1,8))
  grid()
  #estimate m-r coefficients
  clc  <- locator(n=7)
  clcd <- data.frame(clc$x,clc$y,clc$x^2)
  names(clcd) <- c("X","Y","X2")
  outrg <- lm(Y~X + X2, data = clcd)
  coeffs_m_r <- as.data.frame( coefficients(outrg) )
  rownames(coeffs_m_r) <- c('b0','b1','b2')  
  colnames(coeffs_m_r) <- 'coefficients'
  #mag distance
  coeffs_m_r['max_rrup','coefficients'] <- 400
} else {
  # #option 1
  # coeffs_m_r <- data.frame(coefficients=c(1.515945, -0.0008673127, 2.725194e-05), row.names = c('b0','b1','b2') )
  # #option 2
  # coeffs_m_r <- data.frame(coefficients=c(1.238563, 0.0002829483, 2.65235e-05),   row.names = c('b0','b1','b2') )
  # #option 3
  # coeffs_m_r <-  data.frame(coefficients=c(1.731417, 0.003432009, 1.273215e-05),  row.names = c('b0','b1','b2')
  # read from file
  coeffs_m_r <- read.csv(file.path(out_dir, 'usable_Mag_Rrup_coeffs.csv'), row.names=1)
  # plot M/R distribution
  png(file=file.path(out_dir, 'usable_Mag_Rrup_range.png'), width=500, height=500)
  plot(rrup_array,mag_array,pch=19,xlim=c(1,1000),ylim=c(1,8))
  grid()
}


#plot M/R limits
line_mag_rrup <- data.frame(seq(1,1000,20),coeffs_m_r['b0','coefficients'] +
                                           coeffs_m_r['b1','coefficients'] *seq(1,1000,20) +
                                           coeffs_m_r['b2','coefficients'] *seq(1,1000,20)^2)
lines(line_mag_rrup[,1],line_mag_rrup[,2],col=2)
abline(v = coeffs_m_r['max_rrup','coefficients'],col=2,lty=2)
if (!flag_reg) dev.off()
  
# Output 
# ---------------------------
#save coefficients
write.csv(coeffs_m_r, file=file.path(out_dir, 'usable_Mag_Rrup_coeffs.csv'))



