/*********************************************
Stan program to obtain VCM parameters
lower dimensions is used (event terms/station terms)

This model explicitly estimates the latent event terms and station terms.
This model includes a spatially varying earthquake term, a spatially 
varying station term, a spatially independent station term, and the 
between and within event residuals. 
The spatially varying terms are modeled as chelosky decomposition of the
kernel function multiplied with standard normal variates.
 ********************************************/

data {
  int N;      // number of records
  int NEQ;    // number of earthquakes
  int NSTAT;  // number of stations
  
  //event and station ID
  int<lower=1,upper=NEQ> eq[N];     // event id (in numerical order from 1 to last)
  int<lower=1,upper=NSTAT> stat[N]; // station id (in numerical order from 1 to last)

  //observations
  vector[N] Y; // median predictions for each record with anelasic attenuation taken out

  //mean ground motion
  vector[N] rec_mu; 

  //Earthquake, Station coordinates
  vector[2] X_e[NEQ];   // event coordinates for each record
  vector[2] X_s[NSTAT]; // station coordinates for each record
}

transformed data {
  real delta = 1e-9;

  //compute distances
  matrix[NEQ, NEQ] dist_e;
  matrix[NSTAT, NSTAT] dist_s;
        
  //compute earthquake distances
  for(i in 1:NEQ) {
    for(j in i:NEQ) {
      real d_e = distance(X_e[i],X_e[j]);
      dist_e[i,j] = d_e;
      dist_e[j,i] = d_e;
    }
  }
  
  //compute station distance
  for(i in 1:NSTAT) {
    for(j in i:NSTAT) {
      real d_s = distance(X_s[i],X_s[j]);
      dist_s[i,j] = d_s;
      dist_s[j,i] = d_s;
    }
  }
}

parameters {
  //Aleatory Variability Terms
  real<lower=0> phi_0;  // phi_0 - remaining aleatory variability of within-event residuals
  real<lower=0> tau_0;  // tau_0 - remaining aleatory variability of between-event residuals
  
  //Epistemic Uncertainty Terms
  real<lower=0.0>  ell_1e;
  real<lower=0.0>  omega_1e;
  real<lower=0.0>  ell_1as;
  real<lower=0.0>  omega_1as;
  real<lower=0.0>  omega_1bs;
 
  //spatially correlated coefficients
  real dc_0;             //constant shift
  vector[NSTAT] dc_1bs;  //zero correlation station term

  //standardized normal variables for spatially correlated coefficients
  vector[NEQ]   z_1e;   //spatially varying eq coeff
  vector[NSTAT] z_1as;  //spatially varying stat coeff
  
  //between event terms
  vector[NEQ]   dB;
}

transformed parameters{
  //Spatially correlated coefficients
  vector[NEQ]   dc_1e;   //spatially varying eq coeff
  vector[NSTAT] dc_1as;  //spatially varying stat coeff

  //spatillay latent variable for event contributions to GP
  {
    matrix[NEQ,NEQ] COV_1e;
    matrix[NEQ,NEQ] L_1e;
    for(i in 1:NEQ) {
      //diagonal terms
      COV_1e[i,i] = omega_1e^2 + delta;
      //off-diagonal terms
      for(j in (i+1):NEQ) {
        real C_1e = (omega_1e^2 * exp(-dist_e[i,j]/ell_1e));
        COV_1e[i,j] = C_1e;
        COV_1e[j,i] = C_1e;
      }
    }
    L_1e = cholesky_decompose(COV_1e);
    dc_1e = L_1e * z_1e;
  }


  //Spatially latent variable for station contributions to GP
  { 
    matrix[NSTAT,NSTAT] COV_1as;
    matrix[NSTAT,NSTAT] L_1as;
    for(i in 1:NSTAT) {
      //diagonal terms
      COV_1as[i,i] = omega_1as^2 + delta;
      //off-diagonal terms
      for(j in (i+1):NSTAT) {
        real C_1as = (omega_1as^2  * exp(-dist_s[i,j]/ell_1as));
        COV_1as[i,j] = C_1as;
        COV_1as[j,i] = C_1as;
      }
    }
    L_1as = cholesky_decompose(COV_1as);
    dc_1as = L_1as * z_1as;
  }

}

model {
  //non-ergodic mean
  vector[N] rec_nerg_dB;
  
  //Aleatory Variability Terms
  phi_0 ~ lognormal(-1.20,0.3);
  tau_0 ~ lognormal(-1,0.3);
  //Station and earthquake paramters
  dB ~ normal(0,tau_0);
  
  //non-ergodic hyper-parameters
  ell_1e  ~ inv_gamma(2.,50);
  ell_1as ~ inv_gamma(2.,50);
  omega_1e  ~ exponential(5);
  omega_1as ~ exponential(5);
  omega_1bs ~ exponential(5);
  
  //constant shift
  dc_0 ~ normal(0.,0.1);
  
  //standardized event contributions to GP
  z_1e ~ std_normal();

  //standardized station contributions to GP
  z_1as ~ std_normal();
  
  //station contributions with zero correlation length
  dc_1bs ~ normal(0,omega_1bs);

  //Mean non-ergodic including dB
  rec_nerg_dB = rec_mu + dc_0 + dc_1e[eq] + dc_1as[stat] + dc_1bs[stat] + dB[eq];
  
  Y ~ normal(rec_nerg_dB,phi_0);
}

