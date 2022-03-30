/*********************************************
Stan program to obtain VCM parameters lower dimensions 
is used (event terms/station terms/ attenuation coefficient).

This model explicitly estimates the latent event terms, station terms, and
anelastic attenuation coefficients. 
This model includes a spatially varying earthquake term, a spatially 
varying station term, a spatially independent station term, a partially 
spatially correlated cell-specific anelastic attenuation, and the 
between and within event residuals. 
The spatially varying terms are modeled as multi-normal distributions
with kernel function as covariance matrix.
 ********************************************/

data {
  int N;      // number of records
  int NEQ;    // number of earthquakes
  int NSTAT;  // number of stations
  int NCELL;  // number of cells
    
  //event and station ID
  int<lower=1,upper=NEQ> eq[N];     // event id (in numerical order from 1 to last)
  int<lower=1,upper=NSTAT> stat[N]; // station id (in numerical order from 1 to last)

  //observations
  vector[N] Y; // median predictions for each record with anelasic attenuation taken out

  //mean ground motion
  vector[N] rec_mu; 
  //ergodic anelastic attenuation coefficient
  real c_a_erg;

  //Earthquake, Station, and Cell coordinates
  vector[2] X_e[NEQ];   // event coordinates for each record
  vector[2] X_s[NSTAT]; // station coordinates for each record
  vector[2] X_c[NCELL]; //cell coordinates
  
  //cell distance matrix
  matrix[N, NCELL] RC;  // cell paths for each record
}

transformed data {
  real delta = 1e-9;
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
  //attenuation cells
  real<upper=0.0>  mu_cap;
  real<lower=0.0>  ell_ca1p;
  real<lower=0.0>  omega_ca1p; 
  real<lower=0>    omega_ca2p;      // std of cell-specific attenuation
  
   
  //spatially correlated coefficients
  real dc_0;             //constant shift
  vector[NEQ]   dc_1e;   //spatially varying eq coeff
  vector[NSTAT] dc_1as;  //spatially varying stat coeff
  vector[NSTAT] dc_1bs;  //zero correlation station term
  //cell-specific attenuation
  vector<upper=0>[NCELL]  c_cap;
    
  //between event terms
  vector[NEQ]   dB;
}

model {
  //non-ergodic mean
  vector[N] rec_nerg_dB;
  //effect anelastic attenuation
  vector[N] inatten;
  
  //Aleatory Variability Terms
  phi_0 ~ lognormal(-1.20,0.3);
  tau_0 ~ lognormal(-1,0.3);
  //Station and earthquake parameters
  dB ~ normal(0,tau_0);
  
  //non-ergodic hyper-parameters
  ell_1e  ~ inv_gamma(2.,50);
  ell_1as ~ inv_gamma(2.,50);
  omega_1e  ~ exponential(5);
  omega_1as ~ exponential(5);
  omega_1bs ~ exponential(5);
  //cell specific attenuation hyper-parameters
  ell_ca1p ~ inv_gamma(2.,50);
  omega_ca1p ~ exponential(250);
  omega_ca2p ~ exponential(250);
  
  //constant shift
  dc_0 ~ normal(0.,0.1);
  
  //spatillay latent variable for event contributions to GP
  {
    matrix[NEQ,NEQ] COV_1e;
    
    for(i in 1:NEQ) {
      for(j in i:NEQ) {
        real d_e;
        real C_1e;
        
        d_e = distance(X_e[i],X_e[j]);
  
        C_1e = (omega_1e^2 * exp(-d_e/ell_1e));
  
        COV_1e[i,j] = C_1e;
        COV_1e[j,i] = C_1e;
      }
      COV_1e[i,i] += delta;
    }
    dc_1e ~ multi_normal(rep_vector(0.,NEQ), COV_1e);
  }

  //Spatially latent variable for station contributions to GP
  { 
    matrix[NSTAT,NSTAT] COV_1as;

    for(i in 1:NSTAT) {
      for(j in i:NSTAT) {
        real d_s;
        real C_1as;
  
        d_s = distance(X_s[i],X_s[j]);
        
        C_1as = (omega_1as^2  * exp(-d_s/ell_1as));
  
        COV_1as[i,j] = C_1as;
        COV_1as[j,i] = C_1as;
      }
      COV_1as[i,i] += delta;
    }
    dc_1as ~ multi_normal(rep_vector(0.,NSTAT), COV_1as);
  }
  
  //station contributions with zero correlation length
  dc_1bs ~ normal(0,omega_1bs);

  //mean cell attenuation
  mu_cap ~ normal(c_a_erg, 0.01); //mean anelastic attenuation
  
  //generate latent variables for spatially correlated anelastic attenuation cells
  {
    matrix[NCELL, NCELL] COV_cap;
    
    for(i in 1:NCELL) {
      for(j in i:NCELL) {
        real d_c;
        real C_cap;
        
        d_c = distance(X_c[i],X_c[j]);
  
        C_cap = (omega_ca1p^2 * exp(-d_c/ell_ca1p));  //negative exp cov matrix
  
        COV_cap[i,j] = C_cap;
        COV_cap[j,i] = C_cap;
      }
      COV_cap[i,i] += omega_ca2p^2 + delta;
    }
    c_cap ~ multi_normal(rep_vector(mu_cap,NCELL), COV_cap);
  }
  
  //anelastic attenuation
  inatten = RC * c_cap;
  
  //Mean non-ergodic including dB
  rec_nerg_dB = rec_mu + dc_0 + dc_1e[eq] + dc_1as[stat] + dc_1bs[stat] + inatten + dB[eq];
  
  Y ~ normal(rec_nerg_dB,phi_0);
}

