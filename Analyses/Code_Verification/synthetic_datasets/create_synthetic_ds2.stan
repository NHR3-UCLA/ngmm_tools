/*********************************************
Stan program to create a synthetic data-set 
with a zero correlation length station term,
an earthquake and station spatially varying 
terms, spatially correlated cell specific 
anelastic attenuation and between and 
within event aleatory terms
********************************************/

data {
  int N;      // number of records
  int NEQ;    // number of earthquakes
  int NSTAT;  // number of stations
  int NCELL;  // number of stations
    
  //event and station ID
  int<lower=1,upper=NEQ> eq[N];     // event id (in numerical order from 1 to last)
  int<lower=1,upper=NSTAT> stat[N]; // station id (in numerical order from 1 to last)

  //earthquake and station coordinates
  vector[2] X_e[NEQ];
  vector[2] X_s[NSTAT];
  vector[2] X_c[NCELL];
  //cell distance matrix
  vector[NCELL] RC[N];  // cell paths for each record
    
  //assumed hyper-parameters
  //earthquake and site constants
  real omega_0;
  real omega_1e;
  real omega_1as;
  real omega_1bs;
  real ell_1e;
  real ell_1as;
  //cell specific anelastic attenuation
  real c_cap_erg;
  real omega_cap_mu;
  real omega_ca1p;
  real omega_ca2p;
  real ell_ca1p;
  //aleatory terms
  real tau_0;
  real phi_0;
  
  //mean of ergodic GMM
  vector[N] mu_gmm;
}

transformed data {
  real delta = 1e-9;

  //priors means
  real          dc_0_mu   = 0.;
  vector[NEQ]   dc_1e_mu  = rep_vector(0.,NEQ);
  vector[NSTAT] dc_1as_mu = rep_vector(0.,NSTAT);
  vector[NSTAT] dc_1bs_mu = rep_vector(0.,NSTAT);
}

parameters {}

model {}

generated quantities {
  //coefficient samples
  real          dc_0;
  vector[NEQ]   dc_1e;   //spatially varing terms
  vector[NSTAT] dc_1as;
  vector[NSTAT] dc_1bs;
  //anelastic attenuation
  real          c_cap_mu;
  vector[NCELL] c_cap;
  //samples of aleatory terms
  vector[NEQ]   dB;
  vector[N]     dW;    
  //gm samples
  vector[N]     Y_var_ceoff;
  vector[N]     Y_nerg_med;
  vector[N]     Y_aleat;
  vector[N]     Y_inattent;
  vector[N]     Y_tot;

  //latent variable for constant shift
  {
    dc_0 = normal_rng(dc_0_mu,omega_0);
  }

  //generate latent variable for spatially varying earthquake term
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
    dc_1e = multi_normal_rng(dc_1e_mu, COV_1e);
  }

  //generate latent variable for spatially varying station term
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
    dc_1as = multi_normal_rng(dc_1as_mu, COV_1as);
  }
  
  //generate latent variable for independent varying station term
  {
    for(i in 1:NSTAT) {
      dc_1bs[i] = normal_rng(dc_1bs_mu[i], omega_1bs);
    }
  }
  
  //generate latent variables for spatially correlated anelastic attenuation cells
  {
    matrix[NCELL, NCELL] COV_cap;
    
    c_cap_mu = normal_rng(c_cap_erg, omega_cap_mu);
    
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
    c_cap = multi_normal_rng(rep_vector(c_cap_mu,NCELL),COV_cap);
  }

  //generate aleatory terms
  {
    for(i in 1:N) {
      dW[i] = normal_rng(0., phi_0);
    }
    for(i in 1:NEQ) {
      dB[i] = normal_rng(0., tau_0);
    }
  }
 
  //generate gm random samples
  //add contributions of spatially varying terms
  {
    Y_var_ceoff =  dc_0 + dc_1e[eq] + dc_1as[stat] + dc_1bs[stat];
  }
  //add effect of anelastic attenuation
  {
    for(i in 1:N)
      Y_inattent[i] =  dot_product(c_cap,RC[i]);
  }

  //median ground motion
  Y_nerg_med =  mu_gmm + Y_var_ceoff + Y_inattent;
  //aleatory variability
  Y_aleat = dW + dB[eq];
  //total gm
  Y_tot = Y_nerg_med + Y_aleat;
}


