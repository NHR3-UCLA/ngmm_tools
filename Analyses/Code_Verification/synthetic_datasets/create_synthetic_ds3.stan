/*********************************************
Stan program to create a synthetic data-set 
with a zero correlation length station term,
an earthquake and station spatially varying 
terms, a spatially varying geometrical spreading
and Vs30 term, spatially correlated cell specific 
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
    
  //covariates
  vector[N]     x_2; //geometrical spreading
  vector[NSTAT] x_3;//Vs30 scaling
    
  //assumed hyper-parameters
  //earthquake and site constants
  real omega_0;
  real omega_1e;
  real omega_1as;
  real omega_1bs;
  real ell_1e;
  real ell_1as;
  //geometrical spreading
  real c_2_erg;
  real omega_2;
  real omega_2p;
  real ell_2p;
  //Vs30 scaling
  real c_3_erg;
  real omega_3;
  real omega_3s;
  real ell_3s;
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
  real          c_2_mu;
  vector[NEQ]   c_2p;
  real          c_3_mu;
  vector[NSTAT] c_3s;
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

  //generate latent variables for spatially correlated geometrical spreading
  {
    matrix[NEQ,NEQ] COV_2p;
    
    c_2_mu = normal_rng(c_2_erg, omega_2);
    
    for(i in 1:NEQ) {
      for(j in i:NEQ) {
        real d_e;
        real C_2p;
        
        d_e = distance(X_e[i],X_e[j]);
  
        C_2p = (omega_2p^2 * exp(-d_e/ell_2p));
  
        COV_2p[i,j] = C_2p;
        COV_2p[j,i] = C_2p;
      }
      COV_2p[i,i] += delta;
    }
    c_2p = multi_normal_rng(rep_vector(c_2_mu,NEQ),COV_2p);
  }
  
  //generate latent variables for spatially correlated vs30 scaling
  {
    matrix[NSTAT,NSTAT] COV_3s;
    
    c_3_mu = normal_rng(c_3_erg, omega_3);

    for(i in 1:NSTAT) {
      for(j in i:NSTAT) {
        real d_s;
        real C_3s;
  
        d_s = distance(X_s[i],X_s[j]);
        
        C_3s = (omega_3s^2 * exp(-d_s/ell_3s));
  
        COV_3s[i,j] = C_3s;
        COV_3s[j,i] = C_3s;
      }
      COV_3s[i,i] += delta;
    }
    c_3s = multi_normal_rng(rep_vector(c_3_mu,NSTAT),COV_3s);
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
    Y_var_ceoff =  dc_0 + dc_1e[eq] + dc_1as[stat] + dc_1bs[stat] + c_2p[eq] .* x_2 + c_3s[stat] .* x_3[stat];
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



