/*********************************************
Stan program to obtain VCM parameters lower dimensions 
is used (event terms/station terms/ attenuation coefficient).

This model explicitly estimates the latent event terms, station terms, and
anelastic attenuation coefficients. 
This model includes a spatially varying earthquake term, a spatially 
varying station term, a spatially independent station term, a spatially 
varying geometrical spreading, a spatially varying Vs30 term, a partially 
spatially correlated cell-specific anelastic attenuation, and the 
between and within event residuals. 
The spatially varying terms are modeled as chelosky decomposition of the
kernel function multiplied with standard normal variates.
The cell-path distance matrix is stored in a CSR format
 ********************************************/

data {
  int N;      // number of records
  int NEQ;    // number of earthquakes
  int NSTAT;  // number of stations
  int NCELL;  // number of cells
  int NCELL_SP; //number of non-zero elements
      
  //event and station ID
  int<lower=1,upper=NEQ> eq[N];     // event id (in numerical order from 1 to last)
  int<lower=1,upper=NSTAT> stat[N]; // station id (in numerical order from 1 to last)

  //observations
  vector[N] Y; // median predictions for each record with anelasic attenuation taken out

  //mean ground motion
  vector[N] rec_mu; 
  //ergodic geometrical spreading and Vs30
  real c_2_erg;
  real c_3_erg;
  //ergodic anelastic attenuation coefficient
  real c_a_erg;
  
  //covariates
  vector[N]     x_2; //geometrical spreading
  vector[NSTAT] x_3; //Vs30 scaling

  //Earthquake, Station, and Cell coordinates
  vector[2] X_e[NEQ];   // event coordinates for each record
  vector[2] X_s[NSTAT]; // station coordinates for each record
  vector[2] X_c[NCELL]; //cell coordinates
  
  //cell distance matrix
  vector[NCELL_SP]    RC_val;  // non-zero values
  array[NCELL_SP] int RC_w;    // indices
  array[N+1]      int RC_u;    // row start indices
}

transformed data {
  real delta = 1e-9;

  //compute distances
  matrix[NEQ, NEQ] dist_e;
  matrix[NSTAT, NSTAT] dist_s;
  matrix[NCELL, NCELL] dist_c;
          
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

  //compute cell distance
  for(i in 1:NCELL) {
    for(j in i:NCELL) {
      real d_c = distance(X_c[i],X_c[j]);
      dist_c[i,j] = d_c;
      dist_c[j,i] = d_c;
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
  //geometrical spreading and vs30 hyper-parameters
  real<upper=0.0>  mu_2p;
  real<lower=0.0>  ell_2p;
  real<lower=0.0>  omega_2p;
  real             mu_3s;
  real<lower=0.0>  ell_3s;
  real<lower=0.0>  omega_3s;
  //attenuation cells
  real<upper=0.0>  mu_cap;
  real<lower=0.0>  ell_ca1p;
  real<lower=0.0>  omega_ca1p; 
  real<lower=0.0>  omega_ca2p;  
  
  //spatially correlated coefficients
  real dc_0;             //constant shift
  vector[NSTAT] dc_1bs;  //zero correlation station term
  //standardized normal variables for spatially correlated coefficients
  vector[NEQ]   z_1e;   
  vector[NSTAT] z_1as;  
  //standardized normal variable for geometrical spreading
  vector[NEQ]   z_2p;
  //standardized normal variable for Vs30
  vector[NSTAT] z_3s;  
  //cell-specific attenuation
  vector<upper=0>[NCELL]  c_cap;  
      
  //between event terms
  vector[NEQ]   dB;
}

transformed parameters{
  //spatially correlated coefficients
  vector[NEQ]   dc_1e;   //spatially varying eq coeff
  vector[NSTAT] dc_1as;  //spatially varying station term
  vector[NEQ]   c_2p;    //spatially varying geometrical spreading term
  vector[NSTAT] c_3s;    //spatially varying Vs30 term
  
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
  
  //Spatially latent variable for spatially correlated geometrical spreading
  {
    matrix[NEQ,NEQ] COV_2p;
    matrix[NEQ,NEQ] L_2p;
    for(i in 1:NEQ) {
      //diagonal terms
      COV_2p[i,i] = omega_2p^2 + delta;
      //off-diagonal terms
      for(j in (i+1):NEQ) {
        real C_2p = (omega_2p^2 * exp(-dist_e[i,j]/ell_2p));
        COV_2p[i,j] = C_2p;
        COV_2p[j,i] = C_2p;
      }
    }
    L_2p = cholesky_decompose(COV_2p);
    c_2p = mu_2p + L_2p * z_2p;
  }
  
  //Spatially latent variable for vs30 spatially varying term
  { 
    matrix[NSTAT,NSTAT] COV_3s;
    matrix[NSTAT,NSTAT] L_3s;
    for(i in 1:NSTAT) {
      //diagonal terms
      COV_3s[i,i] = omega_3s^2 + delta;
      //off-diagonal terms
      for(j in (i+1):NSTAT) {
        real C_3s = (omega_3s^2  * exp(-dist_s[i,j]/ell_3s));
        COV_3s[i,j] = C_3s;
        COV_3s[j,i] = C_3s;
      }
    }
    L_3s = cholesky_decompose(COV_3s);
    c_3s = mu_3s + L_3s * z_3s;
  }
  
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
  //geometrical and vs30 hyper-parameters
  ell_2p ~ inv_gamma(2.,50);
  ell_3s ~ inv_gamma(2.,50);
  omega_2p ~ exponential(5);
  omega_3s ~ exponential(5);
  //cell specific attenuation hyper-parameters
  ell_ca1p ~ inv_gamma(2.,50);
  omega_ca1p ~ exponential(250);
  omega_ca2p ~ exponential(250);
  
  //constant shift
  dc_0 ~ normal(0.,0.1);
  
  //standardized event contributions to GP
  z_1e ~ std_normal();

  //standardized station contributions to GP
  z_1as ~ std_normal();
  
  //station contributions with zero correlation length
  dc_1bs ~ normal(0,omega_1bs);
  
  //constant shift of gs from ergodic coeff
  mu_2p ~ normal(c_2_erg,0.2);
  //standardized geometrical spreading contributions to GP
  z_2p ~ std_normal();
  
  //constant shift of Vs30 scaling from ergodic coeff
  mu_3s ~ normal(c_3_erg,0.2);
  //standardized vs30 contributions to GP
  z_3s ~ std_normal();

  //cell attenuation
  mu_cap ~ normal(c_a_erg, 0.01); //mean anelastic attenuation
  //generate latent variables for spatially correlated anelastic attenuation cells
  {
    matrix[NCELL, NCELL] COV_cap;
    for(i in 1:NCELL) {
      //diagonal terms
      COV_cap[i,i] = omega_ca1p^2 + omega_ca2p^2 + delta;
      //off-diagonal terms
      for(j in (i+1):NCELL) {
        real C_cap = (omega_ca1p^2 * exp(-dist_c[i,j]/ell_ca1p));  //negative exp cov matrix
        COV_cap[i,j] = C_cap;
        COV_cap[j,i] = C_cap;
      }
    }
    c_cap ~ multi_normal(rep_vector(mu_cap,NCELL),COV_cap);
  }
    
  //anelastic attenuation
  inatten = csr_matrix_times_vector(N, NCELL, RC_val, RC_w, RC_u, c_cap); 
  
  //Mean non-ergodic including dB
  rec_nerg_dB = rec_mu + dc_0 + dc_1e[eq] + dc_1as[stat] + dc_1bs[stat] + c_2p[eq].*x_2 + c_3s[stat].*x_3[stat] + inatten + dB[eq];
  
  Y ~ normal(rec_nerg_dB,phi_0);
}

