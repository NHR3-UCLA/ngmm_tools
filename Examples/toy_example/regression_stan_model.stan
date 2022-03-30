/*********************************************
Stan program for toy example

 ********************************************/

data {
  int N;  // number of observations
  int NG; // number of grid points
  
  //grid IDs
  int<lower=1,upper=NG> gid[N]; // grid id

  //observations
  vector[N] Y; 

  //coordinates
  vector[2] X_g[NG];
}

transformed data {
  real delta = 1e-9;
}

parameters {
  //aleatory std
  real<lower=0> sigma; 
  //kernel hyper-paramters
  real<lower=0.0>  ell;
  real<lower=0.0>  omega;
 
  //model coefficient
  real c_0;
  //standardized normal variables for spatially correlated coefficient
  vector[NG] z_1;
}

transformed parameters {
  //spatially correlated coefficient
  vector[NG] c_1;

  {
    matrix[NG,NG] COV_1;
    matrix[NG,NG] L_1;
    for(i in 1:NG) {
      for(j in i:NG) {
        real C_1 = (omega^2 * exp(-distance(X_g[i],X_g[j])/ell));
        COV_1[i,j] = C_1;
        COV_1[j,i] = C_1;
      }
      COV_1[i,i] += delta;
    }
    L_1 = cholesky_decompose(COV_1);
    c_1 = L_1 * z_1;
  }
}


model {
  //hyper-parameters
  ell   ~ inv_gamma(2.,50);
  omega ~ exponential(5);
  sigma ~ lognormal(-1,0.3);

  //constant shift
  c_0 ~ normal(0.,0.1);
  //standardized normal variables for spatially correlated coefficient
  z_1 ~ std_normal();

  //likelihood
  Y ~ normal(c_0 + c_1[gid], sigma);
}

