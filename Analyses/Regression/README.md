# File Descriptions

### INLA
 * Type-1 NGMM: ``nonerg_gmm_regression_type1_inla.ipynb``
 * Type-2 NGMM with spatially uncorrelated anelastic attenuation cells: ``nonerg_gmm_regression_type2_uncorrcells_inla.ipynb`` 
 * Type-3 NGMM with spatially uncorrelated anelastic attenuation cells: ``nonerg_gmm_regression_type3_uncorrcells_inla.ipynb``

### CMDSTAN
 * Type-1 NGMM: ``nonerg_gmm_regression_type1_cmdstan.ipynb``
 * Type-2 NGMM with spatially correlated anelastic attenuation cells: ``nonerg_gmm_regression_type2_corrcells_cmdstan.ipynb`` 
 * Type-2 NGMM with spatially uncorrelated anelastic attenuation cells: ``nonerg_gmm_regression_type2_uncorrcells_cmdstan.ipynb``
 * Type-3 NGMM with spatially correlated anelastic attenuation cells: ``nonerg_gmm_regression_type3_corrcells_cmdstan.ipynb``
 * Type-3 NGMM with spatially uncorrelated anelastic attenuation cells: ``nonerg_gmm_regression_type3_uncorrcells_cmdstan.ipynb``

### PYSTAN
 * Type-1 NGMM: ``nonerg_gmm_regression_type1_pystan.ipynb``
 * Type-2 NGMM with spatially correlated anelastic attenuation cells: ``nonerg_gmm_regression_type2_corrcells_pystan.ipynb``
 * Type-2 NGMM with spatially uncorrelated anelastic attenuation cells: ``nonerg_gmm_regression_type2_uncorrcells_pystan.ipynb``
 * Type-3 NGMM with spatially correlated anelastic attenuation cells: ``nonerg_gmm_regression_type3_corrcells_pystan.ipynb``
 * Type-3 NGMM with spatially uncorrelated anelastic attenuation cells: ``nonerg_gmm_regression_type3_uncorrcells_pystan.ipynb``

# Non-ergodic Ground Motion Model Types:

 * Type-1: Three non-ergodic terms: a spatially varying earthquake constant ( $c_{1,E}$ ), a spatially varying site constant ( $c_{1a,S}$ ), and a spatially independent site constant ( $c_{1b,S}$ ). 
 * Type-2: Four non-ergodic terms: a spatially varying earthquake constant ( $c_{1,E}$ ), a spatially varying site constant ( $c_{1a,S}$ ), a spatially independent site constant ( $c_{1b,S}$ ), and cell-specific anelastic attenuation ( $c_{ca,P} $). 
 * Type-3: Six non-ergodic terms: a spatially varying earthquake constant ( $c_{1,E}$ ), a spatially varying site constant ( $c_{1a,S}$ ), a spatially independent site constant ( $c_{1b,S}$ ),  a spatially varying geometrical spreading coefficient ( $c_{2,P}$ ), a spatially varying $V_{S30}$ scaling ( $c_{3,S}$ )and cell-specific anelastic attenuation ( $c_{ca,P} $). 

