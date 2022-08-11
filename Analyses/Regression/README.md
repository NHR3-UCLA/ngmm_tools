# Non-ergodic Ground Motion Model Types:

 * Type-1: Three non-ergodic terms: 
 
 $$
  f_{nerg}(M,R_{rup},V_{S30},..., \vec{t_{E}}, \vec{t_{S}}) = f_{erg}(M,R_{rup},V_{S30},...) + \delta  c_{1,E}(\vec{t_{E}}) + \delta c_{1a,S}(\vec{t_{S}}) + \delta  c_{1b,S}(\vec{t_{S}})
 $$
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 a spatially varying earthquake constant ( $\delta  c_{1,E}$ ), a spatially varying site constant ( $\delta c_{1a,S}$ ), and a spatially independent site <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 constant ( $\delta  c_{1b,S}$ ). 
 
 * Type-2: Four non-ergodic terms: 
 
 $$
 \begin{aligned}
  f_{nerg}(M,R_{rup},V_{S30},..., \vec{t_{E}}, \vec{t_{S}}) =& \left( f_{erg}(M,V_{S30},...) - c_{a~erg}~R_{rup} \right) + \delta  c_{1,E}(\vec{t_{E}}) + \delta c_{1a,S}(\vec{t_{S}}) + \delta  c_{1b,S}(\vec{t_{S}}) + \\
  & \mathbf{c}_{ca,P} \cdot \Delta R 
 \end{aligned}
 $$
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 a spatially varying earthquake constant ( $\delta c_{1,E}$ ), a spatially varying site constant ( $\delta c_{1a,S}$ ), a spatially independent site<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 constant ( $\delta c_{1b,S}$ ), and cell-specific anelastic attenuation ( $\mathbf{c}_{ca,P} $). 
 
 * Type-3: Six non-ergodic terms: 
 
  $$
 \begin{aligned}
  f_{nerg}(M,R_{rup},V_{S30},..., \vec{t_{E}}, \vec{t_{S}}) =& \left( f_{erg}(M,V_{S30},...) - (c_2 ~ f_{gs}(M,R) + c_3 ~ f_{V_{S30}}(V_{S30})) + c_{a~erg} ~ R_{rup}) \right) + \\
  & \delta  c_{1,E}(\vec{t_{E}}) + \delta c_{1a,S}(\vec{t_{S}}) + \delta  c_{1b,S}(\vec{t_{S}}) + \\
  &  c_{2,E}(\vec{t_{E}}) f_{gs}(M,R_{rup}) + \delta c_{1a,S}(\vec{t_{S}}) +  \mathbf{c}_{ca,P} \cdot \Delta R
 \end{aligned}
 $$
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 a spatially varying earthquake constant ( $\delta c_{1,E}$ ), a spatially varying site constant ( $\delta c_{1a,S}$ ), a spatially independent site<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 constant ( $\delta c_{1b,S}$ ),  a spatially varying geometrical spreading coefficient ( $c_{2,P}$ ), a spatially varying $V_{S30}$ scaling ( $c_{3,S}$ ), <br> 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 and cell-specific anelastic attenuation ( $\mathbf{c}_{ca,P} $). 

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
