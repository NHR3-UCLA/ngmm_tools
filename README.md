# Non-ergodic Methodology and Modeling Tools

This repository contains software tools for developing Nonergodic Ground Motion Models (NGMMs) based on the varying coefficient (Landwehr et al., 2016) and cell-specific anelastic attention approach (Dawood and Rodriguez‐Marek, 2013). 
Developed tools are available for R using the statistical package (R-INLA, https://www.r-inla.org/) and in python using the CMDSTAN and PYSTAN interface packages for the Bayesian software (Stan, https://mc-stan.org/). Documentation and detailed instructions on the use of the developed tools is provided in [Lavrentiadis et al., 2022a GIRS report](https://www.risksciences.ucla.edu/girs-reports/2022/04). A general introduction and considerations for the development of NGMMs are presented in Lavrentiadis et al., 2022b.

## Home Page
The project's home page with links to the various project deliverables is: https://www.risksciences.ucla.edu/nhr3/ngmm
The project's data are accesible through [DesignSafe](https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-5771)

## Folder Structure
The main folder ``Analyses`` contains all the regression, prediction, hazard implementation, testing, and library scripts. 
Within the ``Analyses``  folder,  ``Data_Preparation`` includes preprocessing scripts to prepare the ground-motion data for the NGMM regression. ``Regression`` contains the Jupyter notebooks for running the NGMM regressions using Stan and INLA. ``Predictions`` includes the scripts for the conditional predictions for new scenarios based on the regression results. ``Code_Verification`` contains the codes associated with the verification exercise. 
Lastly, folders ``Python_lib``, ``R_lib``, and ``Stan_lib`` contain various scripts invoked in the main functions.

The main folder ``Data`` mirrors the structure of the ``Analyses`` folder and contains all the input and output files.

The ``Raw_files`` includes the files used to construct the synthetic datasets for the verification exercise.

    .
    |--Analyses
    |     |--Data_Preparation
    |     |--Regression
    |     |--Predictions
    |     |--Code_Verification
    |     |--Python_lib
    |     |--R_lib
    |     |--Stan_lib
    |
    |--Data
    |     |--Regression
    |     |--Predictions
    |     |--Code_Verification
    |     
    |--Raw_files

An example regression dataset can be downloaded with ``source ./download_exampfiles.sh``.
The syntetic datasets and raw metadata can be downloaded by running ``source ./download_data.sh`` and ``source ./download_rawfiles.sh``, respectively.

## Acknowledgments 
Financial support by the California Department of Transportation and Pacific Gas & Electric Company is greatly appreciated.  

## References
Dawood, H. M., & Rodriguez‐Marek, A. (2013). A method for including path effects in ground‐motion prediction equations: An example using the M w 9.0 Tohoku earthquake aftershocks. Bulletin of the Seismological Society of America, 103(2B), 1360-1372.

Landwehr, N., Kuehn, N. M., Scheffer, T., & Abrahamson, N. (2016). A nonergodic ground‐motion model for California with spatially varying coefficients. Bulletin of the Seismological Society of America, 106(6), 2574-2583.

Lavrentiadis, G., Nicolas, K. M., Bozorgnia, Y., Seylabi, E., Meng, X., Goulet, C., & Kottke, A. (2022a) Non‐ergodic Methodology and Modeling Tools. Natural Hazards Risk and Resiliency Research Center: The Garrick Institute for the Risk Sciences, University of California, Los Angeles

Lavrentiadis, G., Abrahamson, N. A., Nicolas, K. M., Bozorgnia, Y., Goulet, C. A., Babič, A., ... & Walling, M. (2022b). Overview and Introduction to Development of Non-Ergodic Earthquake Ground-Motion Models. Bulletin of Earthquake Engineering
