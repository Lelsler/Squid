# Climate impacts on the Mexican Humboldt squid fishery

This GitHub repository contains the data, code, and output for:
Elsler, L.G., Frawley, T.H., Britten, G.L., Crowder, L.B., DuBois, T.C., Radosavljevic, S., Gilly, W.F., Crépin, A.-S., Schlüter, M., 2021. Social relationship dynamics mediate climate impacts on income inequality: evidence from the Mexican Humboldt squid fishery. Reg Environ Change 21, 35. [link](https://doi.org/10.1007/s10113-021-01747-5)

Supplementary information available here [link](https://static-content.springer.com/esm/art%3A10.1007%2Fs10113-021-01747-5/MediaObjects/10113_2021_1747_MOESM1_ESM.pdf)

Here's a quick run down of the files that assemble the input data and where this data is saved:
    
    CODE - A folder containing the code used to analyze the processed data and prepare tables/figures
           * 0_pe_fitting_CT.R - export price fitting
           * 1_model_master.py - model master file for reference
           * 2_timeseries.py - catch and price timeseries 
           * 3_interventions.py - demand and trader competition intervention
           * 4_parameter_sweep.py - climate-driven changes in price difference 
           * 5_SI.py - SI figures

    DATA - A folder containing the raw data, the code used to process the raw data, and the data generated from the processing
    
    FIGURES - A folder containing all figures from the article and SI tables (parameter values, datasets)


ComTrade files containing the squid market price data are very large (>500MB) and are not included in the repository. Please email Laura Elsler (l.elsler AT outlook.com) to request access to these files.

Acknowledgment: First and foremost, we are grateful to our interviewees and the fishery participants for their time and for sharing their knowledge about the squid fishery. We extend our thanks in the field to the MAREA team that was instrumental in introducing L.G.E. to Mexican small-scale fisheries. We thank dataMares and Universidad Nacional Aut ́ onoma de M ́ exico (UNAM) for sharing fishery data. We also thank our colleagues for technical help, comments, and discussion of the manuscript: F. Diekert, J. Norberg, R. Blasiak, J. Gars, I. Fetzer, K. Arroyo-Ramirez, U. Markaida, and M. Oostdijk.


