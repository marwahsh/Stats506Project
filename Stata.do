*------------------------------------------------------------------------------*
* Stats 506, Fall 2018
* Group 5 Group Project
* 
* Tutorial on Logistic Regression with Model Diagnostics on Stata
*  
* Data: The National Institute of Diabetes and Digestive and Kidney Diseases conducted a study on 768 adult female Pima Indians living near Phoenix.
* From faraway v1.0.7 by Julian Faraway

* Dervived data sets: pima.csv
*
* Author: Gabrielle Angela Santos
* Updated: November 27, 2018
*------------------------------------------------------------------------------*
*80: ---------------------------------------------------------------------------

* Install HL package: ----------------------------------------------------------
net from https://www.sealedenvelope.com/
// Follow instructions to install hl

* Import dataset: --------------------------------------------------------------

import delimited using pima.csv, clear
summarize

* Clean data: ------------------------------------------------------------------
drop if glucose == 0 | diastolic == 0 | triceps ==0 | insulin == 0 | bmi == 0
summarize 

* Implement logistic model: ----------------------------------------------------
logit test pregnant glucose diastolic triceps insulin bmi diabetes age

* Backward elimination using BIC: ----------------------------------------------
// Use stepwise to determine the order at which to drop the variables
stepwise, pr(0.05): logit test pregnant glucose diastolic triceps insulin bmi diabetes age
// Drop order: diastolic, insulin, triceps, pregnant


// Check BIC for each step

// Full model: BIC = 397.7626
logit test pregnant glucose diastolic triceps insulin bmi diabetes age
estat ic

// Model 1 (drop diastolic): BIC = 391.8057
logit test pregnant glucose triceps insulin bmi diabetes age
estat ic

// Model 2 (drop insulin): BIC = 386.223
logit test pregnant glucose triceps bmi diabetes age
estat ic

// Model 3 (drop triceps): BIC = 380.7127
logit test pregnant glucose bmi diabetes age
estat ic

// Model 4 (drop pregnant): BIC = 377.0913
logit test glucose bmi diabetes age
estat ic

// Check p-values of remaining model
logit test glucose bmi diabetes age
// All variables are significant

* Model fitting: ---------------------------------------------------------------
predict p

* Classification report: -------------------------------------------------------
estat classification

* Hosmer-Lemeshow Test: --------------------------------------------------------
estat gof, group(10)
egen dec=cut(p), at(0(0.1)1)
hl test p, q(dec) plot

* ROC Curve: -------------------------------------------------------------------
lroc, title("Receiving Operator Characteristic Curve") xtitle("False Positive Rate") ytitle("True Positive Rate")



