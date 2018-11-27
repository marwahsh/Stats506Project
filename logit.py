#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 17:51:23 2018

@author: fanghongru
"""

import pandas as pd
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from scipy.stats import chi2


data=pd.read_csv("pima.txt",
   delim_whitespace=True,
   skipinitialspace=True)
data['intercept'] = 1.0

## Data Cleaning
data = data[data['bmi']>0]
data = data[data['diastolic']>0]
data = data[data['glucose']>0]
data = data[data['triceps']>0]
data = data[data['insulin']>0]
y=data['test']
X=data.drop(['test'], axis=1)

## First fit model
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

## Backward stepwise with BIC
# Triceps individually highly insignificant so drop the variable and assess the model
X1=X.drop(['triceps'], axis=1)
logit_model1=sm.Logit(y,X1)
result1=logit_model1.fit()
print(result1.bic)

# Drop diastolic (individually statistically insignificant) and assess the model
X2=X.drop(['triceps','diastolic'], axis=1)
logit_model2=sm.Logit(y,X2)
result2=logit_model2.fit()
print(result2.bic)

# Drop insulin (indiviadually insignificant) and assess the model
X3=X.drop(['triceps','diastolic','insulin'], axis=1)
logit_model3=sm.Logit(y,X3)
result3=logit_model3.fit()
print(result3.bic)

# Drop pregnant (indiviadually insignificant) and assess the model
X4=X.drop(['triceps','diastolic','insulin', 'pregnant'], axis=1)
logit_model4=sm.Logit(y,X4)
result4=logit_model4.fit()
print(result4.summary2())


# Reset design matrix
X=X4.drop(['intercept'], axis=1)
# Fitting value
logreg = LogisticRegression()
logreg.fit(X, y)

y_pred = logreg.predict(X)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X, y)))

# classification report
print(classification_report(y, y_pred))

# Confusion matrix
confusion_matrix = confusion_matrix(y, y_pred)
print(confusion_matrix)

# ROC curve
logit_roc_auc = roc_auc_score(y, logreg.predict(X))
fpr, tpr, thresholds = roc_curve(y, logreg.predict_proba(X)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# Hoslem-Lemeshow Test
y_prob = logreg.predict_proba(X)
y_test = pd.DataFrame(data['test'])
y_test.reset_index(drop=True, inplace=True)
y_prob = pd.DataFrame(y_prob)
y_prob1 = pd.concat([y_prob, y_test], axis =1)
y_prob1['decile'] = pd.qcut(y_prob1[1], 10)

obsevents_pos = y_prob1['test'].groupby(y_prob1.decile).sum()
obsevents_neg = y_prob1[1].groupby(y_prob1.decile).count() - obsevents_pos
expevents_pos = y_prob1[1].groupby(y_prob1.decile).sum()
expevents_neg = y_prob1[1].groupby(y_prob1.decile).count() - expevents_pos

hl = ((obsevents_neg - expevents_neg)**2/expevents_neg).sum()
print('chi-square: {:.2f}'.format(hl))
## df = group-2
pvalue=1-chi2.cdf(hl,8)
print('p-value: {:.2f}'.format(pvalue))

obsevents_pos = pd.DataFrame(obsevents_pos)
obsevents_neg = pd.DataFrame(obsevents_neg)
expevents_pos = pd.DataFrame(expevents_pos)
expevents_neg = pd.DataFrame(expevents_neg)
final = pd.concat([obsevents_pos, obsevents_neg, expevents_pos, expevents_neg], axis =1)
final.columns=['obs_pos','obs_neg','exp_pos', 'exp_neg']
print(final)

