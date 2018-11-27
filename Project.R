library(faraway)
data(pima)
summary(pima)
# the variables Diastolic and bmi exhibit zero values, however, that is not physically possible
# measurement error or missing values might have been coded as 0. In an attempt to clean up the data,
#remove these values.
library(dplyr)
pima_clean = filter(pima, diastolic > 0 & bmi > 0) %>% filter(insulin >0 & glucose >0)
summary(pima_clean)

#logistic Model with test as the response variable and 8 predictor variables
#Choose the best model using the technique of Backward Elimination, focussing on the 
#individual p-values and model BIC

Fit = glm(test~ ., family = binomial(link="logit"), data = pima_clean)
summary(Fit)
BIC(Fit)

#drop diastolic (individually highly statistically insignificant) and assess the model
Fit1 = glm(test ~ pregnant+triceps+glucose+insulin+bmi+diabetes+age, family = binomial(link="logit"), data= pima_clean)
summary(Fit1)
BIC(Fit1)

#drop insulin (individually insignificant) and assess the model
Fit2 = glm(test ~ pregnant+triceps+glucose+bmi+diabetes+age, family = binomial(link="logit"), data= pima_clean)
summary(Fit2)
BIC(Fit2)

#triceps individually highly insignificant so drop the variable and assess the model
Fit3 = glm(test ~ pregnant+glucose+bmi+diabetes+age, family = binomial(link="logit"), data= pima_clean)
summary(Fit3)
BIC(Fit3)

#drop variable pregnant (individually insignificant) and assess the model
Fit4 = glm(test ~ glucose+bmi+diabetes+age, family = binomial(link="logit"), data= pima_clean)
summary(Fit4)
BIC(Fit4)

#Retain the predictor variables in Fit 4: glucose, bmi, diabetes, and age


#computing confidence intervals:
confint(Fit4)

#Hosmer-Lemeshow Test
install.packages("ResourceSelection")
library(ResourceSelection)
h1 = hoslem.test(Fit4$y, fitted(Fit4), g = 10)
h1
#inspect the expected and observed values
h2 = cbind(h1$expected, h1$observed)
h2

#Confusion Matrix

p = predict(Fit4, pima_clean, type = "response")
Con_table = table(p > 0.5, pima_clean$test)
Con_table

#ROC
install.packages("ROCR")
library(ROCR)
p = predict(Fit4, pima_clean, type = "response")
pred = prediction(p, pima_clean$test)
roc = performance(pred, "tpr", "fpr")
plot(roc,
     main = "ROC Curve")
abline(a=0, b=1)
#Higher area under the curve the better the fit (AUC)
auc = performance(pred, "auc")
auc = unlist(slot(auc, "y.values"))
auc = round(auc, 2)
legend(.6, .2, auc, title = "Area under the Curve", cex = .75)


