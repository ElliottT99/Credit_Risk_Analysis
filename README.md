# Credit Risk Analysis

## Overview
The purpose of this analysis is to adapt a supervised machine learning model, if one brings up promising results, to predict whether a loan is high-risk or low-risk, based on a credit card data-set lent to us by Lending Club. The challenge here is due to potential high-risk loans being few and far between, it is difficult to calculate due to bias towards low-risk loans. Rather than look at the overall data, we will be focusing in on high-risk loans and our machine learning predictions.

### Results
The data gathered takes two forms: a prediction table, as well as a imbalanced classification report.

This is the prediction table, outlining what each value means based on position:

![predictionTable](https://github.com/ElliottT99/Credit_Risk_Analysis/blob/main/Resources/predictionTable.PNG)

Here is an example, from our Random Oversampling Model:

![conMatRanOver](https://github.com/ElliottT99/Credit_Risk_Analysis/blob/main/Resources/Random%20Oversampling/conMatRanOver.PNG)

This is the imbalanced classification report, with multiple categories such as Precision(pre), Sensativity(rec), and F1(f1). We will be focusing on these three statistics to accuracy analysis of each model. (The report below is our Random Oversampling Model)

![imbClassRepRanOver](https://github.com/ElliottT99/Credit_Risk_Analysis/blob/main/Resources/Random%20Oversampling/imbClassRepRanOver.PNG)

Using 6 different machine learning models, we found these results for each model:

#### Random Oversampling

![conMatRanOver](https://github.com/ElliottT99/Credit_Risk_Analysis/blob/main/Resources/Random%20Oversampling/conMatRanOver.PNG)

![imbClassRepRanOver](https://github.com/ElliottT99/Credit_Risk_Analysis/blob/main/Resources/Random%20Oversampling/imbClassRepRanOver.PNG)

Based on this data, we find the following for high-risk loans:
* Accuracy was calculated to be roughly 83.3%
* We have over 32 times the amount of false-positive compared to true-positives
* There are 18 false-negatives
* Precision is very low, at 3%
* Sensativity is high, at 82%

#### SMOTE Oversampling

![conMatSMOTE](https://github.com/ElliottT99/Credit_Risk_Analysis/blob/main/Resources/SMOTE/conMatSMOTE.PNG)

![imbClassRepSMOTE](https://github.com/ElliottT99/Credit_Risk_Analysis/blob/main/Resources/SMOTE/imbClassRepSMOTE.PNG)

Based on this data, we find the following for high-risk loans:
* Accuracy was calculated to be roughly 83.9%
* We have almost 28 times the amount of false-positive compared to true-positives
* There are 19 false-negatives
* Precision is very low, at 3%
* Sensativity is high, at 81%

#### Cluster Centroids Undersampling

![conMatCC](https://github.com/ElliottT99/Credit_Risk_Analysis/blob/main/Resources/Cluster%20Centroids/conMatCC.PNG)

![imbClassRepCC](https://github.com/ElliottT99/Credit_Risk_Analysis/blob/main/Resources/Cluster%20Centroids/imbClassRepCC.PNG)

Based on this data, we find the following for high-risk loans:
* Accuracy was calculated to be roughly 83.9%
* We have over 46 times the amount of false-positive compared to true-positives
* There are 14 false-negatives
* Precision is very low, at 2%
* Sensativity is high, at 86%

#### SMOTEENN Combination Sampling

![conMatSMOTEENN](https://github.com/ElliottT99/Credit_Risk_Analysis/blob/main/Resources/SMOTEENN/conMatSMOTEENN.PNG)

![imbClassRepSMOTEENN](https://github.com/ElliottT99/Credit_Risk_Analysis/blob/main/Resources/SMOTEENN/imbClassRepSMOTEENN.PNG)

Based on this data, we find the following for high-risk loans:
* Accuracy was calculated to be roughly 60.6%
* We have over 127 times the amount of false-positive compared to true-positives
* There are 14 false-negatives
* Precision is very low, at 1%
* Sensativity is high, at 86%

##### Balanced Random Forest Classifier

![conMatBRFC](https://github.com/ElliottT99/Credit_Risk_Analysis/blob/main/Resources/Balanced%20Random%20Forest%20Classifier/conMatBRFC.PNG)

![imbClassRepBRFC](https://github.com/ElliottT99/Credit_Risk_Analysis/blob/main/Resources/Balanced%20Random%20Forest%20Classifier/imbClassRepBRFC.PNG)

Based on this data, we find the following for high-risk loans:
* Accuracy was calculated to be roughly 78.9%
* We have over 30 times the amount of false-positive compared to true-positives
* There are 30 false-negatives
* Precision is very low, at 3%
* Sensativity is relavitely low, at 70%

#### Easy Ensemble AdaBoost Classifier

![conMatEEAC](https://github.com/ElliottT99/Credit_Risk_Analysis/blob/main/Resources/Easy%20Ensemble%20AdaBoost%20Classifier/conMatEEAC.PNG)

![imbClassRepEEAC](https://github.com/ElliottT99/Credit_Risk_Analysis/blob/main/Resources/Easy%20Ensemble%20AdaBoost%20Classifier/imbClassRepEEAC.PNG)

Based on this data, we find the following for high-risk loans:
* Accuracy was calculated to be roughly 93.2%
* We have just over 10 times the amount of false-positive compared to true-positives
* There are 8 false-negatives
* Precision is low but higher than other models, at 9%
* Sensativity is relatively low, at 70%

### Summary
Based on our collected data, we can see that our machine learning models do not do a very good job at detecting pure-positives, our high risk loans, as most models have false positives in the thousands. That is not the worst thing however, as false positives can be looked into and corrected(not that you would want to call thousands of people to sort out discrepencies).

A primary concern with a program looking for and attempting to decide whether a loan is high risk or not is that it does not identify a high risk loan. Of our six models, our Easy Ensemble AdaBoost Classifier model has the least amount of false-negatives, 8 total false-negatives, as well as having the highest accuracy rate amongst our models, at a high 93.2%. If a model has to be recommended out of our 6 tested models, I would have to recommend the Easy Ensemble AdaBoost Classifier model as it has the best numbers in our priority areas. 8 missed incorrectable predictions out of 17,205 total is quite good, and although the ideal system would provide no false-positive/negatives, the system which produces the least amount of incorrectable errors is our next best (additionally, this model has the least amount of false-positives, so that's a plus since that is less data that needs to be validated and reproccessed)
