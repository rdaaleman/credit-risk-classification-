# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
  The purpose of this analysis is to predict if a potential loan would be a "healthy loan" which means would be paid back in full and be a good investment based on the factors such as income, number of accounts, debt, number of accounts, etc. The analysis also predicts if a loan would be "high risk" which means potentially not a good investment and potentially lose money when given an investment. 
* Explain what financial information the data was on, and what you needed to predict.
  The features of the data were:
  -amount of money requested to loan
  -interest rate
  -the person who wanted the loan (borrower)'s income
  -borrower's number of accounts
  -total debt of the borrower
  -derogatory marks or information on the borrower's credit
  -debt to income which is the financial metric comparing the borrow's monthly debt payments to their gross monthly income 
  We needed to predict the features lead to the loan being a healthy loan or a high risk. To predict if future loans with the features could be a healthy or high risk loan. 
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
The value_counts were the number of datasets we had to train the data. 
* Describe the stages of the machine learning process you went through as part of this analysis.
-I had to make a copy of the data and identify a X and y.  Then split it into a training and testing data set
-Then I had to choose a model selection with Model 1 I choose Logistic Regression and then made a prediction to train the model 
-then produced an accuracy and confusion matrix to produce the classification model to show how accurate the model would be 


* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
-


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
Classifcation Report
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384

For Class 0 because the Precision is 1.00 that means all predicted loans are healthy. The recall being 0.99 means that the model correctly predicts 99% of the predicted healthy loans. Overall this indictates that for predicting a healthy loan is highly likely using this model. For Class 1 the precision is 0.85 which means that the model is able to predict 85% of the predicted high-risk loans that are actually high-risk. The recall being 0.91 means that the model is able to predict 91% of the actual high-risk loans. 


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  Classifcation Report
              precision    recall  f1-score   support

           0       0.99      0.99      0.99     56271
           1       0.99      0.99      0.99     56271

    accuracy                           0.99    112542
   macro avg       0.99      0.99      0.99    112542
weighted avg       0.99      0.99      0.99    112542

For Class 0 because the Precision is .99 that means 99% predicted loans are healthy. The recall being 0.99 means that the model correctly predicts 99% of the predicted healthy loans. Overall this indictates that for predicting a healthy loan is highly likely using this model. For Class 1 the precision is 0.99 which means that the model is able to predict 99% of the predicted high-risk loans that are actually high-risk. The recall being 0.99 means that the model is able to predict 99% of the actual high-risk loans.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
  Machine Learning Model 2 seems to perform the best because it is able to predict both healthy and high risk loans more accurately compared to to the Machine Learning Module 1. 
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
