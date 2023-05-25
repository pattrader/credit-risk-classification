# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.

The purpose of the analysis is to build a predictive model using logistic regression to assess the risk associated with loans. The goal is to predict whether a loan is classified as a "healthy loan" (0) or a "high-risk loan" (1) based on the provided financial information.

* Explain what financial information the data was on, and what you needed to predict.

The dataset contains financial information related to loans. The variable of interest, which needs to be predicted, is the "loan_status" column. This column contains two labels: 0 represents a healthy loan, and 1 represents a high-risk loan.

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).

To understand the balance of the labels, the value_counts() function is used to count the occurrences of each label in the "loan_status" column. This provides an overview of the distribution of healthy loans and high-risk loans in the dataset.

* Describe the stages of the machine learning process you went through as part of this analysis.

1- Data loading and exploration
2- Model Training and Prediction
3- Model Evaluation 
4- Model Training and Prediction for resample data
5- Model Evaluation for resample data


* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

Logistic Regression: Logistic regression is a binary classification algorithm used to model the relationship between a set of features and a binary target variable. In this analysis, the logistic regression model is implemented using LogisticRegression() from the sklearn.linear_model module. Random Oversampling: Random oversampling is a resampling technique used to address class imbalance in a dataset. It duplicates instances from the minority class to balance the distribution. The RandomOverSampler() from the imbalanced-learn library is used to resample the training data and create a balanced dataset for model training.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.

Balanced Accuracy Score: The balanced accuracy score for Model 1 is 0.9442676901753825.
Precision Score: The precision score for Model 1 is 0.87.
Recall Score: The recall score for Model 1 is 0.89.


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
Balanced Accuracy Score: The balanced accuracy score for Model 2 is 0.994180571103648

Precision Score: The precision score for Model 2 is 0.99.
Recall Score: The recall score for Model 2 is 0.99.
## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?

Comparing the performance of the two models, it is evident that Model 2 outperforms Model 1 in terms of both balanced accuracy and precision/recall scores. Model 2 has a higher balanced accuracy score of 0.994180571103648, indicating better overall predictive performance. Additionally, Model 2 has high precision and recall scores of 0.99 for both label categories, suggesting that it is accurate in predicting both healthy loans and high-risk loans.

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

When considering the problem being solved, the importance of predicting the "0" labels (healthy loans) versus the "1" labels (high-risk loans) depends on the specific objectives and context. If the goal is to minimize the number of false positives (misclassifying healthy loans as high-risk), precision becomes crucial. On the other hand, if the focus is on identifying the majority of high-risk loans, recall becomes more important to minimize false negatives (misclassifying high-risk loans as healthy).

In this case, since both models have high precision and recall scores for both label categories, the choice of the best-performing model would depend on the specific requirements and priorities of the problem. If a higher balanced accuracy is desired, Model 2 would be recommended. However, if the emphasis is on precision and recall for both label categories, both models show strong performance.

It is essential to consider the trade-offs between precision and recall based on the problem's specific goals and consequences associated with misclassifications. Further evaluation, including considering other performance metrics and potential business considerations, would be beneficial in making a more informed decision on which model to use.


If you do not recommend any of the models, please justify your reasoning.
