SVM Regressor with Permutation Importance for Predicting Gym, Pool and Fitness Activities

This project utilizes the Support Vector Machine (SVM) Regressor to predict various fitness-related activities, including gym access, pool access, classes, and more, based on user data. The model also includes permutation importance to evaluate the influence of different features on the predictions.

Introduction

This repository contains Python code that applies the SVM Regressor to predict the likelihood of various fitness center activities. The project demonstrates how to preprocess data, train a machine learning model, and use permutation importance to understand the impact of each feature on the model's predictions.

Ensure to have the following libraries installed:

pandas
scikit-learn
matplotlib
seaborn

Data Processing

The script processes the data by converting date columns into numerical formats, encoding categorical variables, and calculating additional features like the frequency of entry.

Model Training and Evaluation

The SVM Regressor is used to train a model for each activity. The script splits the data into training and testing sets, fits the model, and evaluates its performance using metrics like R², MSE, and RMSE.

Feature Importance Visualization

The script uses permutation importance to calculate and visualize the influence of each feature on the model's predictions. This provides insights into which factors are most influential in determining gym or pool access, participation in classes, etc.

Evaluation Metrics

The script calculates and prints several evaluation metrics:

R² (R-squared): Indicates how well the model explains the variance in the data.
MSE (Mean Squared Error): Measures the average squared difference between predicted and actual values.
RMSE (Root Mean Squared Error): The square root of the MSE, providing error measurement in the same units as the target variable.
Prediction for Individual Members

The script includes a function to predict the likelihood of different activities for a specific gym member based on their unique key. It also provides a detailed breakdown of their predicted activities.
