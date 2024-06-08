# Regression: Predicting continuous outcomes
# Simple linear regression: Models the relationship between a dependent variable and one or more independent variables
#   by fitting a linear equation to the observed data
#   Minimize the residual sum of squares or mean square error to find the best-fitting line
# Logistic Regression: Models the probability of a binary outcome based on one or more independent variables
#   Maximize the likelihood function or minimize the cross-entropy loss to find the optimal parameters
# Decision Trees: Non-parametric supervised learning algorithms used for both regression and classification tasks
#   Represented as a tree structure where each internal node represents a decision based on a feature,
#       each branch represents the outcome of the decision, and each leaf node represents the final prediction.
#   Split the feature space into regions that minimize impurity or maximize information gain
# K-Nearest Neighbors: Non-parametric classification algorithm that makes predictions based on the majority class
#   of its k nearest neighbors in the feature space.
#   Calculate the distance between the query point and all other points in the dataset,
#   select the k nearest neighbors, and assign the majority class label to the query point
# Dimensionality Reduction: aim to reduce the number of features in a dataset while preserving most of the relevant information
#   Alleviates the curse of dimensionality, improves computational efficiency, and facilitates visualization of high-dimensional data
# Regularization: add a penalty term to the loss function during model training to prevent over-fitting and promote simpler models
#   Helps to control model complexity, improve generalization, and handle multicollinearity in regression and classification models

# y = mx + b
# m= Numerator - (Number of data points * the sum of the product of each x and y) - (the sum of all x values * the sum of all y values)
#    Denominator - (the number of data points * the sum of each x value squared) - ((the sum of each x value) ^ 2)
# b = Numerator - the sum of all y values - m( the sum of all x values)
#     Denominator - the number of data points
