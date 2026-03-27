📌 Project: Decision Tree & Linear Regression Tree from Scratch
Overview

This project implements Decision Tree and its extension — Linear Regression Tree — from scratch. The goal is to study how tree-based models behave, how splits are selected, and how model complexity affects generalization.

🔧 Implemented features
Decision Tree (classification)
Gini-based split selection
Support for real and categorical features
Recursive tree construction
Hyperparameters: max_depth, min_samples_split, min_samples_leaf
Linear Regression Tree (regression)
Linear models in leaf nodes
Split selection based on minimizing weighted MSE of child models
Quantile-based threshold selection
Piecewise-linear approximation of target function
📊 Experiments
Visualization of decision boundaries on synthetic datasets
Analysis of overfitting and model complexity
Feature importance analysis via Gini curves
Evaluation on real datasets (Mushroom, Tic-Tac-Toe)
Regression experiments on nonlinear data (Friedman dataset)
📈 Model comparison

Compared:

Linear Regression
Decision Tree Regressor
Linear Regression Tree (custom)

Results show that Linear Regression Tree:

captures nonlinear patterns better than Linear Regression
produces smoother predictions than standard Decision Tree
achieves better generalization on structured data
🧠 Key takeaway

Tree-based models can be significantly improved by replacing constant predictions in leaves with local models. This allows combining interpretability of trees with flexibility of parametric models.
