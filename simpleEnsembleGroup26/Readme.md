# Unified Modeling Toolkit for Machine Learning

This R package provides a comprehensive solution for machine learning model development, offering a range of techniques and options to enhance model performance and robustness. The package supports various models, including linear regression, logistic regression, lasso, ridge, elastic net, and random forest.

## Key Features

### Single Model Building

The package allows you to build a single machine learning model of your choice, including:

- Linear Regression
- Logistic Regression
- Lasso Regression
- Ridge Regression
- Elastic Net Regression
- Random Forest

This approach is suitable when you have a specific model in mind. We have complimented this approach with cross validation and parameter tuninig which improves the model accuracy ultimately.
The function returns the trained model object, predictions on the training data, and relevant performance metrics like accuracy, mean squared error, or R-squred error.

**Constraints and Considerations:**
    When using the single model building approach, only one model should be provided. The function will raise an error if multiple models are specified without enabling bagging or ensembling

### Bagging with Soft Averaging Technique

Bagging (short for Bootstrap Aggregating) is a technique that helps improve the performance of a single machine learning model. Instead of training the model once on the entire dataset, we randomly sample the training data with replacement.

In order to average the predictions, we implement bagging with a soft averaging approach, where the final predictions are calculated as the average of the predicted probabilities from individual models.

There are several advantages of using bagging with soft averaging:
- When we average predictions from various models (of the same type) trained on different data subsets, our final prediction is less influenced by any individual unusual or noisy data point.
- Averaging insights from multiple models frequently results in higher overall accuracy compared to relying on just one model trained on the entire dataset.
- As each bagged model is trained on a distinct subset of the data, the likelihood of overfitting to the training data diminishes. This results in improved performance when dealing with fresh, unseen data.

This approach is a way to improve the performance of a single model by training multiple versions of that model on different subsets of the data and then combining their predictions through soft averaging. This approach makes the final prediction more robust and accurate.

### Ensemble Technique with Multiple Options

Ensemble learning combines the predictions of multiple models (distinct type) to achieve better performance than any single model. We are offering two ensemble techniques in this package (stacking method is default).

- Stacking: With stacking, we first train several base models (like logistic regression, random forest, etc.) on the data. Then, we train another model called a meta-model on the predictions made by the base models. This meta-model learns how to best combine the predictions from the base models. The advantage of stacking is that it can take the strengths of different models and combine them in an intelligent way, leading to more accurate and reliable predictions.
- Majority Voting (for Classification) or Soft Averaging (for Regression): For classification problems, where the goal is to predict a category (like yes or no), the package uses majority voting. It combines the predictions from multiple models, and the final prediction is the category that most models voted for. For regression problems, where the goal is to predict a numerical value, the package uses soft averaging. It calculates the average of the predictions from multiple models and uses that as the final prediction. These two methods are simple are simple but effective and widely used approaches.

### Cross-Validation and Parameter Tuning

The package includes options for cross-validation and parameter tuning, which can be applied to any of the supported models. This technique can help us assess the model's performance and generalization capability, while parameter tuning optimizes the model's hyperparameters to achieve better results.


## Comprehensive Data Preprocessing

The package includes a robust data preprocessing pipeline that handles missing values, converts and cleans columns, removes outliers, encodes categorical variables, and scales continuous data. This preprocessing step must to make sure that the data is in a suitable format for any machine learning algorithms that we are supporting.

## Top K Feature Selection and Handling High-Dimensional Data
In scenarios where the number of predictor variables (p) is much larger than the number of observations (n), the package offers a mechanism to select the top k most informative predictors. This feature selection approach can improve model performance and interpretability by only focussing on the important features.

The *select_top_k_predictors* function automates the process of selecting the top k predictors using various feature selection methods:
- Logistic Regression: For binary classification problems, the function uses the coefficients from logistic regression to rank the predictors based on their importance.
- Correlation: For regression problems, the function ranks the predictors based on their correlation with the target variable.
- Linear Regression: For regression problems, the function uses the coefficients from linear regression to rank the predictors based on their importance.
- Random Forest: The function can also use the importance scores from a random forest model to identify the most important predictors.

The function automatically detects the problem type (classification or regression) and applies the appropriate feature selection method if the method is not chose by the user. based on the problem type, it defaults to the logistic regression method for classification and correlation for regression.

Before performing feature selection, the function scales the continuous predictor variables. This step ensures that all predictors are on a comparable scale, leading to more reliable and interpretable results.

If the user does not provide a value for k (the number of top predictors to select), the function finds an optimal value based on the dataset size.

## Use Case Example:
```R
library(simpleEnsembleGroup26)
n = 100
data <- data.frame(
  feature_num1 = rnorm(n, mean = 5, sd = 2),  # Numerical feature
  feature_num2 = rnorm(n, mean = 55, sd = 42),  # Numerical feature
  feature_num3 = rnorm(n, mean = 544, sd = 52),  # Numerical feature
  feature_num4 = rnorm(n, mean = 59, sd = 16),  # Numerical feature
  feature_cat = sample(c("Category1", "Category2", "Category3"), n, replace = TRUE),  # Categorical feature
  target = sample(0:1, n, replace = TRUE)  # Binary target variable
)
X <- data[, -which(names(data) == "target")]
y <- data[["target"]]
models = list("logistic_regression","lasso","ridge","elastic_net","random_forest")
model_tuning_params <- list()
results <- unifiedModelingToolkit(X = X, y = y, models = models, model_tuning_params = model_tuning_params,
                                  bagging = FALSE, bagging_R = 100, ensemble = TRUE, feature_selection_method = NULL,
                                  k = NULL, drop_missing_records = FALSE, fill_missing_method = "mean",
                                  scale_data = TRUE, remove_outliers = FALSE, seed = 123, parameter_tuning = TRUE, cv_folds = 10, ensemble_combine_method = NULL)
print(results$predictions)
print(results$metrics)
```