# bagging.R


#' Performs Bagging operation
#'
#' Applies a bagging ensemble method to either regression or binary classification tasks
#' using specified model types and model parameters.
#'
#' @param X The predictor variables as a matrix or dataframe.
#' @param y The response variable vector.
#' @param model_type The type of model to use ('random_forest', 'linear_regression', 'logistic_regression', 'lasso', 'ridge', 'elastic_net').
#' @param params Parameters specific to the chosen model type.
#' @param B The number of bootstrap samples to generate.
#' @param classification_threshold The threshold for classifying observations as 1 in binary classification tasks.
#' @return A list containing three elements:
#'         - `predictions`: the predicted values or classes,
#'         - `variable_importance`: the importance scores of the predictor variables,
#'         - `metrics`: performance metrics of the model (accuracy, sensitivity, specificity, precision for classification; MSE, RMSE, MAE, R-squared for regression).
#'
#' @importFrom stats coef
#' @examples
#' n = 100
#' data <- data.frame(
#'     feature_num1 = rnorm(n, mean = 5, sd = 2),  # Numerical feature
#'     feature_num2 = rnorm(n, mean = 55, sd = 42),  # Numerical feature
#'     feature_num3 = rnorm(n, mean = 544, sd = 52),  # Numerical feature
#'     feature_num4 = rnorm(n, mean = 59, sd = 16),  # Numerical feature
#'     # target = sample(0:1, n, replace = TRUE) # For classification
#'     target = rnorm(sample(0:1, n, replace = TRUE)) # For regression
#'   )
#' X <- data[, -which(names(data) == "target")]
#' y <- data[["target"]]
#' model_type <- "lasso"
#' params <- list(ntree = 100)
#' classification_threshold <- 0.5  # Relevant for binary classification
#' results <- perform_bagging(X, y, model_type, params, B = 100, classification_threshold)
#' print(results)
#'
#' @export
perform_bagging <- function(X, y, model_type, params, B = 100, classification_threshold) {
  if (!is.matrix(X)) X <- as.matrix(X)
  is_binary_classification = is_binary_response(y)
  n <- nrow(X)
  p <- ncol(X)
  predictions_matrix <- matrix(nrow = n, ncol = B)
  variable_importance_scores <- rep(0, p)
  predictions = list()
  metrics = list()

  if (model_type == "random_forest") {
    model <- fit_model(X, y, model_type, params)
    predictions <- predict(model, newdata = as.data.frame(X), type = if (is_binary_classification) "prob" else "response")
    predictions = if (is_binary_classification) predictions[, "1"] else predictions
    variable_importance_scores <- randomForest::importance(model)
  } else {
    for (b in 1:B) {
      bootstrap_indices <- sample(1:n, size = n, replace = TRUE)
      X_bootstrap <- X[bootstrap_indices, , drop = FALSE]
      y_bootstrap <- y[bootstrap_indices]

      model <- fit_model(X_bootstrap, y_bootstrap, model_type, params)

      X_bootstrap = if (model_type %in% c('linear_regression','logistic_regression')) as.data.frame(X_bootstrap) else as.matrix(X_bootstrap)
      predictions <- predict(model, X_bootstrap, type = "response")
      predictions_matrix[, b] <- predictions

      if (model_type %in% c("lasso", "ridge", "elastic_net")) {
        coef_info <- coef(model, s = "lambda.min")[-1]  # Excluding intercept
      } else {
        coef_info <- coef(model)[-1]  # Excluding intercept
      }
      selected_vars <- which(coef_info != 0)
      variable_importance_scores[selected_vars] <- variable_importance_scores[selected_vars] + 1
    }
    predictions <- rowMeans(predictions_matrix)
    variable_importance_scores <- variable_importance_scores / B
  }
  if (is_binary_classification == TRUE){
    predictions <- ifelse(predictions >= classification_threshold, 1, 0)
    metrics = calculate_binary_classification_metrics(y, predictions)
  } else {
    metrics = calculate_regression_metrics(y, predictions)
  }
  return(list(predictions = predictions, variable_importance = variable_importance_scores, metrics = metrics))
}

