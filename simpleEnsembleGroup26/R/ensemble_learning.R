#ensemble_learning.R

#' Soft Averaging Ensemble
#'
#' Combines predictions from multiple models by calculating 
#' the average of  predicted probabilties.
#'
#' @param predictions_list A list of prediction vectors from different models.
#' @return A vector of averaged predictions.
#'
#' @examples
#' predictions1 <- runif(10, 0, 1)
#' predictions2 <- runif(10, 0, 1)
#' avg_predictions <- soft_averaging(list(predictions1, predictions2))
#' @export 
soft_averaging <- function(predictions_list) {
  predictions_matrix <- do.call(cbind, predictions_list)
  average_predictions <- rowMeans(predictions_matrix, na.rm = TRUE)
  return(average_predictions)
}


#' Majority Voting for classification predictions
#'
#' Combines binary class predictions from multiple models using majority voting.
#'
#' @param predictions_list A list of binary prediction vectors from different models.
#' @param classification_threshold Threshold for converting probabilities to class labels.
#' @return A vector of predictions based on majority voting.
#'
#' @examples
#' predictions1 <- rbinom(10, 1, 0.5)
#' predictions2 <- rbinom(10, 1, 0.5)
#' voted_predictions <- majority_voting(list(predictions1, predictions2), 0.5)
#' @export 
majority_voting <- function(predictions_list, classification_threshold) {
  predictions_matrix <- do.call(cbind, predictions_list)
  class_predictions <- ifelse(predictions_matrix > classification_threshold, 1, 0)
  majority_votes <- apply(class_predictions, 1, function(row) {
    round(mean(row))
  })
  return(majority_votes)
}


#' Stacking Ensemble
#'
#' Combines predictions using stacking, where a meta-model is trained on the predictions of base models.
#'
#' @param predictions_list A list of prediction vectors from base models.
#' @param y The true response values.
#' @param is_binary_classification Boolean indicating if the task is binary classification.
#' @param classification_threshold Threshold for binary classification.
#' @return Predictions from the stacking model.
#'
#' @examples
#' predictions1 <- runif(10, 0, 1)
#' predictions2 <- runif(10, 0, 1)
#' y <- rbinom(10, 1, 0.5)  # For binary classification
#' # y <- runif(10, 0, 1)  # For regression
#' stacked_predictions <- do_stacking(list(predictions1, predictions2), y, TRUE, 0.5)
#' @export 
do_stacking <- function(predictions_list, y, is_binary_classification, classification_threshold) {

  new_X <- do.call(cbind, predictions_list)
  if (is_binary_classification) {
    # logistic regression as meta-learner for binary classification
    meta_model <- glmnet(new_X, y, family = "binomial", alpha = 1, lambda = 0.001)
  } else {
    # linear regression as meta-learner for regression
    meta_model <- glmnet(new_X, y, alpha = 1, lambda = 0.001)
  }
  # predictions from meta-learner
  meta_predictions <- predict(meta_model, new_X, type = "response")[,'s0']

  if (is_binary_classification) {
    final_predictions <- ifelse(meta_predictions > classification_threshold, 1, 0)
  } else {
    final_predictions <- meta_predictions
  }
  return(final_predictions)
}


#' Perform Ensembling with combinations of models.
#'
#' Applies ensemble learning methods to combine predictions from multiple models.
#'
#' @param X Predictor variables.
#' @param y Response variable.
#' @param models List of model types to ensemble. Supported 
#'  models = list("logistic_regression","linear_regression","lasso","ridge","elastic_net","random_forest")
#' @param params List of parameters for each model.
#' @param ensemble_combine_method Method for combining model predictions ('stacking', or 
#'  NULL for default methods: soft_averaging - regression, majority_voting - classification).
#' @param classification_threshold Threshold for binary classification.
#' @param parameter_tuning Boolean indicating whether to perform parameter tuning.
#' @return A list containing combined predictions and performance metrics.
#'
#' @examples
#' n <- 1000
#' p <- 10
#' X <- data.frame(matrix(rnorm(n * p), n, p))
#' y <- factor(sample(0:1, n, replace = TRUE))
#' models = list("random_forest", "logistic_regression")
#' params <- list("random_forest" = list())
#' result = perform_ensembling(X, y, models, params=params,
#'  classification_threshold=0.5, parameter_tuning = TRUE)
#' print(result)
#' @export 
perform_ensembling <- function(X, y, models, params, ensemble_combine_method="stacking", classification_threshold=0.5, parameter_tuning) {
  is_binary_classification <- is_binary_response(y)
  model_predictions <- list()
  for (model_type in models) {
    model_params <- if (model_type %in% names(params)) params[[model_type]] else list()
    best_params = list()
    if (parameter_tuning == TRUE) {
      best_params <- tune_hyperparameters(X, y, model_type, model_params, cv_folds = 5)
    } else {
      best_params <- model_params
    }
    model <- fit_model(X, y, model_type, best_params)
    if (model_type == "random_forest") {
      predictions <- predict(model, newdata = as.data.frame(X), type = if (is_binary_classification) "prob" else "response")
      predictions = if (is_binary_classification) predictions[, "1"] else predictions
    } else {
      X_new = if (!(model_type %in% c('linear_regression','logistic_regression'))) as.matrix(X)
      predictions <- predict(model, X_new, type = "response")
      if (model_type %in% c('lasso','ridge','elastic_net')) {
        predictions = predictions[,'s0']
      }
    }
    model_predictions[[model_type]] <- predictions
  }
  combined_predictions = list()
  metrics = list()
  # Use stacking as the combining method
  if (!is.null(ensemble_combine_method) && ("stacking" == ensemble_combine_method)) {
    combined_predictions <- do_stacking(model_predictions, y, is_binary_classification, classification_threshold)
    if (is_binary_classification == TRUE) {
      metrics <- calculate_binary_classification_metrics(y, combined_predictions)
    } else {
      metrics <- calculate_regression_metrics(y, combined_predictions)
    }
  } else if (is_binary_classification == TRUE) {
    # Majority voting for binary classification
    combined_predictions <- majority_voting(model_predictions, classification_threshold)
    log_message("Ensembling: Combined Ensembled predictions with 'Majority Voting' method.")
    metrics <- calculate_binary_classification_metrics(y, combined_predictions)
  } else {
    # Soft averaging for regression
    combined_predictions <- soft_averaging(model_predictions)
    log_message("Ensembling: Combined Ensembled predictions with 'Soft averaging' method.")
    metrics <- calculate_regression_metrics(y, combined_predictions)
  }
  return(list(predictions=combined_predictions, metrics=metrics))
}

