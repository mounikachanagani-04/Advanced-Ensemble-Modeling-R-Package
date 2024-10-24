# unifiedModelingToolkit_learner.R

library(glmnet)
library(randomForest)



#' Unified Modeling Toolkit for Machine Learning
#'
#' This function provides a comprehensive approach to machine learning model development,
#' including preprocessing, model fitting, and ensemble learning. It supports various models,
#' preprocessing techniques such as feature selection, missing data imputation, scaling,
#' and outlier removal, as well as advanced model fitting strategies like bagging and ensembling.
#' Parameter tuning and model evaluation metrics are also integrated into the function.
#'
#' @param X Predictor variables as a dataframe or matrix.
#' @param y Response variable as a vector.
#' @param models A list of model names to be used for training.
#' @param bagging Logical flag indicating whether bagging should be applied (FALSE by default).
#' @param bagging_R The number of bootstrap replicates to use for bagging.
#' @param ensemble Logical flag indicating whether ensemble learning should be applied (FALSE by default).
#' @param feature_selection_method The method to be used for feature selection (if any).
#' @param k The number of top features to select.
#' @param drop_missing_records Logical flag indicating whether to drop records with missing values (TRUE by default).
#' @param fill_missing_method The method to be used for imputing missing values ('mean' by default).
#' @param scale_data Logical flag indicating whether to scale continuous variables (TRUE by default).
#' @param remove_outliers Logical flag indicating whether to remove outliers (FALSE by default).
#' @param seed An integer value to set the random seed for reproducibility (123 by default).
#' @param parameter_tuning Logical flag indicating whether to perform hyperparameter tuning (FALSE by default).
#' @param cv_folds The number of folds to use for cross-validation during parameter tuning (5 by default).
#' @param model_tuning_params A list of parameters for tuning the models (list() by default).
#' @param classification_threshold The threshold for converting probabilities to binary class labels
#' in classification tasks (0.5 by default).
#' @param ensemble_combine_method The method to combine models in ensemble learning ('stacking' by default).
#' @return A list of items - The exact contents of the list depend on the chosen modeling strategy.
#'  1. containing model(s), predictions, and metrics if both ensemble and bagging are FALSE
#'  2. predictions, variable importance scores, and metrics for bagging = TRUE && ensemble = FALSE,
#'  3. predictions, and metrics for ensemble = TRUE && bagging = FALSE
#'
#' @examples
#' n <- 1000
#' p <- 10
#' X <- data.frame(matrix(rnorm(n * p), n, p))
#' y <- sample(0:1, n, replace = TRUE)
#' train_idx <- sample(1:n, 0.7 * n)
#' X_train <- X[train_idx, ]
#' y_train <- y[train_idx]
#' X_test <- X[-train_idx, ]
#' y_test <- y[-train_idx]
#' models = list("random_forest")
#' model_tuning_params <- list("random_forest" = list(ntree=10))
#' results <- unifiedModelingToolkit(X = X_train, y = y_train, models = models,
#'  model_tuning_params = model_tuning_params, bagging = FALSE, bagging_R = 100,
#'  ensemble = FALSE, feature_selection_method = NULL, k = NULL,
#'  drop_missing_records = FALSE, fill_missing_method = "mean", scale_data = FALSE,
#'  remove_outliers = FALSE, seed = 123, parameter_tuning = FALSE, cv_folds = 10,
#'  ensemble_combine_method = NULL)
#' make_predictions(results$model, X_test)
#' @export
unifiedModelingToolkit <- function(X, y, models, bagging = FALSE, bagging_R = 100, ensemble = FALSE,
                                   feature_selection_method = NULL, k = NULL,
                                   drop_missing_records = TRUE, fill_missing_method = "mean",
                                   scale_data = FALSE, remove_outliers = FALSE, seed = 123,
                                   parameter_tuning = FALSE, cv_folds = 5, model_tuning_params = list(),
                                   classification_threshold = 0.5, ensemble_combine_method = 'stacking') {

  if (!is.list(models)){
    stop("'models' is not a list. Please pass the model name(s) as a list and retry. Ex: list('random_forest')")
  } else {
    models = tolower(models)
  }

  supported_models = c("linear_regression","logistic_regression","lasso","ridge","elastic_net","random_forest")
  if (!all(models %in% supported_models)) {
    stop(sprintf("Given model(s) is not supported. Please try: %s.", paste(supported_models, collapse = ", ")))
  }

  if (bagging && ensemble) {
    stop("Both bagging and ensembling cannot be enabled at the same time. Please choose only one.")
  }

  if (bagging && length(models) != 1) {
    stop("When using bagging, only one model should be provided.")
  }

  if (ensemble && length(models) <= 1) {
    stop("When using ensembling, multiple models should be provided.")
  }

  if (!bagging && !ensemble && length(models) != 1) {
    stop("Only one model should be provided when bagging and ensembling are not enabled.")
  }

  # Preprocess data
  processed_data <- preprocess_data(
    y = y, x = X, models,
    k = k, feature_selection_method = feature_selection_method,
    drop_missing_records = drop_missing_records, fill_missing_method = fill_missing_method,
    scale_data = scale_data, remove_outliers = remove_outliers, seed = seed
  )

  X_processed <- processed_data$x
  y_actuals <- processed_data$y

  best_params = list()
  ensemble_predictions <- list()
  if (bagging == TRUE) {
    model_type = models[1]
    user_params <- if (model_type %in% names(model_tuning_params)) model_tuning_params[[model_type]] else list()
    if (parameter_tuning == TRUE) {
      best_params <- tune_hyperparameters(X_processed, y_actuals, model_type, user_params, cv_folds)
    } else {
      best_params <- user_params
    }
    bagging_results <- perform_bagging(X_processed, y_actuals, model_type, best_params, B = bagging_R, classification_threshold)
    return(list(predictions = bagging_results$predictions, variable_importance_scores = bagging_results$variable_importance, metrics = bagging_results$metrics))
  }
  if(ensemble == TRUE) {
    ensemble_predictions <- perform_ensembling(X_processed, y_actuals, models, model_tuning_params, ensemble_combine_method, classification_threshold, parameter_tuning)
    return(list(predictions=ensemble_predictions$predictions, metrics=ensemble_predictions$metrics))
  }

  model_type = models[1]
  user_params <- if (model_type %in% names(model_tuning_params)) model_tuning_params[[model_type]] else list()
  if (parameter_tuning == TRUE) {
    best_params <- tune_hyperparameters(X_processed, y_actuals, model_type, user_params, cv_folds)
  } else {
    best_params <- user_params
  }
  # Fit model with the best parameters found (if tuning performed) or user_params
  model <- fit_model(X_processed, y_actuals, model_type, best_params)
  is_binary_classification = is_binary_response(y)
  metrics = list()
  if (model_type == "random_forest") {
    predictions <- predict(model, newdata = as.data.frame(X_processed), type = if (is_binary_classification) "prob" else "response")
    predictions = if (is_binary_classification) predictions[, "1"] else predictions
  } else {
    predictions <- predict(model, newx = as.matrix(X_processed), type = "response")
    if (model_type %in% c('lasso','ridge','elastic_net')) {
      predictions = predictions[,'s0']
    }
  }
  if (is_binary_classification == TRUE){
    predictions <- ifelse(predictions >= classification_threshold, 1, 0)
    metrics = calculate_binary_classification_metrics(y_actuals, predictions)
  } else {
    metrics = calculate_regression_metrics(y_actuals, predictions)
  }

  # Storing some information for future predictions
  model$one_hot_levels = processed_data$one_hot_levels
  model$scale_data = scale_data
  model$scale_params = processed_data$scale_params
  model$classification_threshold = classification_threshold
  model$model_type = model_type
  model$top_k_feature_names = processed_data$top_k_feature_names


  return(list(
    model = model,
    predictions = predictions,
    metrics = metrics)
  )
}
