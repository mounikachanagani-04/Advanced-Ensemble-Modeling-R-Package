# parameter_tuning.R

# Required libraries
library(glmnet)
library(randomForest)


#' Tune glmnet Models - lasso, ridge or elastic_net
#'
#' Tunes the lambda parameter for glmnet models using cross-validation.
#'
#' @param X_train Training features.
#' @param y_train Training labels.
#' @param alpha Elastic net mixing parameter.
#' @param user_params Parameters list, including lambda.
#' @param cv_folds Number of cross-validation folds.
#' @return List with best lambda and alpha.
#'
#' @importFrom glmnet cv.glmnet
#'
#' @examples
#' # Example usage:
#' n = 100
#' X_train <- data.frame(
#'  feature_num1 = rnorm(n, mean = 5, sd = 2),  # Numerical feature
#'  feature_num2 = rnorm(n, mean = 55, sd = 42),  # Numerical feature
#'  feature_num3 = rnorm(n, mean = 544, sd = 52),  # Numerical feature
#'  feature_num4 = rnorm(n, mean = 59, sd = 16)  # Numerical feature
#'  )
#' y_train = sample(0:1, n, replace=TRUE)
#' best_params <- handle_glmnet_tuning(X_train, y_train, 1, list(lambda = c(0.01, 0.1)), 10)
#' 
#' @export
handle_glmnet_tuning <- function(X_train, y_train, alpha, user_params, cv_folds) {
  lambda_values <- if(!is.null(user_params$lambda)) user_params$lambda else NULL
  type.measure <- if(is_binary_response(y_train)) "class" else "mse"
  family <- if(is_binary_response(y_train)) "binomial" else "gaussian"
  cv_fit <- cv.glmnet(as.matrix(X_train), y_train, type.measure = type.measure,nfolds = cv_folds, alpha = alpha, lambda = lambda_values, family = family)

  best_lambda <- cv_fit$lambda.min
  return(list(lambda = best_lambda, alpha = alpha))
}


#' Tune Random Forest Model
#'
#' Finds the best `mtry` parameter for a Random Forest model using internal tuning.
#'
#' @param X_train Training features.
#' @param y_train Training labels.
#' @param user_params Custom parameters for tuning.
#' @return List with best `mtry` and other tuning parameters.
#'
#' @importFrom randomForest tuneRF
#'
#' @examples
#' # Example usage:
#' n <- 1000
#' p <- 10
#' X <- matrix(rnorm(n * p), n, p)
#' y <- sample(0:1, n, replace = TRUE)
#' rf_params <- handle_rf_tuning(X, y)
#' print(rf_params)
#' 
#' @export
handle_rf_tuning <- function(X_train, y_train, user_params = list()) {
  user_params$ntree = if (is.null(user_params$ntree)) 500 else user_params$ntree
  user_params$stepFactor = if (is.null(user_params$stepFactor)) 1.5 else user_params$stepFactor
  user_params$improve = if (is.null(user_params$improve)) 0.05 else user_params$improve

  X_train_df <- as.data.frame(X_train)
  tune_result <- tuneRF(X_train_df, y_train, ntree=user_params$ntree,stepFactor = user_params$stepFactor, improve = user_params$improve)
  best_mtry <- tune_result[which.min(tune_result[, "OOBError"]), "mtry"]
  return(list(
    mtry = best_mtry,
    ntree = user_params$ntree,
    stepFactor = user_params$stepFactor,
    improve = user_params$improve
  ))
}



#' Hyperparameter Tuning Wrapper function
#'
#' Decides and applies the appropriate tuning function based on the model type.
#'
#' @param X_train Training features.
#' @param y_train Training labels.
#' @param model_type Type of model ('lasso', 'ridge', 'elastic_net', 'random_forest').
#' @param user_params Custom parameters for tuning.
#' @param cv_folds Number of cross-validation folds (for glmnet models).
#' @return Best parameters for the model.
#'
#' @importFrom randomForest tuneRF
#' 
#' @example 
#' n <- 1000
#' p <- 10
#' X <- data.frame(matrix(rnorm(n * p), n, p))
#' y <- rnorm(sample(0:1, n, replace = TRUE))
#' model_type <- "linear_regression"
#' params = list(linear_regression = list())
#' tune_hyperparameters(x, y, model_type, user_params = params)
#' @example 
#' model_type <- "elastic_net"
#' params = list(elastic_net = list(lambda=list(0.01,0.02,0.06,0.1)))
#' tune_hyperparameters(X, y, model_type, user_params = params)
#'
#' @export
tune_hyperparameters <- function(X_train, y_train, model_type, user_params = list(), cv_folds = 5) {
  best_params = list()
  # Handle models without hyper-parameters
  if (model_type %in% c("logistic_regression", "linear_regression")) {
    log_message(sprintf("Info: No hyper-parameters to tune for %s using glm. Consider 
    feature selection or model specification as alternatives.", model_type))
  } else if (model_type %in% c("lasso", "ridge", "elastic_net")) {
    y_train = as.numeric(as.character(y_train))
    alpha <- ifelse(model_type == "lasso", 1, ifelse(model_type == "elastic_net", 0.5, 0))
    best_params = handle_glmnet_tuning(X_train, y_train, alpha, user_params, cv_folds)
    log_message(sprintf('Info: Best parameters from Parameter tuning are: %s', 
      paste(names(best_params), "=", unlist(best_params), collapse=", ", sep=" ")))
  } else if (model_type == "random_forest") {
    # Perform tuning with handle_rf_tuning
    best_params = handle_rf_tuning(X_train, y_train, user_params)
    log_message(sprintf('Info: Best parameters from Parameter tuning are: %s', 
    paste(names(best_params), "=", unlist(best_params), collapse=", ", sep=" ")))
  }
  return(best_params)
}

