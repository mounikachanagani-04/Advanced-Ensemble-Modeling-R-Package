# model_fitting.R


#' Fit Various Types of Models
#'
#' Fits a model based on input data, model type, and parameters.
#' Supports logistic regression, linear regression, lasso, ridge, elastic net, and random forest.
#'
#' @param X Input matrix of predictors.
#' @param y Response variable.
#' @param model_type Type of model to fit.
#' @param params List of additional parameters for model fitting.
#' @return Fitted model object.
#' @importFrom stats glm as.formula
#' @importFrom glmnet glmnet
#' @importFrom randomForest randomForest
#' 
#' @examples
#' n <- 20
#' X <- matrix(rnorm(n * 10), ncol = 10)
#' y <- rnorm(100, 1, 0.5)
#' fit_model(X, y, "linear_regression")
#' @examples
#' n = 10
#' X1 <- rnorm(n, mean = 10, sd = 2)
#' X2 <- runif(n, min = 0, max = 1)
#' X3 <- rbinom(n, size = 1, prob = 0.3)
#' X <- data.frame(X1, X2, X3)
#' true_coeffs <- c(2, -3, 1)
#' y <- X1 * true_coeffs[1] + X2 * true_coeffs[2] + X3 * true_coeffs[3] + rnorm(n, mean = 0, sd = 1)
#' params = list(lamda = 0.05)
#' fit_model(X, y, model_type="elastic_net", params=params)
#' 
#' @export 
fit_model <- function(X, y, model_type, params = list()) {
  # Set default family (binary or continuous)
  family = if (is_binary_response(y)) "binomial" else "gaussian"

  model <- NULL
  if (model_type == "logistic_regression"){
    if (!is_binary_response(y))
      stop("Logistic regression requires a binary response variable. Please choose different model.")
    data = data.frame(y = y, X)
    model <- glm(
      as.formula("y ~ ."),
      family = "binomial",
      data = data
    )
  } else if (model_type == "linear_regression") {
    if (is_binary_response(y))
      stop("Linear regression requires a non-binary response variable. Please choose different model.")
    data = data.frame(y = y, X)
    model <- glm(
      as.formula("y ~ ."),
      family = "gaussian",
      data = data
    )
  } else if (model_type == "lasso"){
    alpha <- if (is.null(params$alpha)) 1 else params$alpha
    lambda <- if (is.null(params$lambda)) 0.01 else params$lambda
    model <- glmnet(
      x = X,
      y = y,
      alpha = alpha,
      lambda = lambda,
      family = family
    )
  } else if (model_type == "ridge"){
    alpha <- if (is.null(params$alpha)) 0 else params$alpha
    lambda <- if (is.null(params$lambda)) 0.01 else params$lambda
    model <- glmnet(
      x = X,
      y = y,
      alpha = alpha,
      lambda = lambda,
      family = family
    )
  } else if (model_type == "elastic_net"){
    alpha <- if (is.null(params$alpha)) 0.5 else params$alpha
    lambda <- if (is.null(params$lambda)) 0.01 else params$lambda
    if (alpha <= 0 || alpha >= 1) stop("Elastic Net requires 0 < alpha < 1.")
    model <- glmnet(x = X, y = y, alpha = alpha, lambda = lambda, family = family)
  } else if (model_type == "random_forest"){
    if (!is.data.frame(X)){ # Convert X to data.frame for randomForest compatibility
      X_df <- as.data.frame(X)
    }
    ntree <- params$ntree
    stepFactor <- params$stepFactor
    improve <- params$improve
    mtry <- params$mtry
    args_list <- list()
    if (!is.null(ntree)) args_list$ntree <- ntree
    if (!is.null(mtry)) args_list$mtry <- mtry
    if (!is.null(stepFactor)) args_list$stepFactor <- stepFactor
    if (!is.null(improve)) args_list$improve <- improve
    model <- do.call(randomForest, c(list(formula = as.formula("y ~ ."),
                                          data = cbind(y = y, as.data.frame(X))),args_list))
  } else {
    stop(sprintf("Unsupported model type '%s'. Choose from 'logistic_regression', 'linear_regression', 'lasso', 'ridge', 'elastic_net', or 'random_forest'.", model_type))
  }
  model$is_binary_classification = is_binary_response(y)
  return(model)
}
