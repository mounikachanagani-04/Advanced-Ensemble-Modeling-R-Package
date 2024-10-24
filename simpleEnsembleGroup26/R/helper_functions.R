# helper_functions.R

enable_logs = TRUE


#' Log a Message
#'
#' Conditionally logs a message to the R console, depending on the global `enable_logs` flag.
#'
#' @param msg The message string to log.
#' @details If `enable_logs` is TRUE, the message will be prefixed with "Message: " and printed.
#' @export
log_message <- function(msg){
  if (enable_logs){
    log_msg <- sprintf("Message: %s", msg)
    message(log_msg)
  }
}


#' Check if a Vector Represents a Binary Response
#'
#' Determines whether a numeric vector represents a binary response variable (0 and 1 only).
#'
#' @param y Numeric vector to check.
#' @return TRUE if `y` contains only 0s and 1s, FALSE otherwise.
#' @examples
#' is_binary_response(c(0, 1, 0, 1)) # Returns TRUE
#' is_binary_response(c(1, 2, 3, 4)) # Returns FALSE
#' @export
is_binary_response <- function(y) {
  all(y %in% c(0, 1))
}


#' Calculate Binary Classification Metrics
#'
#' Computes accuracy, sensitivity (recall), specificity, and precision for binary classification predictions.
#'
#' @param true_labels A vector of true binary labels.
#' @param predicted_labels A vector of predicted binary labels.
#' @return A named vector containing accuracy, sensitivity, specificity, and precision.
#' @examples
#' true_labels <- c(1, 0, 1, 0)
#' predicted_labels <- c(1, 0, 0, 1)
#' calculate_binary_classification_metrics(true_labels, predicted_labels)
#' @export
calculate_binary_classification_metrics <- function(true_labels, predicted_labels) {
  if (!is.factor(true_labels)) { # Check if true_labels is a factor, if not, convert to factor with levels 0 and 1
    true_labels <- factor(true_labels, levels = c(0, 1))
  } else {
    levels(true_labels) <- c(0, 1)
  }
  if (!is.factor(predicted_labels)) { # Check if predicted_labels is a factor, if not, convert to factor with levels 0 and 1
    predicted_labels <- factor(predicted_labels, levels = c(0, 1))
  } else {
    levels(predicted_labels) <- c(0, 1)
  }

  confusion_matrix <- table(true_labels, predicted_labels) # Confusion matrix

  # True Positives, False Negatives, True Negatives, False Positives
  tp <- confusion_matrix[2, 2]
  fn <- confusion_matrix[2, 1]
  tn <- confusion_matrix[1, 1]
  fp <- confusion_matrix[1, 2]

  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix) # Accuracy
  sensitivity <- tp / (tp + fn) # Sensitivity (Recall)
  specificity <- tn / (tn + fp) # Specificity
  precision <- tp / (tp + fp) # Precision
  return(c(accuracy = accuracy,
           sensitivity = sensitivity,
           specificity = specificity,
           precision = precision))
}



#' Calculate Regression Metrics
#'
#' Computes Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared for regression predictions.
#'
#' @param true_values A numeric vector of true values.
#' @param predicted_values A numeric vector of predicted values.
#' @return A named vector containing MSE, RMSE, MAE, and R-squared.
#' @examples
#' true_values <- c(3, -0.5, 2, 7)
#' predicted_values <- c(2.5, 0.0, 2, 8)
#' calculate_regression_metrics(true_values, predicted_values)
#' @export
calculate_regression_metrics <- function(true_values, predicted_values) {
  mse <- mean((true_values - predicted_values)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(true_values - predicted_values))
  ss_tot <- sum((true_values - mean(true_values))^2)
  ss_res <- sum((true_values - predicted_values)^2)
  r_squared <- 1 - (ss_res / ss_tot)
  return(c(MSE = mse,
           RMSE = rmse,
           MAE = mae,
           "R-squared" = r_squared))
}



#' Scale Continuous Variables in a Dataframe
#'
#' This function scales the continuous (numeric) variables in a dataframe by centering
#' and scaling them individually. It only applies scaling for columns with non-zero variance
#' as columns with 0 variance are constant columns. The function returns the scaled dataframe and
#' the scaling parameters (mean and standard deviation) for each continuous column.
#'
#' @param dataframe A dataframe containing the data to be scaled.
#' @return A list with two parts: `x_data`, the changed dataframe, and `scale_params`,
#'         which will have mean and standard deviation used for each column.
#' 
#' @importFrom stats var
#'
#' @examples
#' # Load an example dataset
#' data(iris)
#'
#' # Scale the continuous variables in the iris dataset
#' scaled_data <- scale_continuous_data(iris)
#'
#' # Access the scaled dataframe
#' scaled_iris <- scaled_data$x_data
#'
#' # Access the scaling parameters
#' scaling_params <- scaled_data$scale_params
#' @export
scale_continuous_data <- function(dataframe) {
  continuous_cols <- names(dataframe)[sapply(dataframe, is.numeric)] # Identifying continuous (numeric) columns
  non_constant_cols <- continuous_cols[sapply(dataframe[continuous_cols], function(x) var(x, na.rm = TRUE) > 0)] # Remove columns with zero variance (constant columns or columns with only one unique value)
  scale_params <- list() # Initialize a list to store scaling parameters
  for (col_name in non_constant_cols) { # Loop over non-constant continuous columns
    col_data <- dataframe[[col_name]]
    scaled_col <- scale(col_data, center = TRUE, scale = TRUE) # Scale the column data (center and scale), handling missing values
    # Store scaling parameters (mean and standard deviation) for the column
    scale_params[[col_name]] <- list(
      mean = attr(scaled_col, "scaled:center"),
      sd = attr(scaled_col, "scaled:scale")
    )
    dataframe[[col_name]] <- as.numeric(scaled_col)
  }
  if (!is.data.frame(dataframe)) dataframe <- as.data.frame(dataframe)
  return(list(x_data = dataframe, scale_params = scale_params))
}

#' Apply Scaling Parameters to New Data
#'
#' This function applies previously calculated scaling parameters (mean and standard deviation)
#' to the continuous variables in a new dataframe (`new_x`). It's designed to scale new data
#' using the scaling parameters derived from a training dataset. This ensures consistency
#' in feature scaling across both the training phase and when making predictions on new data.
#'
#' @param new_x A dataframe containing the new data to be scaled. This dataframe
#'              should include the same continuous variables that were present in the
#'              original dataset from which the scaling parameters were derived.
#' @param scale_params A list containing the scaling parameters (mean and standard deviation)
#'                     for each continuous variable that was scaled in the training dataset.
#'                     Each entry in this list should have two elements:
#'                     \code{$mean} and \code{$sd}, representing the mean and standard deviation
#'                     used for scaling the corresponding variable.
#'
#' @return A dataframe (`new_x`) with the continuous variables scaled according to the provided
#'         scaling parameters. Columns that not found in \code{scale_params} are left unchanged.
#'
#' @examples
#' n = 100
#' training_data <- data.frame(
#'     feature_num1 = rnorm(n, mean = 5, sd = 2),  # Numerical feature
#'     feature_num2 = rnorm(n, mean = 55, sd = 42),  # Numerical feature
#'     feature_num3 = rnorm(n, mean = 544, sd = 52),  # Numerical feature
#'     feature_num4 = rnorm(n, mean = 59, sd = 16)  # Numerical feature
#'   )
#' scaled_training_data = scale_continuous_data(training_data)
#' n = 10
#' new_x <- data.frame(
#'     feature_num1 = rnorm(n, mean = 5, sd = 2),  # Numerical feature
#'     feature_num2 = rnorm(n, mean = 55, sd = 42),  # Numerical feature
#'     feature_num3 = rnorm(n, mean = 544, sd = 52),  # Numerical feature
#'     feature_num4 = rnorm(n, mean = 59, sd = 16)  # Numerical feature
#'   )
#' # Apply the obtained scaling parameters to the new data:
#' scaled_new_data <- apply_scaling(new_x, scaled_training_data$scale_params)
#'
#' @export
apply_scaling <- function(new_x, scale_params) {
  for (col_name in names(scale_params)) {
    if (col_name %in% names(new_x)) {
      col_mean <- scale_params[[col_name]]$mean
      col_sd <- scale_params[[col_name]]$sd
      new_x[[col_name]] <- (new_x[[col_name]] - col_mean) / col_sd # Apply the scaling transformation to the current column in new_x
    }
  }
  return(new_x)
}


#' Perform One-Hot Encoding on Factor Columns
#'
#' This function performs one-hot encoding on factor columns in a dataframe. It replaces
#' each factor column with a set of binary columns, one for each level of the factor.
#' The function can optionally return the levels of the factor columns along with the
#' encoded dataframe.
#'
#' @param dataframe A dataframe containing the data to be one-hot encoded.
#' @param return_levels A logical value indicating whether to return the levels of the
#'   factor columns along with the encoded dataframe. Default is FALSE.
#'
#' @return If `return_levels` is FALSE, it will only return the changed dataframe.
#'         If TRUE, it will return a list with the dataframe and the levels of each factor column.
#'
#' @importFrom stats model.matrix
#' 
#' @examples
#' # Load an example dataset
#' data(iris)
#'
#' # One-hot encode the 'Species' column
#' encoded_data <- perform_one_hot_encoding(iris)
#'
#' # One-hot encode and return the levels
#' encoded_data_with_levels <- perform_one_hot_encoding(iris, return_levels = TRUE)
#' encoded_iris <- encoded_data_with_levels$data
#' species_levels <- encoded_data_with_levels$levels
#'
#' @export
perform_one_hot_encoding <- function(dataframe, return_levels = FALSE) {
  factor_cols <- sapply(dataframe, is.factor)
  factor_col_names <- names(dataframe)[factor_cols]
  levels_list <- list() # Initialize a list to store levels for each factor column
  for (col_name in factor_col_names) {
    levels_list[[col_name]] <- levels(dataframe[[col_name]])
    formula <- as.formula(paste("~", col_name, "- 1"))
    encoded_data <- model.matrix(formula, data = dataframe)
    encoded_data <- encoded_data[, -1]
    new_col_names <- colnames(encoded_data)
    col_index <- match(col_name, names(dataframe))
    dataframe <- dataframe[, !(names(dataframe) %in% col_name), drop = FALSE]

    if (col_index > ncol(dataframe)) {
      dataframe <- cbind(dataframe, encoded_data)
      names(dataframe)[(ncol(dataframe) - length(new_col_names) + 1):ncol(dataframe)] <- new_col_names
    } else {
      df_before <- dataframe[, 1:(col_index - 1), drop = FALSE]
      df_after <- dataframe[, col_index:ncol(dataframe), drop = FALSE]
      dataframe <- cbind(df_before, encoded_data, df_after)
      names(dataframe)[(ncol(df_before) + 1):(ncol(df_before) + length(new_col_names))] <- new_col_names
    }
  }
  unique_column_names <- unique(names(dataframe))
  dataframe <- dataframe[, unique_column_names, drop = FALSE]

  if (return_levels) {
    return(list(data = dataframe, levels = levels_list))
  } else {
    return(dataframe)
  }
}

#' Apply One-Hot Encoding to New Data
#'
#' Encodes new data to fit the one-hot encoding scheme used on the training data. It ensures that only previously seen categories are present in the new data.
#'
#' @param new_data The dataframe containing new observations.
#' @param levels_list A list detailing the categories for each categorical column from the original dataset.
#' @return The transformed dataset with original dataset's structure.
#'
#' @examples
#' train_data <- data.frame(Color = factor(c("Red", "Green", "Blue")),
#'                          Shape = factor(c("Square", "Circle", "Triangle")),
#'                          Value = c(10, 20, 30))
#' encoded_train_data <- perform_one_hot_encoding(train_data, return_levels = TRUE)
#' levels_list <- encoded_train_data$levels
#' new_data <- data.frame(Color = factor(c("Red", "Green")), Shape = factor(c("Square", "Circle")))
#' encoded_new_data <- apply_one_hot_encode_new_data(new_data, levels_list)
#' @export
apply_one_hot_encode_new_data <- function(new_data, levels_list) {
  for (col_name in names(levels_list)) {
    if (col_name %in% names(new_data)) {
      new_values <- levels(new_data[[col_name]])
      unseen_values <- setdiff(new_values, levels_list[[col_name]])
      if (length(unseen_values) > 0) {
        stop(paste("Error: Unseen factor levels in column", col_name, ":", paste(unseen_values, collapse = ", ")))
      }
      new_data[[col_name]] <- factor(new_data[[col_name]], levels = levels_list[[col_name]])
    }
  }
  new_data <- perform_one_hot_encoding(new_data)
  return(new_data)
}

#' Generate Predictions Using a Pre-Trained Model - supports both binary classification and regression.
#'
#' Prepares new data according to the specifications of a pre-trained model (including one-hot encoding and scaling),
#' optionally selects the same top k predictors used in training phase and then uses the model to predict outcomes.
#'
#' @param model An object representing the pre-trained model, containing all necessary preprocessing parameters.
#' @param new_x The new dataset for making predictions.
#' @return A list with the predicted values or classes for the new data.
#' 
#' @importFrom stats predict
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
make_predictions = function (model, new_x) {
  if (anyNA(new_x)) {
    stop("Input data contains missing values. Please provide complete data.")
  }
  new_x = apply_one_hot_encode_new_data(new_x, levels_list=model$one_hot_levels)

  if (model$scale_data == TRUE){
    new_x = apply_scaling(new_x, model$scale_params)
  }
  top_k_feature_names = model$top_k_feature_names

  if (length(top_k_feature_names) > 0) {
    new_x = new_x[, top_k_feature_names, drop = FALSE]
  }
  model_type = model$model_type
  is_binary_classification = model$is_binary_classification
  if (model_type == "random_forest") {
    predictions <- predict(model, newdata = as.data.frame(new_x), type = ifelse(is_binary_classification, "prob", "response"))
    predictions = if (is_binary_classification) predictions[, "1"] else predictions
  } else {
    new_x = if (model_type %in% c('linear_regression','logistic_regression')) as.data.frame(new_x) else as.matrix(new_x)
    predictions <- predict(model, new_x, type = "response")
    if (model_type %in% c('lasso','ridge','elastic_net')) {
      predictions = predictions[,'s0']
    }
  }
  if (is_binary_classification == TRUE){
    predictions <- ifelse(predictions >= model$classification_threshold, 1, 0)
  }
  return(list(predictions=predictions))
}
