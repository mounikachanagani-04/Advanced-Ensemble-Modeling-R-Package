# data_handling.R

#' Calculate Mode
#'
#' Finds the most frequent value in a vector, excluding NA.
#'
#' @param x numerical list or vector.
#' @return The mode of 'x'.
calculate_mode <- function(x) {
  unique_x <- unique(x[!is.na(x)])
  unique_x[which.max(tabulate(match(x, unique_x)))]
}


# Function to impute missing values based on the specified method
#' Impute Missing Values
#'
#' Replaces missing values in a column using the specified method: mean, median, or mode.
#'
#' @param column to impute.
#' @param method imputation method ('mean', 'median', 'mode').
#' @return imputed column
#' 
#' @importFrom stats median
#' 
#' @examples
#' impute_missing_values(c(1, 2, NA, 4), method = "mean")
#' 
#' @export 
impute_missing_values <- function(column, method = "mean") {
  if (method == "mean") {
    return(ifelse(is.na(column), mean(column, na.rm = TRUE), column))
  } else if (method == "median") {
    return(ifelse(is.na(column), median(column, na.rm = TRUE), column))
  } else if (method == "mode") {
    mode_value <- calculate_mode(column)
    return(ifelse(is.na(column), mode_value, column))
  } else {
    stop("Unsupported imputation method.")
  }
}


#' Handle Missing Values in Data
#'
#' This function handles missing values in the dataset by either dropping columns
#' or imputing them based on the given method.
#'
#' @param y Response variable vector.
#' @param x Predictor variables dataframe.
#' @param drop flag indicating if records with missing values should be dropped. Default is TRUE.
#' @param method The method used for imputing missing values in case 'drop' is FALSE.
#' Supported methods are "mean", "median" for continous data and "mode" for categorical data.
#'
#' @return A list with modified 'y' and 'x'
#'
#' @examples
#' data <- data.frame(a = c(1, 2, NA), b = c(NA, 2, 3), c = c(1, 2, 3))
#' response <- c(1, NA, 2)
#' result <- handle_missing_values(response, data, drop = FALSE, method = "mean")
#'
#' @export
handle_missing_values <- function(y, x, drop = TRUE, method = "mean") {
  # Drop columns with all NA values
  na_cols = colSums(is.na(x)) == nrow(x)
  if (any(na_cols)) {
    x <- x[, !na_cols, drop = FALSE]
    log_message("Dropped columns with all NA values.\n")
  }

  if (!drop && (method %in% c("mean", "median", "mode"))) {
    # Impute missing values for numeric and categorical data
    for (col_name in names(x)) {
      if (is.numeric(x[[col_name]]) || is.integer(x[[col_name]])) {
        x[[col_name]] <- impute_missing_values(x[[col_name]], method = method)
      } else if (is.factor(x[[col_name]]) || is.character(x[[col_name]])) {
        x[[col_name]] <- impute_missing_values(x[[col_name]], method = "mode")
      }
    }
    log_message(paste0("Imputed missing values with ", method, ".\n"))
  } else if (drop) {
    # Drop rows with any NA values
    temp_df = data.frame(y = y, x)
    nrows_x_before_dropping_na = nrow(temp_df)
    # complete_data <- na.omit(cbind(y, x))
    complete_data <- na.omit(temp_df)
    y <- temp_df$y
    x <- temp_df[, -1, drop = FALSE]
    # y <- complete_data[, 1, drop = FALSE]
    # x <- complete_data[, -1, drop = FALSE]
    nrows_x_after_dropping_na = nrow(temp_df)
    log_message(sprintf("Dropped %s rows with missing values from the data.\n", (nrows_x_before_dropping_na - nrows_x_after_dropping_na)))
  }
  return(list(y = y, x = x))
}


#' This function checks if the response variable `y` is appropriate for the chosen feature selection method.
#'
#' @param y The response variable
#' @param method The feature selection method being used ("logistic" or "correlation" or "random_forest" or "auto").
#'
#' @return No return value; stops with an error message if assumptions are not met.
#'
#' @examples
#' # For a binary response variable
#' feature_selection_method_assumptions(c(0, 1, 0, 1), "logistic")
#'
#' # For a continuous response variable
#' feature_selection_method_assumptions(runif(10), "correlation")
#'
#' @export 
feature_selection_method_assumptions <- function(y, method) {

  if (method == "logistic" && !(length(unique(y)) == 2)) {
    stop("Selecting top k predictors: For logistic method, y must be binary.")
  } else if (method == "correlation" && !is.numeric(y)) {
    stop("Selecting top k predictors: For correlation method, y must be continuous.")
  }
}

#' Find the optimal value for the number of top predictors (k).
#'
#' This function calculates an optimal value for the number of top predictors (k) based on the size of the dataset.
#'
#' @param n_predictors The total number of predictors in the dataset.
#' @param n_records The total number of records in the dataset.
#'
#' @return The optimal value for the number of top predictors (k).
#'
#' @examples
#' find_optimal_k(100, 200)
#' # Returns 50
#'
#' @export 
find_optimal_k <- function(n_predictors, n_records) {
  # Set k to half the number of records or predictors, whichever is smaller
  k <- min(n_predictors, floor(n_records * 0.5))

  # If k equals the number of predictors, set k to half the number of predictors
  if (k == n_predictors) {
    k <- min(n_predictors, floor(n_predictors * 0.5))
  }

  return(k)
}


#' Check if the number of top predictors (k) is valid.
#'
#' This function checks if your chosen number of top predictors (k) is sensible for the dataset.
#'
#' @param k Number of top predictors you want to use. Must be a positive whole number.
#' @param x Dataset to check the number of predictors against. Can't have more predictors than variables.
#'
#' @return TRUE if 'k' is valid, FALSE otherwise.
#'
#' @examples
#' is_k_top_predictors_valid(5, iris) # Should be TRUE if iris has 5 or more variables.
#' is_k_top_predictors_valid(0, iris) # Will be FALSE, 0 is not a valid number.
#'
#' @export
is_k_top_predictors_valid = function(k, x){
  return(!is.null(k) && is.numeric(k) && !(k <= 0) && (k == round(k)) && !(k > ncol(x)))
}


#' Check if a feature selection method is valid.
#'
#' This function checks if a given feature selection method is valid based.
#'
#' @param method A character vector specifying the feature selection method to be validated.
#'
#' @return TRUE if the method is valid, FALSE otherwise.
#'
#' @details Valid feature selection methods include "auto", "logistic", "correlation", "random_forest".
#'
#' @examples
#' is_feature_selection_method_valid("random_forest")
#' # Returns TRUE
#'
#' is_feature_selection_method_valid("PCA")
#' # Returns FALSE
#'
#' @export
is_feature_selection_method_valid <- function(method) {
  return(!is.null(method) && method %in% c("auto", "logistic", "correlation", "random_forest", "lasso", "ridge", "elastic_net"))
}

#' Select Top K Predictors
#'
#' Selects top K predictors from 'x' for the response variable 'y' using specified or auto-detected method.
#'
#' @param x Predictor dataset.
#' @param y Response variable.
#' @param k Number of top predictors to select.
#' @param method Method for selection ('auto', 'logistic', 'correlation', 'random_forest'). 'auto' detects based on 'y'.
#' @return Names of the top K predictors.
#'
#' @importFrom stats glm
#' @importFrom randomForest randomForest importance
#'
#' @examples
#' n = 100
#' data <- data.frame(
#'     feature_num1 = rnorm(n, mean = 5, sd = 2),  # Numerical feature
#'     feature_num2 = rnorm(n, mean = 55, sd = 42),  # Numerical feature
#'     feature_num3 = rnorm(n, mean = 544, sd = 52),  # Numerical feature
#'     feature_num4 = rnorm(n, mean = 59, sd = 16),  # Numerical feature
#'     target = sample(0:1, n, replace = TRUE)  # Binary target variable
#'   )
#' x <- data[, -which(names(data) == "target")]
#' y <- data[["target"]]
#' select_top_k_predictors(x = x, y = y, k = 3, method = "auto")
#' 
#' @export
select_top_k_predictors <- function(x, y, k, method = "auto") {
  if (!(method %in% c("auto","logistic","correlation","random_forest"))) {
    method = 'auto'
  }
  if (method == "auto") { # Automatically determine the method based on the response variable if set to 'auto'
    method <- if (is_binary_response(y)) "logistic" else "correlation"
  }

  x = scale_continuous_data(x)$x_data

  if (length(unique(y)) == 2 && !is.factor(y)) {
    y <- as.factor(y)
  }

  feature_selection_method_assumptions(y, method)

  if (!is.data.frame(x)) as.data.frame(x)

  temp_df = data.frame(y = y, x)

  if (method == "correlation" || method == "logistic") {
    family = if (method == "correlation") "gaussian" else "binomial"
    glm_model <- glm(as.formula("y ~ ."), family = family, data = temp_df)
    scores = coef(glm_model)[-1]
    top_k_indices_glm = order(scores, decreasing = TRUE)[1:k]
    top_k_columns = names(scores[top_k_indices_glm])
  } else if (method == "random_forest") {
    temp_df = data.frame(y = y, x)
    rf_model <- randomForest(formula = as.formula("y ~ ."), data = temp_df, importance = TRUE)
    scores <- randomForest::importance(rf_model, type = 1) # Default type: Assuming using mean decrease in Gini
    top_k_indices_rf <- order(scores, decreasing = TRUE)[1:k]
    top_k_columns = names(x)[top_k_indices_rf]
  } else {
    stop("Unsupported method for selecting top k predictors.")
  }
  return(top_k_columns)
}


#' Convert and Clean Dataframe Columns
#'
#' Converts character and logical columns to factors, binary numeric columns to factors,
#' and splits Date columns into year, month, and day. Drops constant columns.
#'
#' @param dataframe Dataframe to be processed.
#' @return Dataframe with cleaned and potentially converted columns.
#' @importFrom stats na.omit
#' 
#' @examples
#' df <- data.frame(a = c(1, 1, 1), b = c(TRUE, FALSE, TRUE),
#'   date = as.Date(c('2020-01-01', '2020-01-02', '2020-01-03')))
#' cleaned_df <- convert_and_clean_columns(df)
#' @export 
convert_and_clean_columns <- function(dataframe) {
  cols_to_drop <- c()
  for (col_name in names(dataframe)) {
    # Check for constant columns (columns with a single unique value)
    if (length(unique(dataframe[[col_name]])) < 2) {
      cols_to_drop <- c(cols_to_drop, col_name)
      next # Skip further processing for this column
    }
    if (is.numeric(dataframe[[col_name]])) {
      # Check for binary numeric columns, accounting for possible NA values
      unique_values <- unique(na.omit(dataframe[[col_name]]))
      if (length(unique_values) == 2) {
        dataframe[[col_name]] <- factor(dataframe[[col_name]])
      }
      # Continuous numeric columns are left as is
    } else if (is.character(dataframe[[col_name]]) || is.logical(dataframe[[col_name]])) { # nolint
      # Convert logical and character columns to factor
      dataframe[[col_name]] <- factor(dataframe[[col_name]])
    } else if (inherits(dataframe[[col_name]], "Date")) {
      # Extract year, month, and potentially day from Date columns
      dataframe[paste0(col_name, "_Year")] <- as.integer(format(dataframe[[col_name]], "%Y")) # nolint
      dataframe[paste0(col_name, "_Month")] <- as.integer(format(dataframe[[col_name]], "%m")) # nolint
      dataframe[paste0(col_name, "_Day")] <- as.integer(format(dataframe[[col_name]], "%d")) # nolint
      cols_to_drop <- c(cols_to_drop, col_name)
    }
  }

  if (length(cols_to_drop) > 0) {
    dataframe <- dataframe[, !(names(dataframe) %in% cols_to_drop)]
  }
  return(dataframe)
}


#' Remove Outliers Using IQR Method
#'
#' Identifies and removes outliers from a dataframe based on the Interquartile Range (IQR) method.
#'
#' @param dataframe Dataframe containing predictor variables.
#' @param y Response variable.
#' @return A list with updated 'y' (response variable) and 'dataframe' excluding rows identified as outliers.
#' 
#' @importFrom stats quantile
#' 
#' @examples
#' df <- data.frame(a = rnorm(100), b = rnorm(100))
#' y <- rnorm(100)
#' cleaned_data <- remove_outliers_iqr(df, y)
#'
#' @export
remove_outliers_iqr <- function(dataframe, y) {
  temp_df = data.frame(y = y, dataframe)
  initial_row_count <- nrow(temp_df)
  continuous_cols <- names(temp_df)[sapply(temp_df, is.numeric)]
  outlier_indices <- c()  # Initialize an empty vector to store indices of outliers

  # Loop through each continuous column (except 'y') to find outliers
  for (col_name in continuous_cols[!continuous_cols %in% "y"]) {
    data <- temp_df[[col_name]]
    Q1 <- quantile(data, 0.25)
    Q3 <- quantile(data, 0.75)
    IQR <- Q3 - Q1
    # Find indices of outliers in the column
    outliers_in_col <- which(data < (Q1 - 1.5 * IQR) | data > (Q3 + 1.5 * IQR))
    outlier_indices <- unique(c(outlier_indices, outliers_in_col))  # Combine and deduplicate indices
  }

  if (length(outlier_indices) > 0 && length(outlier_indices) < initial_row_count) {
    # Remove rows containing outliers in any continuous column
    temp_df <- temp_df[-outlier_indices, , drop = FALSE]
    final_row_count <- nrow(temp_df)
    message(paste("Removed", initial_row_count - final_row_count, "rows containing outliers.\n"))
  } else {
    message("No outliers were detected and removed.\n")
  }
  y <- temp_df$y
  x <- temp_df[, -1, drop = FALSE]
  return(list(y = y, x = x))
}


#' Data Preprocessing Function
#'
#' Performs comprehensive preprocessing on any given dataset, including handling missing values,
#' converting and cleaning columns, removing outliers, encoding categorical variables,
#' selecting top K predictors, and scaling the data (optional).
#'
#' @param y Response variable.
#' @param x Predictor dataframe
#' @param models List of model names ('logistic_regression','linear_regression','lasso','ridge','elastic_net','random_forest)
#' @param feature_selection_method Method for feature selection.
#' @param k Number of top predictors.
#' @param drop_missing_records Boolean; flag that indicates whether to drop the records (TRUE by default)
#' @param fill_missing_method Method for imputing missing values ('mean' by default).
#' @param scale_data Boolean; flag that indicates whether to scale continous data.
#' @param remove_outliers Boolean; flag that indicates whether to remove outliers (FALSE by default).
#' @param seed Seed for reproducibility (123 by default).
#' @return A list containing preprocessed dataset x and additional information like scaling parameters, one-hot encoding levels, and names of selected top K features.
#' @examples
#' n = 100
#' data <- data.frame(
#'     feature_num1 = rnorm(n, mean = 5, sd = 2),  # Numerical feature
#'     feature_num2 = rnorm(n, mean = 55, sd = 42),  # Numerical feature
#'     feature_num3 = rnorm(n, mean = 544, sd = 52),  # Numerical feature
#'     feature_num4 = rnorm(n, mean = 59, sd = 16),  # Numerical feature
#'     feature_cat = sample(c("Category1", "Category2", "Category3"), 
#' n, replace = TRUE),  # Categorical feature
#'     target = sample(0:1, n, replace = TRUE)  # Binary target variable
#'   )
#' x <- data[, -which(names(data) == "target")]
#' y <- data[["target"]]
#' models <- c("logistic_regression","lasso")
#' preprocessed <- preprocess_data(y, x, models, scale_data = TRUE, remove_outliers = TRUE)
#' 
#' @export
preprocess_data <- function(
    y, x, models, feature_selection_method = NULL, k = NULL,
    drop_missing_records = TRUE, fill_missing_method = "mean",
    scale_data = FALSE, remove_outliers = FALSE, seed = 123) {

  if (!is.data.frame(x) && !is.matrix(x)) {
    x <- as.data.frame(x)
  }
  # Check if 'x' is empty
  if (nrow(x) == 0) {
    stop("The data frame 'x' is empty. This may occur if all rows were considered outliers and removed. Please check your outlier removal criteria.")
  }
  if (is.null(y)) {
    stop("response variable 'y' cannot be NULL.")
  }
  problem_type <- ifelse(is_binary_response(y), "classification", "regression")
  # Check alignment between 'y' and 'x'
  if (length(y) != nrow(x)) {
    stop("Length of 'y' does not match the number of rows in 'x'. Please ensure they are aligned before processing.")
  }
  if (!problem_type %in% c("classification", "regression")) {
    stop("Model type must be either 'classification' or 'regression'.")
  }
  if (problem_type == "classification" && length(unique(y)) != 2) {
    stop("This package only support binary classification")
  }
  if (problem_type == "regression" && !is.numeric(y)) {
    stop("For regression, 'y' must be numeric.")
  }
  if (problem_type == "classification" && !is.factor(y)) {
    y <- as.factor(y)
  }

  handled_data <- handle_missing_values(y, x, drop = drop_missing_records, method = fill_missing_method)
  y <- handled_data$y
  x <- handled_data$x

  x <- convert_and_clean_columns(x)

  if (remove_outliers == TRUE) {
    outlier_response <- remove_outliers_iqr(x, y)
    x = outlier_response$x
    y = outlier_response$y
  }
  one_hoted_x = x
  one_hot_levels = list()
  one_hot_response <- perform_one_hot_encoding(x, return_levels = TRUE) # This will now handle all factor columns automatically
  one_hoted_x = one_hot_response$data
  one_hot_levels = one_hot_response$levels

  feature_selected_x = one_hoted_x
  # Top k features selection
  is_fs_method_valid = is_feature_selection_method_valid(feature_selection_method)
  is_k_valid = is_k_top_predictors_valid(k, feature_selected_x)

  p = ncol(feature_selected_x)
  n = nrow(feature_selected_x)

  # top k predictors selection happen in the following scenarios:
  #   1. if p >> n
  #       1.1 Either Feature Selection method or k is valid.
  #       1.2 Model type is Logistic or Linear Regression
  #   2. If Either Feature Selection method or k is valid, irrespective of the model type.
  # Feature selection will not be performed if the following conditions are met.
  #   - If both Feature selection method and k are not valid, and the model type is either lasso, ridge, elastic net or random forest.
  top_k_feature_names = list()
  if (p > n){
    if (is_fs_method_valid || is_k_valid) {
      if (!is_fs_method_valid && is_k_valid) {
        feature_selection_method = if (is_binary_response(y)) 'logistic' else 'correlation'
        log_message(sprintf("Info: Feature selection method is not specified so, proceeding with a default selection method '%s' and k = %s", feature_selection_method, k))
      }
      if (is_fs_method_valid && !is_k_valid) {
        k = find_optimal_k(p, n)
        log_message(sprintf("Info: No.of top predictors value is not specified so, proceeding with an optimal k value %s and feature selection method = %s", k, feature_selection_method))
      }
      selected_features <- select_top_k_predictors(feature_selected_x, y, k, method = feature_selection_method)
      feature_selected_x <- feature_selected_x[, selected_features, drop = FALSE]  # Prevent conversion to vector
      top_k_feature_names = selected_features
    } else {
      # if (('logistic_regression' %in% models) || ('linear_regression' %in% models)) {
      if (any(c("logistic_regression", "linear_regression") %in% models)) {
        feature_selection_method = if (is_binary_response(y)) 'logistic' else 'correlation'
        k = find_optimal_k(ncol(feature_selected_x), nrow(feature_selected_x))
        log_message(paste(
          "Info: In the given data, the number of predictors is much larger than the number of observations,",
          "and one or more models you are trying to build are logistic regression or linear regression models.",
          "As these models aren't well-suited for handling such data where p >> n,",
          sprintf("we are selecting the top %s predictors using '%s' method.",k, feature_selection_method),
          "You can always consider to refit the model with an appropriate algorithm,",
          "feature selection technique, or adjusting the number of top predictors.,"))
        selected_features <- select_top_k_predictors(feature_selected_x, y, k, method = feature_selection_method)
        feature_selected_x <- feature_selected_x[, selected_features, drop = FALSE]  # Prevent conversion to vector
        top_k_feature_names = selected_features
      }
    }
  } else {
    if (is_fs_method_valid || is_k_valid) {
      if (!is_fs_method_valid && is_k_valid) {
        feature_selection_method = if (is_binary_response(y)) 'logistic' else 'correlation'
        log_message(sprintf("Info: Feature selection method is not specified so, proceeding with a default selection method '%s' and k = %s", feature_selection_method, k))
      }
      if (is_fs_method_valid && !is_k_valid) {
        k = find_optimal_k(ncol(feature_selected_x), nrow(feature_selected_x))
        log_message(sprintf("Info: No.of top predictors value is not specified so, proceeding with an optimal k value %s and feature selection method = %s", k, feature_selection_method))
      }
      selected_features <- select_top_k_predictors(feature_selected_x, y, k, method = feature_selection_method)
      feature_selected_x <- feature_selected_x[, selected_features, drop = FALSE]  # Prevent conversion to vector
      top_k_feature_names = selected_features
    }
  }

  scaled_x = feature_selected_x

  scale_params = list()
  if (scale_data == TRUE) {
    scale_output <- scale_continuous_data(feature_selected_x)
    scaled_x = scale_output$x_data
    scale_params = scale_output$scale_params
    if (!is.data.frame(scaled_x)) scaled_x <- as.data.frame(scaled_x)
  }

  # Final check/conversion before calling split_data
  if (!is.data.frame(scaled_x)) scaled_x <- as.data.frame(scaled_x)

  # split_data_result <- split_data(y, x, seed = seed)
  # return(split_data_result)
  return(list(x=scaled_x, y=y,
              scale_params=scale_params, one_hot_levels=one_hot_levels,
              top_k_feature_names=top_k_feature_names))
}

