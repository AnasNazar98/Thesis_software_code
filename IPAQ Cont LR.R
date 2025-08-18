

rm(list = ls())
library(tidyverse)
library(dplyr)
library(ggplot2)
library(skimr)
library(magrittr)
library(readxl)
library(writexl)
library(corrplot)
library(glmnet)
library(caret)
library(pROC)
library(xgboost)
library(PRROC)
library(tidymodels)
library(vip)
library(dials)
library(purrr)
library(tibble)
library(yardstick)
library(recipes)
library(finetune)
library(future)

################################################################################
# Elastic Net IPAQ Cont
################################################################################

rm(list = ls())

sheet_names <- excel_sheets("C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/imputations/all_imputations.xlsx")

for (i in seq_along(sheet_names)){
  sheet_data <- read_excel("C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/imputations/all_imputations.xlsx", 
                           sheet = sheet_names[i])
  assign(paste0("cross", i), sheet_data, envir = .GlobalEnv)
}
cross_all <- list(cross1, cross2, cross3, cross4, cross5, 
                  cross6, cross7, cross8, cross9, cross10)

cross <- cross1
gender <- read_xlsx('C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/Qualtrics_vragenlijst_fysiek_final_241024.xlsx')



data_train <- list()
data_test <- list()

coef_df_list <- list()

predictions_list <- list()


length_loop = 10
i <- 1
for (i in 1:length_loop) {
  cross <- cross_all[[i]]
  


cross$gender <- gender$gender



cross$sit_reach_values_3[is.na(cross$sit_reach_values_3)] <- 0

outcome <- ((cross$ipaq_METminperweek))


cross <- cross %>%
  mutate(across(everything(), ~ as.numeric(as.character(.))))




zero_var_indices <- nearZeroVar(cross)

cross <- cross[, -zero_var_indices]



cross <- cross %>%
  mutate(across(everything(), ~ as.numeric(as.character(.))))


numeric_data <- cross[sapply(cross, is.numeric)]

cor_matrix <- cor(numeric_data, use = "pairwise.complete.obs")

high_corr <- abs(cor_matrix) >= 0.6
high_corr[lower.tri(high_corr, diag = TRUE)] <- FALSE

to_drop <- unique(colnames(numeric_data)[apply(high_corr, 2, any)])

filtered_data <- cross[, !(names(cross) %in% to_drop)]
cross <- filtered_data






numeric_data <- cross[sapply(cross, is.numeric)]
cor_matrix <- cor(numeric_data, use = "pairwise.complete.obs")
high_corr <- abs(cor_matrix) >= 0.6
high_corr[lower.tri(high_corr, diag = TRUE)] <- FALSE
to_drop <- unique(colnames(numeric_data)[apply(high_corr, 2, any)])
filtered_data <- cross[, !(names(cross) %in% to_drop)]
cross <- filtered_data



for (col in names(cross)) {
  unique_vals <- length(unique(na.omit(cross[[col]])))
  if (unique_vals <= 5) {
    cross[[col]] <- as.factor(cross[[col]])
  }
}

cross <- cross %>%
  mutate(across(
    where(is.factor),
    ~ if (all(levels(.) %in% c("1", "2"))){
      factor(ifelse(. == "2", "0", "1"), levels = c("0", "1"))
    } else {
      .
    }
  ))




cross_dummy <- cross  

for (col in names(cross)) {
  if (is.factor(cross[[col]])) {
    levs <- levels(cross[[col]])
    
    for (lev in levs[-1]) {
      cross_dummy[[paste0(col, "_X", lev)]] <- ifelse(cross[[col]] == lev, 1, 0)
    }
    
    cross_dummy[[col]] <- NULL  
  }
}
cross <- cross_dummy

cross <- cross %>%
  mutate(across(everything(), ~ as.numeric(as.character(.))))



cross <- cross %>% 
  dplyr::select(-participant_id, -starts_with("IPAQ"), -starts_with("ipaq"))


model <- 'Logistic Regression'
label <- 'ipaq_METminperweek'




is_numeric <- sapply(cross, is.numeric)

is_dummy <- sapply(cross[, is_numeric], function(x) all(na.omit(unique(x)) %in% c(0, 1)))

continuous_vars <- names(which(is_numeric))[!is_dummy]

for (var in continuous_vars) {
  cross[[var]] <- scale(cross[[var]])
}




cross$ipaq_METminperweek <- outcome

set.seed(123)
data_split <- initial_split(cross, strata = ipaq_METminperweek, prop = 0.7)
data_train[[i]] <- training(data_split)
data_test[[i]] <- testing(data_split)

library(MASS)



null_model <- lm(ipaq_METminperweek ~ 1, data = data_train[[i]])

full_model <- lm(ipaq_METminperweek ~ ., data = data_train[[i]])

forward_model <- stepAIC(
  object = null_model,
  scope = list(lower = null_model, upper = full_model),
  direction = "forward",
  trace = TRUE
  , k = 10
  
)

summary(forward_model)


coef_df <- coef(summary(forward_model)) %>%
  as.data.frame() %>%
  rownames_to_column("feature") %>%
  dplyr::select(feature, coefficient = Estimate)

coef_df_list[[i]] <- coef_df




test_preds <- predict(forward_model, newdata = data_test[[i]], type = "response")

truth <- data_test[[i]]$ipaq_METminperweek



predictions_list[[i]] <- tibble(
  truth = truth,
  test_preds = test_preds
)

}



combined_coefs <- bind_rows(coef_df_list, .id = "imputation")
combined_predictions <- bind_rows(predictions_list, .id = "imputation")





all_preds <- bind_rows(predictions_list, .id = "imputation")


pred_list <- list()

for (i in 1:length_loop) {
  pred_list[[i]] <- predictions_list[[i]]$test_preds
}

avg_preds <- rowMeans(do.call(cbind, pred_list))
truth


library(tibble)

preds <- tibble(
  truth = truth,
  test_preds = avg_preds
)


library(yardstick)



library(vip)

vip(forward_model, 
    num_features = 30,
    geom = "col"
) +
  ggtitle(paste("Most predictive features for", label, "\nusing", model))



##############################################
# custom metric

medae_impl <- function(truth, estimate, case_weights = NULL) {
  median(abs(truth - estimate))
}

medae_vec <- function(truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
  check_numeric_metric(truth, estimate, case_weights)
  
  if (na_rm) {
    result <- yardstick_remove_missing(truth, estimate, case_weights)
    truth <- result$truth
    estimate <- result$estimate
    case_weights <- result$case_weights
  } else if (yardstick_any_missing(truth, estimate, case_weights)) {
    return(NA_real_)
  }
  
  medae_impl(truth, estimate, case_weights = case_weights)
}

medae <- function(data, ...) {
  UseMethod("medae")
}

medae <- new_numeric_metric(medae, direction = "minimize")

medae.data.frame <- function(data, truth, estimate,
                             na_rm = TRUE,
                             case_weights = NULL, ...) {
  numeric_metric_summarizer(
    name = "medae",
    fn = medae_vec,
    data = data,
    truth = !!rlang::enquo(truth),
    estimate = !!rlang::enquo(estimate),
    na_rm = na_rm,
    case_weights = !!rlang::enquo(case_weights)
  )
}

##############################################







medae_result <- medae(preds, truth = truth , estimate = test_preds)
mae_result <- mae(preds, truth = truth , estimate = test_preds)

test_median <- median(data_test[[i]]$ipaq_METminperweek)
test_mean <- mean(data_test[[i]]$ipaq_METminperweek)


mae(preds, truth = truth, estimate = test_preds)
medae(preds, truth = truth, estimate = test_preds)


test_mean
test_median

mae_result[3] / test_mean
medae_result[3] / test_median


