


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
# Logistic regression GDS category
################################################################################


rm(list = ls())
seed <- 42

sheet_names <- excel_sheets("C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/imputations/all_imputations.xlsx")

for (i in seq_along(sheet_names)){
  sheet_data <- read_excel("C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/imputations/all_imputations.xlsx", 
                           sheet = sheet_names[i])
  assign(paste0("cross", i), sheet_data, envir = .GlobalEnv)
}
cross_all <- list(cross1, cross2, cross3, cross4, cross5, 
                  cross6, cross7, cross8, cross9, cross10)

gender <- read_xlsx('C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/Qualtrics_vragenlijst_fysiek_final_241024.xlsx')



data_train <- list()
data_test <- list()

coef_df_list <- list()

predictions_list <- list()

length <- 10

for (i in 1:length) {
  cross <- cross_all[[i]]

  cross$gender <- gender$gender
  cross$sit_reach_values_3[is.na(cross$sit_reach_values_3)] <- 0
  
  outcome <- factor(ifelse(cross$gds_category == '1', 'Yes', 'No'), levels = c('Yes', 'No'))
  
  
  cross <- cross %>%
    mutate(across(everything(), ~ as.numeric(as.character(.))))
  

  
  
  
  for (col in names(cross)) {
    unique_vals <- length(unique(na.omit(cross[[col]])))
    if (unique_vals <= 5) {
      cross[[col]] <- as.factor(cross[[col]])
    }
  }
  
  
  cross <- cross %>%
    mutate(across(
      where(is.factor),
      ~ if (all(levels(.) %in% c("1", "2"))) {
        factor(ifelse(. == "2", "0", "1"), levels = c("0", "1"))
      } else {
        .
      }
    ))
  
  
  
  
  
  
  cross <- cross %>% 
    dplyr::select(-participant_id, -starts_with("gds"))
  
  cross$gds_category <- outcome
  
  cross <- cross %>% mutate(case_wts = ifelse(gds_category == "Yes", 3, 1), 
                            case_wts = importance_weights(case_wts))
  
  model <- 'Logistic Regression'
  label <- 'gds_category'
  
  
  
  

  
  cross$gds_category <- outcome
  
  set.seed(seed)
  data_split <- initial_split(cross, strata = gds_category, prop = 0.70)
  data_train[[i]] <- training(data_split)
  data_test[[i]] <- testing(data_split)
  
  
  
  spec_default <- logistic_reg() %>%
    set_engine("glm") %>%
    set_mode("classification")
  
  
  rec_default <- recipe(gds_category ~ ., data = data_train[[i]]) %>%
    step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
    step_dummy(all_nominal_predictors()) %>%  
    step_zv(all_predictors()) %>% 
    step_normalize(all_numeric_predictors()) %>% 
    step_corr(all_numeric_predictors(), threshold = 0.6)
  
  
  wf_default <- workflow() %>%
    add_recipe(rec_default) %>%
    add_model(spec_default) %>% add_case_weights(case_wts)
  
  
  
  library(FSelectorRcpp)
  
  
  rec_baked <- prep(rec_default, training = data_train[[i]])
  
  data_train_for_vip <- bake(rec_baked, new_data = data_train[[i]])
  
  data_train_for_vip <- data_train_for_vip %>% dplyr::select(
    -case_wts)
  
  
  
  vi_df <- information_gain(gds_category ~ . - case_wts, data = data_train[[i]])
  
  top_vars <- vi_df %>%
    arrange(desc(importance)) %>%
    slice_head(n = 10) %>%
    pull(attributes)
  
  library(stringr)
  
  cleaned_vars <- top_vars %>%
    str_remove("_X\\d+$") %>%
    unique()
  

  
  
  data_train[[i]] <- data_train[[i]] %>% dplyr::select(all_of(c(cleaned_vars, "gds_category", "case_wts")))
  data_test[[i]]  <- data_test[[i]] %>% dplyr::select(all_of(c(cleaned_vars, "gds_category")))
  data_test[[i]]  <- data_test[[i]] %>% dplyr::select(all_of(c(cleaned_vars, "gds_category")))
  
  
  rec_default <- recipe(gds_category ~ ., data = data_train[[i]]) %>%
    step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
    step_dummy(all_nominal_predictors()) %>%  
    step_zv(all_predictors()) %>% 
    step_normalize(all_numeric_predictors()) %>% 
    step_corr(all_numeric_predictors(), threshold = 0.6)
  
  
  
  wf_default <- workflow() %>%
    add_recipe(rec_default) %>%
    add_model(spec_default) %>% add_case_weights(case_wts)
  
  

  
  default_res <- last_fit(
    wf_default,
    split = data_split,
    metrics = metric_set(
      yardstick::f_meas,
      yardstick::precision,
      yardstick::recall,
      yardstick::spec,
      yardstick::accuracy,
      yardstick::bal_accuracy
      
      , yardstick::pr_auc
      
    )
  )
  
  
  collect_metrics(default_res)
  
  preds <- collect_predictions(default_res) %>%
    mutate(.pred_class = factor(if_else(.pred_Yes >= 0.5, "Yes", "No"), levels = c("Yes", "No")))
  
  collect_metrics(default_res)
  conf_mat(preds, truth = gds_category, estimate = .pred_class)
  
  
  
  final_model <- extract_fit_parsnip(default_res$.workflow[[1]])
  summary(final_model$fit)
  

  
  coef_df <- coef(summary(final_model$fit)) %>%
    as.data.frame() %>%
    rownames_to_column("feature") %>%
    dplyr::select(feature, coefficient = Estimate)
  
  coef_df_list[[i]] <- coef_df
  
  
  
  
  test_probs <- preds$.pred_Yes
  test_preds <- preds$.pred_class
  truth <- data_test[[i]]$gds_category
  
  
  predictions_list[[i]] <- tibble(
    truth = truth,
    .pred_class = test_preds,
    .pred_Yes = test_probs
  )
  
}







combined_coefs <- bind_rows(coef_df_list, .id = "imputation")
combined_predictions <- bind_rows(predictions_list, .id = "imputation")





all_preds <- bind_rows(predictions_list, .id = "imputation")


pred_list <- list()

for (i in 1:length) {
  pred_list[[i]] <- predictions_list[[i]]$.pred_Yes
}

avg_preds <- rowMeans(do.call(cbind, pred_list))

truth <- predictions_list[[1]]$truth  

final_avg_preds <- data.frame(
  .pred_Yes = avg_preds,
  truth = factor(truth, levels = c("Yes", "No")),
  .pred_class = factor(ifelse(avg_preds >= 0.5, "Yes", "No"), levels = c("Yes", "No"))
)
conf_mat(final_avg_preds, truth = truth, estimate = .pred_class)



truth <- final_avg_preds$truth
pred <- final_avg_preds$.pred_class
probs <- final_avg_preds$.pred_Yes

truth <- factor(truth, levels = c("Yes", "No"))
pred <- factor(pred, levels = c("Yes", "No"))

f1           <- f_meas_vec(truth, pred)
precision    <- precision_vec(truth, pred)
recall       <- recall_vec(truth, pred)
specificity  <- specificity_vec(truth, pred)
accuracy     <- accuracy_vec(truth, pred)
bal_accuracy <- bal_accuracy_vec(truth, pred)
pr_auc       <- pr_auc_vec(truth, probs, event_level = "first")


metrics <- tibble(
  Metric = c(
    "F1 Score",
    "Precision",
    "Recall (Sensitivity)",
    "Specificity",
    "Accuracy",
    "Bal. Accuracy",
    "PR_AUC"
  ),
  Value = c(
    f1,
    precision,
    recall,
    specificity,
    accuracy,
    bal_accuracy,
    pr_auc
  )
)
(metrics)
conf_mat(final_avg_preds, truth = truth, estimate = .pred_class)

model <- 'Logistic regression'
label <- 'GDS category'

all_coefs <- bind_rows(coef_df_list, .id = "imputation")

pooled_coefs <- all_coefs %>%
  group_by(feature) %>%
  summarise(mean_coef = mean(coefficient, na.rm = TRUE)) %>%
  ungroup()
pooled_coefs <- pooled_coefs %>%
  rename(coef = mean_coef) %>%
  filter(coef != 0)

intercept <- pooled_coefs %>%
  filter(feature == "(Intercept)") %>%
  pull(coef)

coefs <- pooled_coefs %>%
  filter(feature != "(Intercept)")


coef_df <- pooled_coefs %>%
  filter(feature != "(Intercept)", coef != 0) %>%
  mutate(
    direction = ifelse(coef > 0, "Positive", "Negative"),
    abs_coef = abs(coef)
  ) %>%
  slice_max(order_by = abs_coef, n = 10)



model <- 'Logistic Regression'
label <- 'gds_category'

ggplot(coef_df, aes(x = reorder(feature, abs_coef), y = abs_coef, fill = direction)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("Positive" = "dodgerblue", "Negative" = "red")) +
  labs(
    title = paste('Most predictive features for\n', label, 'using', model),
    x = "Feature",
    y = "Importance (|Coefficient|)",
    fill = "Effect Direction"
  ) +
  theme_minimal()

