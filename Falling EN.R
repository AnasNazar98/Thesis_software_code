


rm(list = ls())
library(tidyverse)
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
library(themis)

################################################################################
# Elastic Net falling_1
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


for (i in 1:10) {
cross <- cross_all[[i]]
  
  
cross$gender <- gender$gender
  
cross$sit_reach_values_3[is.na(cross$sit_reach_values_3)] <- 0
  
cross <- cross %>%
    mutate(across(everything(), ~ as.numeric(as.character(.))))
  
  zero_var_indices <- nearZeroVar(cross)
  
  cross <- cross[, -zero_var_indices]
  

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



outcome <- factor(ifelse(cross$falling_1 == '1', 'Yes', 'No'), levels = c('Yes', 'No'))






cross <- cross %>% 
  dplyr::select(-participant_id, -starts_with("falling_1"))

cross$falling_1 <- outcome


cross <- cross %>% mutate(case_wts = ifelse(falling_1 == "Yes", 50, 1),
                          case_wts = importance_weights(case_wts))

model <- 'Elastic Net'
label <- 'Falling'

set.seed(seed)
data_split <- initial_split(cross, strata = falling_1, prop = 0.70)
data_train[[i]] <- training(data_split)
data_test[[i]] <- testing(data_split)
}

table(cross$falling_1)
(start_time <- Sys.time())
for(i in 1:10){
set.seed(seed)
data_folds <- vfold_cv(data_train[[i]], strata = falling_1, v = nrow(data_train[[i]]))
data_folds <- vfold_cv(data_train[[i]], strata = falling_1, v = 10
)

library(tune)
library(doParallel)

spec <- logistic_reg(
  penalty = tune()
  ,mixture = tune()
) %>% 
  set_engine("glmnet"
  ) %>% 
  set_mode("classification")

params <- parameters(
  penalty(range = c(-5, 1))
  ,mixture(range = c(0, 1)))



rec <- recipe(falling_1 ~ ., data = data_train[[i]]) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_upsample(falling_1, over_ratio = 10)




wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(spec) %>% add_case_weights(case_wts)




plan(sequential)
plan(multisession, workers = parallel::detectCores() - 2, gc = TRUE)

# Bayesian tuning
set.seed(seed)
res <- tune_bayes(
  wf,
  resamples = data_folds,
  param_info = params,
  initial = 20,               
  iter = 20,                   
  metrics = metric_set(
    yardstick::precision,
    pr_auc
  )
  ,control = control_bayes(
    verbose = T,
    no_improve = 20,
    seed = 123,
    save_pred = TRUE,
    allow_par = TRUE
  )
)

plan(sequential)
plan()

falling_en_res <- res




best_parms <- select_best(res, metric = "precision")

set.seed(seed)
final <- finalize_workflow(wf, best_parms)

final_res <- last_fit(final, data_split, metrics = metric_set(
  f_meas,
  yardstick::precision,
  yardstick::recall,
  yardstick::specificity,
  yardstick::accuracy,
  yardstick::bal_accuracy,
  pr_auc
  
))
collect_metrics(final_res)

final_fit <- fit(final, data = data_train[[i]])

(glmnet_model <- extract_fit_parsnip(final_fit)$fit)

(best_params <- select_best(res, metric = "precision"))  
(best_lambda <- best_params$penalty)
(best_alpha <- best_params$mixture)

coefs <- coef(glmnet_model, s = best_lambda)

coef_df <- data.frame(
  feature = rownames(coefs),
  coefficient = as.vector((coefs)))

coef_df_list[[i]] <- coef_df

predictions_list[[i]] <- collect_predictions(final_res)
}
end_time <- Sys.time()
(parallel_time <- end_time - start_time)

library(writexl)




combined_coefs <- bind_rows(coef_df_list, .id = "imputation")
combined_predictions <- bind_rows(predictions_list, .id = "imputation")





all_preds <- bind_rows(predictions_list, .id = "imputation")


pred_list <- list()

for (i in 1:10) {
pred_list[[i]] <- predictions_list[[i]]$.pred_Yes
}

avg_preds <- rowMeans(do.call(cbind, pred_list))

truth <- predictions_list[[1]]$falling_1  

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


model <- 'Elastic Net'
label <- 'Falling category'

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
  slice_max(order_by = abs_coef, n = 20)



model <- 'Elastic Net'
label <- 'Falling'

ggplot(coef_df, aes(x = reorder(feature, abs_coef), y = abs_coef, fill = direction)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("Positive" = "dodgerblue", "Negative" = "red")) +
  labs(
    title = paste('Most predictive features for\n', label, 'using', model),
    x = "Feature",y = "Importance (|Coefficient|)",
    fill = "Effect Direction"
  ) +
  theme_minimal()



