

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

library(lightgbm)
################################################################################
# tidymodels xgboost ipaq category
################################################################################
rm(list = ls())
seed <- 42

cross <- read_excel('C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/cross_processed.xlsx')

gender <- read_xlsx('C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/Qualtrics_vragenlijst_fysiek_final_241024.xlsx')

cross$gender <- gender$gender





cross <- cross %>% 
  filter(!IPAQ_category == "1") %>% 
  mutate(IPAQ_category = ifelse(IPAQ_category == "2", 0, 1))


outcome <- factor(ifelse(cross$IPAQ_category == '1', 'Yes', 'No'), levels = c('Yes', 'No'))

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



outcome <- factor(ifelse(cross$IPAQ_category == '1', 'Yes', 'No'), levels = c('Yes', 'No'))




cross <- cross %>% 
  dplyr::select(-participant_id, -starts_with("ipaq"), -starts_with("IPAQ"))

cross$IPAQ_category <- outcome


cross <- cross %>% mutate(case_wts = ifelse(IPAQ_category == "Yes", 1, 2),
                          case_wts = importance_weights(case_wts))


model <- 'Elastic Net'
label <- 'IPAQ category'


set.seed(seed)
data_split <- initial_split(cross, strata = IPAQ_category, prop = 0.7)
data_train <- training(data_split)
data_test <- testing(data_split)
library(bonsai)  


spec_default <- boost_tree() %>%
  set_engine("lightgbm") %>%
  set_mode("classification")


rec_default <- recipe(IPAQ_category ~ ., data = data_train) %>%
  step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
  
  step_dummy(all_nominal_predictors()) 

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
    yardstick::bal_accuracy,
    yardstick::pr_auc
  )
)


collect_metrics(default_res)



preds <- collect_predictions(default_res) %>%
  mutate(.pred_class = factor(if_else(.pred_Yes >= 0.5, "Yes", "No"), levels = c("Yes", "No")))


collect_metrics(default_res)

conf_mat(preds, truth = IPAQ_category, estimate = .pred_class)




fitted_model <- extract_fit_parsnip(default_res)

vip(fitted_model$fit, num_features = 10) +
  ggtitle(paste('Most predictive features for\n', label, 'using', model))



set.seed(seed)
spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  learn_rate = tune()
) %>%
  set_engine("lightgbm", 
             lambda_l1 = tune(), 
             lambda_l2 = tune()
             , num_leaves = tune()) %>%
  set_mode("classification")


library(dials)
set.seed(seed)
params <- parameters(
  trees(),
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  learn_rate(),

  lambda_l1 = penalty(range = c(-5, 1)),  
  lambda_l2 = penalty(range = c(-5, 1))
  , num_leaves()
)


rec <- recipe(IPAQ_category ~ ., data = data_train) %>%
  step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
  step_dummy(all_nominal_predictors()) %>%  
  step_zv(all_predictors()) 

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(spec) %>% add_case_weights(case_wts)




set.seed(seed)

set.seed(seed)
data_folds <- vfold_cv(data_train, strata = IPAQ_category
                       , v = 5
)

data_folds


library(doParallel)


library(future)
plan(multisession, workers = parallel::detectCores() - 4)


# Bayesian tuning 
set.seed(seed)
(start_time <- Sys.time())
res <- tune_bayes(
  wf,
  resamples = data_folds,
  param_info = params,
  initial = 50,
  iter = 20,
  metrics = metric_set(
    yardstick::f_meas,
    yardstick::precision
  ),
  control = control_bayes(
    verbose = TRUE,
    no_improve = 10,
    seed = 123,
    save_pred = TRUE,
    allow_par = TRUE
  )
)
end_time <- Sys.time()
(parallel_time <- end_time - start_time)

ipaq_cat_lgbm_res <- res


res <- ipaq_cat_lgbm_res 



cross <- cross %>% 
  mutate(case_wts = ifelse(IPAQ_category == "Yes", 1, 2),
         case_wts = importance_weights(case_wts))

set.seed(seed)
data_split <- initial_split(cross, strata = IPAQ_category, prop = 0.70)
data_train <- training(data_split)
data_test <- testing(data_split)

collect_metrics(res)

best_parms <- select_best(res, metric = "precision")

spec <- boost_tree(
  trees = best_parms$trees,
  tree_depth = best_parms$tree_depth,
  min_n = best_parms$min_n,
  loss_reduction = best_parms$loss_reduction,
  sample_size = best_parms$sample_size,
  learn_rate = best_parms$learn_rate
) %>%
  set_engine("lightgbm",
             lambda_l1 = best_parms$lambda_l1,
             lambda_l2 = best_parms$lambda_l2
             , num_leaves = best_parms$num_leaves) %>%
  set_mode("classification")


rec <- recipe(IPAQ_category ~ ., data = data_train) %>%
  step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
  step_dummy(all_nominal_predictors()) %>%  
  step_zv(all_predictors()) 

final <- workflow() %>%
  add_recipe(rec) %>%
  add_model(spec) %>% add_case_weights(case_wts)

set.seed(seed)
final_fit <- fit(final, data = data_train)

final_res <- last_fit(final, data_split, metrics = metric_set(
  yardstick::f_meas,
  yardstick::precision,
  yardstick::recall,
  yardstick::spec,
  yardstick::accuracy,
  yardstick::bal_accuracy,
  yardstick::pr_auc
))

collect_metrics(final_res)

preds <- collect_predictions(final_res) %>%
  mutate(.pred_class = factor(if_else(.pred_Yes >= 0.5, "Yes", "No"), levels = c("Yes", "No")))

conf_mat(preds, truth = IPAQ_category, estimate = .pred_class)

label <- 'IPAQ Category'
model <- 'LightGBM'
vip(final_fit, num_features = 10) +
  ggtitle(paste('Most predictive features for\n', label, 'using', model))

