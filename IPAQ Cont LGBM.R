

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
# tidymodels xgboost ipaq_METminperweek
################################################################################
rm(list = ls())
seed <- 42

cross <- read_excel('C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/cross_processed.xlsx')

gender <- read_xlsx('C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/Qualtrics_vragenlijst_fysiek_final_241024.xlsx')

cross$gender <- gender$gender

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



outcome <- ((cross$ipaq_METminperweek))




cross <- cross %>% 
  dplyr::select(-participant_id, -starts_with("ipaq"), -starts_with("IPAQ"))

cross$ipaq_METminperweek <- outcome





model <- 'LightGBM'
label <- 'ipaq_METminperweek'


set.seed(seed)
data_split <- initial_split(cross, strata = ipaq_METminperweek, prop = 0.7)
data_train <- training(data_split)
data_test <- testing(data_split)
library(bonsai)  

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


spec_default <- boost_tree() %>%
  set_engine("lightgbm") %>%
  set_mode("regression")


rec_default <- recipe(ipaq_METminperweek ~ ., data = data_train) %>%
  step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
  
  step_dummy(all_nominal_predictors()) 

wf_default <- workflow() %>%
  add_recipe(rec_default) %>%
  add_model(spec_default) 




default_res <- last_fit(
  wf_default,
  split = data_split,
  metrics = metric_set(
    yardstick::mae
    , medae
    , yardstick::rmse
    , yardstick::rsq
  )
)


collect_metrics(default_res)



preds <- collect_predictions(default_res)

library(yardstick)

medae_result <- medae(preds, truth = ipaq_METminperweek, estimate = .pred)
mae_result <- yardstick::mae(preds, truth = ipaq_METminperweek, estimate = .pred)

test_median <- median(data_test$ipaq_METminperweek, na.rm = TRUE)
test_mean <- mean(data_test$ipaq_METminperweek, na.rm = TRUE)

collect_metrics(default_res)
test_mean
test_median

mae_result[3] / test_mean
medae_result[3] / test_median


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
             , num_leaves = tune()
             , objective = "quantile",
             alpha = 0.5) %>%
  set_mode("regression")


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



set.seed(seed)
wf <- workflow() %>%
  add_formula(ipaq_METminperweek ~ .) %>%
  add_model(spec) 

set.seed(seed)

set.seed(seed)
data_folds <- vfold_cv(data_train, strata = ipaq_METminperweek
                       , v = 10
)

data_folds



library(future)
plan(multisession, workers = parallel::detectCores() - 4)


set.seed(seed)
(start_time <- Sys.time())
res <- tune_bayes(
  wf,
  resamples = data_folds,
  param_info = params,
  initial = 50,
  iter = 20,
  metrics = metric_set(
     yardstick::mae
    , yardstick::rmse
    , yardstick::rsq
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

ipaq_cont_lgbm_res <- res


set.seed(seed)
data_split <- initial_split(cross, strata = ipaq_METminperweek, prop = 0.70)
data_train <- training(data_split)
data_test <- testing(data_split)



collect_metrics(res)

best_parms <- select_best(res, metric = "mae")

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
  set_mode("regression")

final <- workflow() %>%
  add_formula(ipaq_METminperweek ~ .) %>%
  add_model(spec) 
set.seed(seed)
final_fit <- fit(final, data = data_train)




final_res <- last_fit(final, data_split, metrics = metric_set(
  yardstick::mae
  , medae
  , yardstick::rmse
  , yardstick::rsq
))

collect_metrics(final_res)



vip(final_fit, num_features = 10) +
  ggtitle(paste('Most predictive features for\n', label, 'using', model))


