


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
# Elastic Net IPAQ continuous
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



###################################################################
# MedAE metric
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



outcome <- cross$ipaq_METminperweek




cross <- cross %>% 
  dplyr::select(-participant_id, -starts_with("ipaq"), -starts_with("IPAQ"))

cross$ipaq_METminperweek <- outcome



model <- 'Elastic Net'
label <- 'ipaq_METminperweek'

set.seed(seed)
data_split <- initial_split(cross, strata = ipaq_METminperweek, prop = 0.70)
data_train[[i]] <- training(data_split)
data_test[[i]] <- testing(data_split)
}

(start_time <- Sys.time())
for(i in 1:10){
set.seed(seed)
data_folds <- vfold_cv(data_train[[i]], strata = ipaq_METminperweek, v = nrow(data_train[[i]]))
data_folds <- vfold_cv(data_train[[i]], strata = ipaq_METminperweek, v = 10
)

library(tune)
library(doParallel)

spec <- linear_reg(
  penalty = tune()
  ,mixture = tune()
) %>% 
  set_engine("glmnet"
  ) %>% 
  set_mode("regression")

params <- parameters(
  penalty(range = c(-5, 1))
  ,mixture(range = c(0, 1)))


rec <- recipe(ipaq_METminperweek ~ ., data = data_train[[i]]) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())




wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(spec) 





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
    mae
    , rmse
    , rsq
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

ipaq_cont_en_res <- res


best_parms <- select_best(res, metric = "mae")

set.seed(seed)
final <- finalize_workflow(wf, best_parms)



final_res <- last_fit(final, data_split, metrics = metric_set(
  mae
  , medae
  , rmse
  , rsq
  
))
collect_metrics(final_res)

final_fit <- fit(final, data = data_train[[i]])

(glmnet_model <- extract_fit_parsnip(final_fit)$fit)

(best_params <- select_best(res, metric = "mae"))
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
pred_list[[i]] <- predictions_list[[i]]$.pred
}

avg_preds <- rowMeans(do.call(cbind, pred_list))

truth <- predictions_list[[1]]$ipaq_METminperweek  

preds <- final_avg_preds <- data.frame(
  .pred = avg_preds,
  truth = truth
)


medae_result <- medae(preds, truth = truth , estimate = .pred)
mae_result <- mae(preds, truth = truth , estimate = .pred)

test_median <- median(data_test[[1]]$ipaq_METminperweek, na.rm = TRUE)
test_mean <- mean(data_test[[1]]$ipaq_METminperweek, na.rm = TRUE)


mae(preds, truth = truth , estimate = .pred)
medae(preds, truth = truth , estimate = .pred)
rmse(preds, truth = truth , estimate = .pred)
rsq(preds, truth = truth , estimate = .pred)

test_mean
test_median
collect_metrics(final_res)

mae_result[3] / test_mean
medae_result[3] / test_median




model <- 'Elastic Net'
label <- 'IPAQ continuous'

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


autoplot(res)


coef_df$direction <- ifelse(coef_df$coefficient > 0, "Positive", "Negative")

coef_df <- coef_df[coef_df$feature != "(Intercept)", ]
coef_df <- coef_df[order(abs(coef_df$coefficient), decreasing = TRUE), ][1:30, ]

coef_df$direction <- ifelse(coef_df$coefficient > 0, "Positive", "Negative")

ggplot(coef_df, aes(x = reorder(feature, abs(coefficient)), y = coefficient, fill = direction)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_fill_manual(values = c("Positive" = "blue", "Negative" = "red")) +
  labs(
    title = paste("Top 30 Most Predictive Features for", label, "using", model),
    x = "Feature",
    y = "Coefficient"
  ) +
  theme_minimal()




