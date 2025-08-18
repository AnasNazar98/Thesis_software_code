


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
# LOO final analysis
################################################################################
rm(list = ls())
seed <- 42

cross <- read_excel('C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/cross_processed.xlsx')

gender <- read_xlsx('C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/Qualtrics_vragenlijst_fysiek_final_241024.xlsx')

cross$gender <- gender$gender



lopo <- read_excel("C:/Users/anasn/Desktop/E/Semester 4/Thesis/R code/xgboost timeseries/prediction plots xgboost/results/participant_success_loocv_single - Copy last reliable.xlsx")

cross <- cross %>%
  filter(participant_id  %in% lopo$participant_id )

cross$successful <- lopo$successful
cross$successful
class(cross$successful)


################################################################################
# Age

t_test_result <- t.test(Age ~ successful, data = cross)
t_test_result

wilcox_result <- wilcox.test(Age ~ successful, data = cross, exact = TRUE, correct = FALSE)
wilcox_result

################################################################################
# ipaq cont

t_test_result <- t.test(ipaq_METminperweek ~ successful, data = cross)
t_test_result

wilcox_result <- wilcox.test(ipaq_METminperweek ~ successful, data = cross, exact = TRUE, correct = FALSE)
wilcox_result


################################################################################
# gender

cross_filtered <- subset(cross, gender %in% c(1, 2))
cross_filtered$gender <- droplevels(cross_filtered$gender)

tab2 <- table(cross_filtered$gender, cross_filtered$successful)
tab2

chisq.test(tab2)

fisher.test(tab2)

################################################################################
# ipaq cat

cross_filtered <- subset(cross, IPAQ_category %in% c(2, 3))
cross_filtered$IPAQ_category <- droplevels(cross_filtered$IPAQ_category)

tab2 <- table(cross_filtered$IPAQ_category, cross_filtered$successful)
tab2

chisq.test(tab2)

fisher.test(tab2)



################################################################################
# fall

cross$falling_1 <- as.factor(cross$falling_1)

tab <- table(cross$falling_1, cross$successful)
tab

chisq.test(tab)

fisher.test(tab)


################################################################################
# depression

cross$gds_category <- as.factor(cross$gds_category)

tab <- table(cross$gds_category, cross$successful)
tab

chisq.test(tab)

fisher.test(tab)




