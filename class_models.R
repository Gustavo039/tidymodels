## Data load and cleaning

data_cancer = readr::read_csv('./data.csv') |>
  dplyr::mutate(diagnosis = as.factor(diagnosis )) |>
  dplyr::select(id, diagnosis , dplyr::contains('_mean')) 

dplyr::glimpse(data_cancer)

data_cancer |>
  visdat::vis_miss()

data_cancer = data_cancer |>
  dplyr::select(-'...33')

data_cancer |>
  visdat::vis_miss()



## Data split
set.seed(607)

data_cancer_split = rsample::initial_split(data_cancer, prop = 0.75, strata = diagnosis)
data_cancer_train = rsample::training(data_cancer_split)
data_cancer_test = rsample::testing(data_cancer_split)

cv_folds = rsample::vfold_cv(data = data_cancer_train, 
  v = 10) 

## Exploratory data analysis

fa_cancer = data_cancer_train |>
  dplyr::select(dplyr::contains('_mean')) |>
  factanal(factors = 6, scores = "regression")

plot(fa_cancer$scores[,1], fa_cancer$scores[,2], col = as.factor(data_cancer$diagnosis))

data_cancer_train |>
  dplyr::select(-id, -diagnosis) |>
  cor() |>
  ggcorrplot::ggcorrplot(hc.order = TRUE, type = "lower", lab = TRUE)




## Modelling
cancer_recipe = recipes::recipe(diagnosis ~ .,
                                data = data_cancer_train) |>
                recipes::step_rm(id) |>
                recipes::step_corr() |>
                recipes::step_normalize(contains('_mean'))
  
cancer_recipe |>
  recipes::prep() |>
  recipes::bake(new_data = NULL)



logistic_spec = parsnip::logistic_reg() |>
  parsnip::set_engine("glm") |>
  parsnip::set_mode("classification")

dt_spec = parsnip::decision_tree() |>
  parsnip::set_engine('spark') |>
  parsnip::set_mode("classification")
  

rf_spec = parsnip::rand_forest() |>
  parsnip::set_engine("randomForest") |>
  parsnip::set_mode("classification")



cancer_wf = workflowsets::workflow_set(preproc = list(cancer_recipe),
                                       models = list(logistic_spec, rf_spec))

doParallel::registerDoParallel()

cancer_fit = workflowsets::workflow_map(cancer_wf,
                                        "fit_resamples",
                                        resamples = cv_folds,
                                        metrics = yardstick::metric_set(accuracy, roc_auc, rmse))

ggplot2::autoplot(cancer_fit,
                  metric = c('accuracy', 'roc_auc', 'mae'))




fitted_logistic_model = parsnip::logistic_reg() |>
  # Set the engine
  parsnip::set_engine("glm") |>
  # Set the mode
  parsnip::set_mode("classification") |>
  # Fit the model
  parsnip::fit(diagnosis ~ area_mean + texture_mean + compactness_mean + smoothness_mean, data = data_cancer_train)

broom::tidy(fitted_logistic_model) 


