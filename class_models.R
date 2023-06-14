## Data load and cleaning

data_cancer = readr::read_csv('./data.csv') |>
  dplyr::mutate(diagnosis = as.factor(diagnosis )) |>
  dplyr::select(id, diagnosis , dplyr::contains('_mean')) |>
  dplyr::select(-radius_mean, -perimeter_mean, -compactness_mean) 
  

dplyr::glimpse(data_cancer)

data_cancer |>
  visdat::vis_miss()




## Data split
set.seed(607)

data_cancer_split = rsample::initial_split(data_cancer, prop = 0.80, strata = diagnosis)
data_cancer_train = rsample::training(data_cancer_split)
data_cancer_test = rsample::testing(data_cancer_split)

cv_folds = rsample::vfold_cv(data = data_cancer_train, 
                             strata = diagnosis, 
                             v = 10) 


  

## Exploratory data analysis
plots = sapply(names(data_cancer_train[,3:12]),  
         function(var_)
           {
             dd_test = data_cancer_train |>
               dplyr::select(diagnosis, var_) |>
               dplyr::group_by(diagnosis) |>
               dplyr::group_map(~.x)

              ks = ks.test((dd_test[[1]])|>dplyr::pull(var_), (dd_test[[2]])|>dplyr::pull(var_), paired = F)$p.value
              cr = cramer::cramer.test((dd_test[[1]])|>dplyr::pull(var_), (dd_test[[2]])|>dplyr::pull(var_))$p.value
              
              print(ks)
              print(cr)
              

              data_cancer_train |>
              ggplot2::ggplot(ggplot2::aes(x = var_, color = diagnosis)) +
              ggplot2::geom_histogram() +
              # ggplot2::annotate(paste('ks-test' = ks)) +
              ggplot2::labs(title = var_)
           }
        )


hist_vars_cancer = apply(data_cancer_train[,3:12], 2, 
      function(vars_){
          data_cancer_train |>
          ggplot2::ggplot(ggplot2::aes(x = vars_, color = diagnosis)) +
          ggplot2::geom_histogram(ggplot2::aes(y = ..density..)) 

})

do.call(gridExtra::grid.arrange, hist_vars_cancer)

fa_ml_cancer = data_cancer_train |>
  dplyr::select(dplyr::contains('_mean')) |>
  psych::fa(nfactors = 2, rotate = 'varimax', fm = 'ml') 

fa_ml_cancer |> psych::fa.diagram()
fa_ml_cancer$scores |>
  as.data.frame() |>
  dplyr::mutate(diagnosis = data_cancer_train$diagnosis) |>
  ggplot2::ggplot(ggplot2::aes(ML1, ML2, col = diagnosis)) +
  ggplot2::geom_point(size = 2) +
  ggplot2::labs(title = 'Redução via Verossimilhança') +
  ggthemes::scale_colour_colorblind()


fa_pc_cancer = data_cancer_train |>
  dplyr::select(dplyr::contains('_mean')) |>
  psych::fa(nfactors = 2, rotate = 'varimax', fm = 'pa') 

fa_pc_cancer |> psych::fa.diagram()
fa_pc_cancer$scores |>
  as.data.frame() |>
  dplyr::mutate(diagnosis = data_cancer_train$diagnosis) |>
  ggplot2::ggplot(ggplot2::aes(PA1, PA2, col = diagnosis)) +
  ggplot2::geom_point(size = 2) +
  ggplot2::labs(title = 'Redução via Componentes Principais') +
  ggthemes::scale_colour_colorblind()


fa_mr_cancer = data_cancer_train |>
  dplyr::select(dplyr::contains('_mean')) |>
  psych::fa(nfactors = 2, rotate = 'varimax', fm = 'mr')

fa_mr_cancer |> psych::fa.diagram()
fa_mr_scoreplot = fa_mr_cancer$scores |>
  as.data.frame() |>
  dplyr::mutate(diagnosis = data_cancer_train$diagnosis) |>
  ggplot2::ggplot(ggplot2::aes(MR1, MR2, col = diagnosis)) +
  ggplot2::geom_point(size = 2) +
  ggplot2::labs(title = 'Redução via Componentes Principais') +
  ggthemes::scale_colour_colorblind()


pr_cancer = data_cancer_train |>
  dplyr::select(dplyr::contains('_mean')) |>
  cor() |>
  princomp()

pr_cancer |> summary()
fviz_pca_var(pr_cancer)

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


