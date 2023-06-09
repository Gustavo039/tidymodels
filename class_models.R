## Data load and cleaning

data_cancer = readr::read_csv('./data.csv')
dplyr::glimpse(data_cancer)

data_cancer |>
  visdat::vis_miss()

data_cancer = data_cancer |>
  dplyr::select(-'...33')

data_cancer |>
  visdat::vis_miss()



## Data split


## Exploratory data analysis

data_cancer_mean = data_cancer |>
  dplyr::select(id, diagnosis , dplyr::contains('_mean')) 

fa_cancer = data_cancer_mean |>
  dplyr::select(dplyr::contains('_mean')) |>
  factanal(factors = 6, scores = "regression")

plot(fa_cancer$scores[,1], fa_cancer$scores[,4], col = as.factor(data_cancer$diagnosis))


## Modelling

log_reg = glm(
  V1~., 
  data = as.numeric(as.factor(data_cancer$diagnosis))-1 |> cbind(fa_cancer$scores) |> as.data.frame(),
  family = binomial
)

summary(log_reg)
plot(log_reg)
