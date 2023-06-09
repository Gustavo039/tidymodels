ggplot2::mpg |>
  dplyr::glimpse()

mpg_data = ggplot2::mpg

## Data split ------------------------

mpg_split = rsample::initial_split(mpg_data,
                                   prop = 0.75,
                                   strata = hwy )

mpg_training = mpg_split |>
  rsample::training()

mpg_training = mpg_split |>
  rsample::testing()
