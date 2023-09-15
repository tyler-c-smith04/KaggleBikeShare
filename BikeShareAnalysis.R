## Bike Share Clean Code

library(tidyverse)
library(vroom)
library(patchwork)
library(tidymodels)

bike <- vroom("./train.csv")


# Clean -------------------------------------------------------------------

# Take out weather == 4 since there is only one
bike %>% 
  filter(weather !=4)

bike <- bike %>% 
  mutate(weather = ifelse(weather == 4,3,weather))


# Make factors
bike$season <- as.factor(bike$season)
bike$holiday <- as.factor(bike$holiday)
bike$workingday <- as.factor(bike$workingday)
bike$weather <- as.factor(bike$weather)

my_recipe <- recipe(count ~ ., data = bike) %>% 
  step_date(datetime, features = 'dow') %>% 
  step_rm(registered, casual) %>% 
  step_time(datetime, features=c("hour"))

prepped_recipe <- prep(my_recipe)
bike_clean <- bake(prepped_recipe, new_data = bike)


