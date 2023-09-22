## Bike Share Clean Code

library(tidyverse)
library(vroom)
library(patchwork)
library(tidymodels)
library(poissonreg)

bike <- vroom("./train.csv")
bike <- bike %>% 
  select(-casual, -registered)

# Clean -------------------------------------------------------------------

my_recipe <- recipe(count ~ ., data = bike) %>% 
  step_time(datetime, features=c("hour")) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% 
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>%
  step_num2factor(weather, levels=c("partly_cloudy", "misty", "rainy")) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("no", "yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("no", "yes")))

prepped_recipe <- prep(my_recipe)
bike_clean <- bake(prepped_recipe, new_data = bike)

test <- vroom("./test.csv")


# Linear Regression -------------------------------------------------------

my_mod <- linear_reg() %>% # Type of model
  set_engine('lm') #Engine = What R Function to use

bike_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(my_mod) %>% 
  fit(data = bike) # Fit the workflow

bike_predictions <- predict(bike_workflow,
                            new_data = test)

# Round negative numbers to 1 because we can't have negatives
bike_predictions[bike_predictions < 0] <- 0
view(bike_predictions)

# Create a dataframe that only has datetime and predictions (To upload to Kaggle)
predictions <- data.frame(test$datetime, bike_predictions)
colnames(predictions) <- c('datetime', 'count')

# Change formatting of datetime
predictions$datetime <- as.character(predictions$datetime)

# Write that dataset to a csv file
vroom_write(predictions, 'predictions.csv', ",")

# Look at the fitted LM model
extract_fit_engine(bike_workflow) %>% 
  tidy()

extract_fit_engine(bike_workflow) %>% 
  summary

# Poisson Regression ------------------------------------------------------
pois_mod <- poisson_reg() %>% # Type of model
  set_engine('glm') #Engine = What R Function to use

bike_pois_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(pois_mod) %>% 
  fit(data = bike) # Fit the workflow

bike_pois_predictions <- predict(bike_pois_workflow,
                            new_data = test)

# Create a dataframe that only has datetime and predictions (To upload to Kaggle)
pois_predictions <- data.frame(test$datetime, bike_pois_predictions)
colnames(pois_predictions) <- c('datetime', 'count')

# Change formatting of datetime
pois_predictions$datetime <- as.character(pois_predictions$datetime)

# Write that dataset to a csv file
vroom_write(pois_predictions, 'pois_predictions.csv', ",")

pois_mod <- poisson_reg() %>% # Type of model
  set_engine('glm') #Engine = What R Function to use

bike_pois_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(pois_mod) %>% 
  fit(data = bike) # Fit the workflow

bike_pois_predictions <- predict(bike_pois_workflow,
                                 new_data = test)

# Penalized Regression ----------------------------------------------------

log_bike <- bike %>% 
  mutate(count=log(count))

log_recipe <- recipe(count ~ ., data = log_bike) %>% 
  step_time(datetime, features=c("hour")) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% 
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>%
  step_num2factor(weather, levels=c("partly_cloudy", "misty", "rainy")) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("no", "yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("no", "yes"))) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())
  
prepped_pen_recipe <- prep(log_recipe)
bake(prepped_pen_recipe, new_data = log_bike)

## Penalized regression model10
preg_model <- linear_reg(penalty=0, mixture=0) %>% #Set model and tuning11
  set_engine("glmnet") # Function to fit in R12

preg_wf <- workflow() %>% 
  add_recipe(log_recipe) %>% 
  add_model(preg_model) %>% 
  fit(data = log_bike)

pen_predictions <- predict(preg_wf, new_data = test)

# Create a dataframe that only has datetime and predictions (To upload to Kaggle)
pen_predictions <- data.frame(test$datetime, pen_predictions)
colnames(pen_predictions) <- c('datetime', 'count')

# Change formatting of datetime
pen_predictions$datetime <- as.character(pen_predictions$datetime)

# Write that dataset to a csv file
vroom_write(pen_predictions, 'pen_predictions.csv', ",") 


# Test with Poisson and log_bike (BAD) ------------------------------------------
pois_mod <- poisson_reg() %>% # Type of model
  set_engine('glm') #Engine = What R Function to use

log_pois_workflow <- workflow() %>% 
  add_recipe(log_recipe) %>% 
  add_model(pois_mod) %>% 
  fit(data = log_bike) # Fit the workflow

log_pois_predictions <- predict(log_pois_workflow,
                                 new_data = test)

# Create a dataframe that only has datetime and predictions (To upload to Kaggle)
log_pois_predictions <- data.frame(test$datetime, log_pois_predictions)
colnames(log_pois_predictions) <- c('datetime', 'count')

# Change formatting of datetime
log_pois_predictions$datetime <- as.character(log_pois_predictions$datetime)

# Write that dataset to a csv file
vroom_write(log_pois_predictions, 'log_pois_predictions.csv', ",") 


# Test log linear model (BAD)---------------------------------------------------
my_mod <- linear_reg() %>% # Type of model
  set_engine('lm') #Engine = What R Function to use

log_bike_workflow <- workflow() %>% 
  add_recipe(log_recipe) %>% 
  add_model(my_mod) %>% 
  fit(data = log_bike) # Fit the workflow

log_lin_predictions <- predict(log_bike_workflow,
                            new_data = test)

# Round negative numbers to 1 because we can't have negatives
log_lin_predictions[log_lin_predictions < 0] <- 0

# Create a dataframe that only has datetime and predictions (To upload to Kaggle)
log_lin_predictions <- data.frame(test$datetime, log_lin_predictions)
colnames(log_lin_predictions) <- c('datetime', 'count')

# Change formatting of datetime
log_lin_predictions$datetime <- as.character(log_lin_predictions$datetime)

# Write that dataset to a csv file
vroom_write(log_lin_predictions, 'log_lin_predictions.csv', ",")

