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
  mutate(count=log(count)) %>% 
  select(-casual,-registered)

log_recipe <- recipe(count ~ ., data = log_bike) %>% 
  step_time(datetime, features=c("hour")) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% 
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>%
  step_num2factor(weather, levels=c("partly_cloudy", "misty", "rainy")) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("no", "yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("no", "yes"))) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_nzv(all_numeric_predictors())
  
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


# Tuning Models -----------------------------------------------------------
lin_pen_model <- linear_reg(penalty=tune(),
                              mixture=tune()) %>% 
  set_engine('glmnet')

## Set Workflow
lin_pen_wf <- workflow() %>% 
  add_recipe(log_recipe) %>% 
  add_model(lin_pen_model)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

## Split data for CV
folds <- vfold_cv(log_bike, v = 5, repeats=5)

## Run the CV
CV_results <- lin_pen_wf %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq))

## Plot Results
collect_metrics(CV_results) %>% #Gathers metrics into DF
  filter(.metric=='rmse') %>% 
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

## Find Best Tuning Parameters
best_Tune <- CV_results %>% 
  select_best('rmse')

## Finalize the Workflow & fit it
final_wf <- lin_pen_wf %>% 
  finalize_workflow(best_Tune) %>% 
  fit(data = log_bike)

## Predict
lin_pen_predictions <- final_wf %>% 
  predict(new_data = test)

# Create a dataframe that only has datetime and predictions (To upload to Kaggle)
lin_pen_preds <- data.frame(test$datetime, lin_pen_predictions)
colnames(lin_pen_preds) <- c('datetime', 'count')

# Change formatting of datetime
lin_pen_preds$datetime <- as.character(lin_pen_preds$datetime)

# Write that dataset to a csv file
vroom_write(lin_pen_preds, 'lin_pen_preds.csv', ",") 


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

# Regression Tree ---------------------------------------------------------
install.packages('rpart')
library(tidymodels)

log_recipe <- recipe(count ~ ., data = log_bike) %>% 
  step_time(datetime, features=c("hour")) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% 
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>%
  step_num2factor(weather, levels=c("partly_cloudy", "misty", "rainy")) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("no", "yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("no", "yes"))) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_nzv(all_numeric_predictors())

prepped_log_recipe <- prep(log_recipe)
bake(prepped_log_recipe, new_data = log_bike)

tree_mod <- decision_tree(tree_depth = tune(),
                          cost_complexity = tune(),
                          min_n = tune()) %>% # Type of model
  set_engine('rpart') %>%
  set_mode('regression')

## Set Workflow
tree_wf <- workflow() %>% 
  add_recipe(log_recipe) %>% 
  add_model(tree_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = 5)

## Split data for CV
folds <- vfold_cv(log_bike, v = 5, repeats=5)

## Run the CV
CV_results <- tree_wf %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq))

## Find Best Tuning Parameters
best_Tune <- CV_results %>% 
  select_best('rmse')

## Finalize the Workflow & fit it
final_tree_wf <- tree_wf %>% 
  finalize_workflow(best_Tune) %>% 
  fit(data = log_bike)

## Predict
## Get Predictions for test set AND format for Kaggle
tree_preds <- predict(final_tree_wf, new_data = test) %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write predictions to CSV
vroom_write(x=tree_preds, file="./tree_preds.csv", delim=",")


