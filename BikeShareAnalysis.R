## Bike Share Clean Code

library(tidyverse)
library(vroom)
library(patchwork)
library(tidymodels)
library(poissonreg)

bike <- vroom("./train.csv")
bike <- bike %>% 
  select(-casual, -registered)

log_bike <- bike %>% 
  mutate(count = log(count))

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

# Clean -------------------------------------------------------------------

test <- vroom("./test.csv")

# Linear Regression -------------------------------------------------------

my_mod <- linear_reg() %>% # Type of model
  set_engine('lm') #Engine = What R Function to use

bike_workflow <- workflow() %>% 
  add_recipe(log_recipe) %>% 
  add_model(my_mod) %>% 
  fit(data = log_bike) # Fit the workflow

bike_predictions <- predict(bike_workflow,
                            new_data = test)

# Round negative numbers to 1 because we can't have negatives
bike_predictions[bike_predictions < 0] <- 0
view(bike_predictions)

lin_reg_preds <- predict(bike_workflow, new_data = test) %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

# Write that dataset to a csv file
vroom_write(x=lin_reg_preds, file="./lin_reg_preds.csv", delim=",")

# Look at the fitted LM model
extract_fit_engine(bike_workflow) %>% 
  tidy()

extract_fit_engine(bike_workflow) %>% 
  summary

# Poisson Regression ------------------------------------------------------
pois_mod <- poisson_reg() %>% # Type of model
  set_engine('glm') #Engine = What R Function to use

pois_wf <- workflow() %>% 
  add_recipe(log_recipe) %>% 
  add_model(pois_mod) %>% 
  fit(data = log_bike) # Fit the workflow

pois_preds <- predict(pois_wf,
                            new_data = test)

pois_preds <- predict(pois_wf, new_data = test) %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

# Write that dataset to a csv file
vroom_write(x=pois_preds, file="./pois_preds.csv", delim=",")

# Penalized Regression ---------------------------------------------------
  
## Penalized regression model10
preg_model <- linear_reg(penalty=0, mixture=0) %>% #Set model and tuning11
  set_engine("glmnet") # Function to fit in R12

preg_wf <- workflow() %>% 
  add_recipe(log_recipe) %>% 
  add_model(preg_model) %>% 
  fit(data = log_bike)

pen_preds <- predict(preg_wf, new_data = test)

pen_preds <- predict(preg_wf, new_data = test) %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

# Write that dataset to a csv file
vroom_write(x=pen_preds, file="./pen_preds.csv", delim=",")

# Tuning Models -----------------------------------------------------------
penalized_model <- linear_reg(penalty=tune(),
                              mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

# Set Workflow
tuning_wf <- workflow() %>%
  add_recipe(log_recipe) %>%
  add_model(penalized_model)

# Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

# split data for cross validation
folds <- vfold_cv(log_bike, v = 5, repeats = 5)

## Run the CV
CV_results <- tuning_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae)) #Or leave metrics NULL

# plot results
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

# Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("rmse")

# Finalize the Workflow & fit it
tuning_wf <- tuning_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=log_bike)

penalized_model_preds <- predict(tuning_wf, new_data = test) %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=penalized_model_preds, file="./penalized_model_preds.csv", delim=",")

# Regression Tree ---------------------------------------------------------
install.packages('rpart')
library(tidymodels)
library(rpart)

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
reg_tree_tuning_grid <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = 5)

## Split data for CV
folds <- vfold_cv(log_bike, v = 5, repeats=5)

## Run the CV
CV_results <- tree_wf %>% 
  tune_grid(resamples=folds,
            grid=reg_tree_tuning_grid,
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

# Random Forests ----------------------------------------------------------
install.packages('rpart')
library(tidymodels)
library(rpart)
library(ranger)

forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) %>%
  set_engine('ranger') %>%  # What R function to use
  set_mode("regression")

## Set Workflow
rand_wf <- workflow() %>% 
  add_recipe(log_recipe) %>% 
  add_model(forest_mod)

# Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range = c(1, (ncol(log_bike)-1))),
                            min_n(),
                            levels = 5)

# Set up K-fold CV
folds <- vfold_cv(log_bike, v = 5, repeats=5)

## Run the CV
rf_cv_results <- rand_wf %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq))

## Find Best Tuning Parameters
best_Tune <- rf_cv_results %>% 
  select_best('rmse')

## Finalize the Workflow & fit it
final_rand_wf <- rand_wf %>% 
  finalize_workflow(best_Tune) %>% 
  fit(data = log_bike)

## Get Predictions for test set AND format for Kaggle
rand_preds <- predict(final_rand_wf, new_data = test) %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write predictions to CSV
vroom_write(x=rand_preds, file="./rand_preds.csv", delim=",")


# Linear Model ------------------------------------------------------------


### Linear Model
lin_mod <- linear_reg() %>% # Type of model
  set_engine('lm') #Engine = What R Function to use

bike_workflow <- workflow() %>% 
  add_recipe(log_recipe) %>% 
  add_model(lin_mod) %>% 
  fit(data = log_bike) # Fit the workflow

bike_predictions <- predict(bike_workflow,
                            new_data = test)

# Model Stacking ----------------------------------------------------------
library(stacks)
## Split data for CV
stacking_folds <- vfold_cv(log_bike, v = 10, repeats=1)

## Create a control grid
untunedModel <- control_stack_grid() #If tuning over a grid
tunedModel <- control_stack_resamples() #If not tuning a model

## Penalized regression model
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

## Set Workflow
preg_wf <- workflow() %>%
  add_recipe(log_recipe) %>%
  add_model(preg_model)

## Grid of values to tune over
preg_tuning_grid <- grid_regular(penalty(),
                                 mixture(),
                                 levels = 5) ## L^2 total tuning possibilities

## Run the CV
preg_models <- preg_wf %>%
  tune_grid(resamples=stacking_folds,
            grid=preg_tuning_grid,
            metrics=metric_set(rmse),
            control = untunedModel) # including the control grid in the tuning ensures you can
# call on it later in the stacked model

## Create other resampling objects with different ML algorithms to include in a stacked model, for example
lin_reg <-
  linear_reg() %>%
  set_engine("lm")
lin_reg_wf <-
  workflow() %>%
  add_model(lin_reg) %>%
  add_recipe(log_recipe)
lin_reg_model <-
  fit_resamples(
    lin_reg_wf,
    resamples = stacking_folds,
    metrics = metric_set(rmse),
    control = tunedModel
  )

# # add regression tree
reg_tree <- decision_tree(tree_depth = tune(),
                          cost_complexity = tune(),
                          min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # Engine = What R function to use
  set_mode("regression")

reg_tree_wf <-
  workflow() %>%
  add_model(reg_tree) %>%
  add_recipe(log_recipe)

reg_tree_tuning_grid <- grid_regular(tree_depth(),
                                     cost_complexity(),
                                     min_n(),
                                     levels = 6)

reg_tree_model <-
  tune_grid(
    reg_tree_wf,
    grid = reg_tree_tuning_grid,
    resamples = stacking_folds,
    metrics = metric_set(rmse),
    control = untunedModel)


## Specify with models to include
my_stack <- stacks() %>%
  add_candidates(preg_models) %>%
  add_candidates(lin_reg_model) %>% 
  add_candidates(reg_tree)

## Fit the stacked model
stack_mod <- my_stack %>%
  blend_predictions() %>% # LASSO penalized regression meta-learner
  fit_members() ## Fit the members to the dataset

## If you want to build your own metalearner you'll have to do so manually
## using
stackData <- as_tibble(my_stack)

## Use the stacked data to get a prediction
stack_mod %>% predict(new_data=test)

# Get Predictions for test set AND format for Kaggle
stack_preds <- predict(stack_mod, new_data = test) %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

# Write prediction file to CSV
vroom_write(x=stack_preds, file="./stacking_predictions.csv", delim=",")


# Bart Model --------------------------------------------------------------
# create bart model and workflow
library(dbarts)
bart_mod <- bart(trees = tune(),
                   prior_terminal_node_coef = tune(),
                   prior_terminal_node_expo = tune()) %>% 
  set_engine("dbarts") %>% 
  set_mode("regression")

bart_wf <-
  workflow(log_recipe) %>%
  add_model(bart_mod)

# create tuning grid
tuning_grid <- grid_regular(trees(),
                            prior_terminal_node_coef(),
                            prior_terminal_node_expo(),
                            levels = 5)

# cross validation
cv <- vfold_cv(data = log_bike, v = 5, repeats = 1)

cv_results <- bart_wf %>%
  tune_grid(resamples = cv,
            grid = 3,
            metrics=metric_set(rmse))

# Get the best hyperparameters
best_params <- cv_results %>%
  select_best('rmse')

# use the best model
final_wf_bart <- bart_wf %>%
  finalize_workflow(best_params) %>%
  fit(data=log_bike)

# Get Predictions for test set AND format for Kaggle
bart_preds <- predict(final_wf_bart, new_data = test) %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

# Write prediction file to CSV
vroom_write(x=bart_preds, file="./bart_predictions.csv", delim=",")


