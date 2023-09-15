## Bike Share Clean Code

library(tidyverse)
library(vroom)
library(patchwork)
library(tidymodels)

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
