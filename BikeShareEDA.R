## Bike Share EDA Code

library(tidyverse)
library(vroom)
library(patchwork)

bike <- vroom("./train.csv")

install.packages("DataExplorer")
# Variable types, missing values
DataExplorer::plot_intro(bike)

# Highlights key feature of collinearity within variables
plot1 <- DataExplorer::plot_correlation(bike)

# Make season a factor
bike$season <- as.factor(bike$season)

# Boxplot of season and count
plot2 <- ggplot(data = bike, aes(x = season, y = count)) +
  geom_boxplot()

# Scatterplot showing datetime, temp, and count
plot3 <- ggplot(bike, aes(x = datetime, y = count, color = temp)) +
  geom_point() 

# Show distribution of workingday, count, and season
plot4 <- ggplot(bike, aes(x = workingday, y = count, color = season)) +
  geom_violin() +
  coord_flip()

four_panel <- (plot1 + plot2) / (plot3 + plot4)

ggsave("four_panel.png")
