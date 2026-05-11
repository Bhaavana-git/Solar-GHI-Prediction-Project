# ====================================
# SOLAR GHI PREDICTION PROJECT
# ====================================
install.packages(c(
  "readr",
  "dplyr",
  "ggplot2",
  "corrplot",
  "caret",
  "randomForest",
  "Metrics",
  "lubridate",
  "gridExtra",
  "xgboost"
))
library(readr)
library(dplyr)
library(ggplot2)
library(corrplot)
library(caret)
library(randomForest)
library(Metrics)
library(lubridate)
library(gridExtra)
library(xgboost)
# -------------------------------
# 1. Load Dataset
# -------------------------------

data <- read.csv("solar_cleaned_final.csv")

str(data)
summary(data)
colSums(is.na(data))


# -------------------------------
# 2. Convert DateTime
# -------------------------------

data$DateTime <- as.POSIXct(
  data$DateTime,
  format = "%d-%m-%Y %H:%M"
)
head(data$DateTime)


# -------------------------------
# 3. Feature Engineering
# -------------------------------

data$Hour <- as.numeric(format(data$DateTime, "%H"))
data$Month <- as.numeric(format(data$DateTime, "%m"))
data$DayOfYear <- as.numeric(format(data$DateTime, "%j"))



# -------------------------------
#Exploratory Data Analysis
# -------------------------------
#Scatter Plots

#GHI vs Temperature
ggplot(data, aes(x = Temperature, y = GHI)) +
  geom_point(color = "blue", alpha = 0.5) +
  labs(title = "GHI vs Temperature") +
  theme_minimal()


#GHI vs Wind Speed
ggplot(data, aes(x = Wind.Speed, y = GHI)) +
  geom_point(color = "darkgreen", alpha = 0.5) +
  labs(title = "GHI vs Wind Speed") +
  theme_minimal()


#GHI vs Hour
ggplot(data, aes(x = Hour, y = GHI)) +
  geom_point(color = "purple", alpha = 0.5) +
  labs(title = "GHI vs Hour") +
  theme_minimal()

# Histogram
ggplot(data, aes(x = GHI)) +
  geom_histogram(bins = 30, fill = "orange") +
  labs(title = "Distribution of GHI") +
  theme_minimal()

# Time Series
ggplot(data, aes(x = DateTime, y = GHI)) +
  geom_line() +
  labs(title = "GHI over Time") +
  theme_minimal()

#Correlation Matrix
cor_matrix <- cor(
  data[, c("GHI","DHI","DNI","Temperature","Wind.Speed","Relative.Humidity","Hour","Month","DayOfYear")],
  use = "complete.obs"
)

corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8)

#Correlation significance
cor.test(data$Temperature, data$GHI)

#Wind Speed vs GHI
cor.test(data$Wind.Speed, data$GHI)

#Time series
#Convert to time series
ghi_ts <- ts(data$GHI, frequency = 24)
head(ghi_ts)
#Arima Model
library(forecast)
model_arima <- auto.arima(ghi_ts)
summary(model_arima)

forecast_arima <- forecast(model_arima, h = 24)
head(forecast_arima)
plot(forecast_arima)

#splitting data into 80%
#80% of data to train the model
#20% of data to test the model

data <- data[order(data$DateTime), ]
split_point <- floor(0.8 * nrow(data))
train <- data[1:split_point, ]
test  <- data[(split_point+1):nrow(data), ]


# -------------------------------
#Train Linear regression Model
# -------------------------------

model_lm <- lm(GHI ~ DHI + DNI + Temperature + Wind.Speed + Relative.Humidity + Hour + Month + DayOfYear, data=train)
summary(model_lm)

pred_lm <- predict(model_lm, test)


MAE  <- mean(abs(pred_lm - test$GHI))
MAE
RMSE <- sqrt(mean((pred_lm - test$GHI)^2))
RMSE
R2   <- cor(pred_lm, test$GHI)^2
R2

# -------------------------------
#polynomial regression model
# -------------------------------

model_poly <- lm(
  GHI ~ DHI + DNI + 
    poly(Temperature, 2) + 
    poly(Hour, 2) + 
    Wind.Speed + 
    Relative.Humidity + 
    Month+DayOfYear,
  data = train
)



pred_poly <- predict(model_poly, test)

MAE_poly  <- mean(abs(pred_poly - test$GHI))
RMSE_poly <- sqrt(mean((pred_poly - test$GHI)^2))
R2_poly   <- cor(pred_poly, test$GHI)^2

MAE_poly
RMSE_poly
R2_poly

# -------------------------------
#Random forest model
# -------------------------------

install.packages("randomForest")
library(randomForest)
model_rf <- randomForest(
  GHI ~ DHI + DNI + Temperature + Wind.Speed + Relative.Humidity + Hour + Month + DayOfYear,
  data = train,
  ntree = 200,
  importance = TRUE
)
print(model_rf)

pred_rf <- predict(model_rf, test)
#pred_rf

MAE_rf  <- mean(abs(pred_rf - test$GHI))
RMSE_rf <- sqrt(mean((pred_rf - test$GHI)^2))
R2_rf   <- cor(pred_rf, test$GHI)^2

MAE_rf
RMSE_rf
R2_rf

importance(model_rf)
varImpPlot(model_rf)

# -------------------------------
#XG Boost Model
# -------------------------------


train_matrix <- model.matrix(
  GHI ~ DHI + DNI + Temperature + Wind.Speed + Relative.Humidity + Hour + Month + DayOfYear -1,
  data = train
)

test_matrix <- model.matrix(
  GHI ~ DHI + DNI + Temperature + Wind.Speed + Relative.Humidity + Hour + Month + DayOfYear -1,
  data = test
)

train_matrix <- as.matrix(train_matrix)
test_matrix  <- as.matrix(test_matrix)

train_label <- train$GHI
test_label  <- test$GHI

# Train model
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest  <- xgb.DMatrix(data = test_matrix, label = test_label)




# Define parameters
params <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  eval_metric = "rmse"
)


model_xgb <- xgb.train(
  data = dtrain,
  nrounds = 200,
  params = params,
  verbose = 0
)


# Create matrix for full dataset
full_matrix <- model.matrix(
  ~ DHI + DNI + Temperature + Wind.Speed +
    Relative.Humidity + Hour + Month + DayOfYear -1,
  data = data
)

dim(full_matrix)   # should show 35040 rows

dfull <- xgb.DMatrix(data = full_matrix)
data$Predicted_GHI <- predict(model_xgb, dfull)

length(data$Predicted_GHI)   # should show 35040

pred_xgb <- predict(model_xgb, dtest)

MAE_xgb  <- mean(abs(pred_xgb - test_label))
RMSE_xgb <- sqrt(mean((pred_xgb - test_label)^2))
R2_xgb   <- cor(pred_xgb, test_label)^2

MAE_xgb
RMSE_xgb
R2_xgb


# -------------------------------
#comparision of models
# -------------------------------

results <- data.frame(
  Model = c("Linear", "Polynomial", "Random Forest", "XGBoost"),
  MAE   = c(MAE, MAE_poly, MAE_rf, MAE_xgb),
  RMSE  = c(RMSE, RMSE_poly, RMSE_rf, RMSE_xgb),
  R2    = c(R2, R2_poly, R2_rf, R2_xgb)
)

results

#predictions with the test data
efficiency <- 0.18
area <- 6000

test$Predicted_Power_W <- pred_xgb * area * efficiency


test$Predicted_Energy_kWh <- test$Predicted_Power_W / 1000


emission_factor <- 0.82
test$CO2_Saved_kg <- test$Predicted_Energy_kWh * emission_factor


total_energy_kWh <- sum(test$Predicted_Energy_kWh)
total_energy_kWh

total_CO2_saved_kg <- sum(test$CO2_Saved_kg)
total_CO2_saved_kg

total_CO2_saved_tons <- total_CO2_saved_kg / 1000
total_CO2_saved_tons

test$Predicted_GHI_XGB <- pred_xgb


# -------------------------------
#Actual vs Predicted Plot
# -------------------------------
ggplot(test, aes(x = GHI, y = pred_xgb)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Actual vs Predicted GHI (XGBoost)",
       x = "Actual GHI",
       y = "Predicted GHI") +
  theme_minimal()

results

ggplot(results, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Comparison (RMSE)") +
  theme_minimal()

library(Metrics)

rmse(test$GHI, test$Predicted_GHI_XGB)
cor(test$GHI, test$Predicted_GHI_XGB)^2

#To increase print limit
options(max.print = 40000)
data


#predictions with trained data set
# Define constants
# Assumption: Solar panel efficiency ~18% (typical commercial panels)
efficiency <- 0.18

# Assumption: Total panel area = 6000 m² (~1 MW scale solar installation)
area <- 6000
# Assumption: 0.82 kg CO2 per kWh (average fossil-fuel based grid emission factor, India approx)
emission_factor <- 0.82

# Power generation (W)
data$Predicted_Power_W <- data$Predicted_GHI * area * efficiency

# Energy (kWh)
data$Predicted_Energy_kWh <- data$Predicted_Power_W / 1000

# CO2 reduction (kg)
data$CO2_Saved_kg <- data$Predicted_Energy_kWh * emission_factor

# Cumulative CO2 saved
data$Cumulative_CO2_Saved_kg <- cumsum(data$CO2_Saved_kg)


# -------------------------------
# ROUND VALUES (ADD HERE)
# -------------------------------
data$Predicted_GHI <- round(data$Predicted_GHI, 2)
data$Predicted_Power_W <- round(data$Predicted_Power_W, 2)
data$Predicted_Energy_kWh <- round(data$Predicted_Energy_kWh, 2)
data$CO2_Saved_kg <- round(data$CO2_Saved_kg, 2)
data$Cumulative_CO2_Saved_kg <- round(data$Cumulative_CO2_Saved_kg, 2)

# -------------------------------
# EXPORT FINAL FILE
# -------------------------------
write.csv(data, "Solar_Final_Predictions_Output.csv", row.names = FALSE)




