# Replication code for "Pursuing Open-Source Development of Predictive Algorithms: The Case of Criminal Sentencing Algorithms"

# By: Philip Waggoner and Alec MacMillen, University of Chicago

# Load libraries and packages
library(caret)
library(tidyverse)
library(glmnet)
library(mlbench)
library(doParallel)
library(pROC)


# Read in "clean Broward" data
data <- read.csv(file.choose())
data_sub <- data %>% 
  select(two_year_recid, # DV
         race, sex, age, juv_fel_count, juv_misd_count, juv_other_count, priors_count, charge_degree # IVs
  )

#   1.  "Replicate" COMPAS = 0.6537 accuracy
(compas <- mean(data$compas_correct == data$two_year_recid))



#   2.  Improve accuracy with more statistically and computationally appropriate models: LASSO, Ridge, and Elastic-net
#   2.1   LASSO regression
#   2.2   Ridge regression 
#   2.3   Elastic net


###
### Iterative models for accuracy rates and predictions
###


## LASSO LOOP - 1000 iterations
test.lasso <- rep(NA, 1000)

ptm <- proc.time()
for(i in 1:1000){
  training.samples <- data_sub$two_year_recid %>% 
    createDataPartition(p = 0.8, list = FALSE)
  
  train.data  <- data_sub[training.samples, ]
  test.data <- data_sub[-training.samples, ]
  
  # create new vars
  mdlY <- as.factor(as.matrix(train.data$two_year_recid))
  mdlX <- model.matrix(two_year_recid~., train.data)[,-1]
  
  newY <- as.factor(as.matrix(test.data$two_year_recid))
  newX <- model.matrix(two_year_recid~., test.data)[,-1]
  
  # Fit the model
  cv1 <- cv.glmnet(mdlX, mdlY, family = "binomial", nfold = 10, type.measure = "deviance", paralle = TRUE, alpha = 1)
  md1 <- glmnet(mdlX, mdlY, family = "binomial", lambda = cv1$lambda.1se, alpha = 1)
  
  # Make predictions
  lasso.pred <- predict(md1, newX, type = "response")
  lasso.predicted.classes <- ifelse(lasso.pred > 0.5, 1, 0)
  
  # Model accuracy
  lasso.observed.classes <- test.data$two_year_recid
  
    # store accuracy in cell i
  test.lasso[i] <- mean(lasso.predicted.classes == lasso.observed.classes)
}
proc.time() - ptm
lasso_mean <- mean(test.lasso)
lasso_mean # 0.6722642



## RIDGE LOOP - 1000 iterations
test.ridge <- rep(NA, 1000)

ptm <- proc.time()
for(i in 1:1000){
  training.samples <- data_sub$two_year_recid %>% 
    createDataPartition(p = 0.8, list = FALSE)
  
  train.data  <- data_sub[training.samples, ]
  test.data <- data_sub[-training.samples, ]
  
  # create new vars
  mdlY <- as.factor(as.matrix(train.data$two_year_recid))
  mdlX <- model.matrix(two_year_recid~., train.data)[,-1]
  
  newY <- as.factor(as.matrix(test.data$two_year_recid))
  newX <- model.matrix(two_year_recid~., test.data)[,-1]
  
  # Fit the model
  cv2 <- cv.glmnet(mdlX, mdlY, family = "binomial", nfold = 10, type.measure = "deviance", paralle = TRUE, alpha = 0)
  md2 <- glmnet(mdlX, mdlY, family = "binomial", lambda = cv2$lambda.1se, alpha = 0)
  
  # Make predictions
  ridge.pred <- predict(md2, newX, type = "response")
  ridge.predicted.classes <- ifelse(ridge.pred > 0.5, 1, 0)
  
  # Model accuracy
  ridge.observed.classes <- test.data$two_year_recid
  
  # store accuracy in cell i
  test.ridge[i] <- mean(ridge.predicted.classes == ridge.observed.classes)
}
proc.time() - ptm
ridge_mean <- mean(test.ridge)
ridge_mean # 0.6702018



## ELASTIC NET LOOP - 1000 iterations
# parallel backend registration
registerDoParallel(cores = 5)

test.en <- rep(NA, 1000)

ptm <- proc.time()
for(i in 1:1000){
  training.samples <- data_sub$two_year_recid %>% 
    createDataPartition(p = 0.8, list = FALSE)
  
  train.data  <- data_sub[training.samples, ]
  test.data <- data_sub[-training.samples, ]
  
  # create new vars
  mdlY <- as.factor(as.matrix(train.data$two_year_recid))
  mdlX <- model.matrix(two_year_recid~., train.data)[,-1]
  
  newY <- as.factor(as.matrix(test.data$two_year_recid))
  newX <- model.matrix(two_year_recid~., test.data)[,-1]
  
  # search for optimal alpha
  a <- seq(0.1, 0.99, 0.12)
  search <- foreach(i = a, .combine = rbind) %dopar% {
    cv <- cv.glmnet(mdlX, mdlY, family = "binomial", nfold = 10, type.measure = "deviance", paralle = TRUE, alpha = i)
    data.frame(cvm = cv$cvm[cv$lambda == cv$lambda.1se], lambda.1se = cv$lambda.1se, alpha = i)
  }
  
  # Fit the model
  cv3 <- search[search$cvm == min(search$cvm), ]
  md3 <- glmnet(mdlX, mdlY, family = "binomial", lambda = cv3$lambda.1se, alpha = cv3$alpha)
  
  # Make predictions
  en.pred <- predict(md3, newX, type = "response")
  en.predicted.classes <- ifelse(en.pred > 0.5, 1, 0)
  
  # Model accuracy 
  en.observed.classes <- test.data$two_year_recid

  # store accuracy in cell i
  test.en[i] <- mean(en.predicted.classes == en.observed.classes)
}
proc.time() - ptm
en_mean <- mean(test.en) # 0.6742399
en_mean



###
### Inidividual/single shot models for plotting and tables -- RUN CODE STRAIGHT THROUGH FOR SEED VALUE
###


# Split the data into training and test set
set.seed(123)

training.samples <- data_sub$two_year_recid %>% 
  createDataPartition(p = 0.8, list = FALSE)

train.data  <- data_sub[training.samples, ]
test.data <- data_sub[-training.samples, ]

nrow(test.data)

# create new vars
mdlY <- as.factor(as.matrix(train.data$two_year_recid))
mdlX <- model.matrix(two_year_recid~., train.data)[,-1]

newY <- as.factor(as.matrix(test.data$two_year_recid))
newX <- model.matrix(two_year_recid~., test.data)[,-1]



# LASSO WITH ALPHA = 1
cv1 <- cv.glmnet(mdlX, mdlY, family = "binomial", nfold = 10, type.measure = "deviance", paralle = TRUE, alpha = 1)
md1 <- glmnet(mdlX, mdlY, family = "binomial", lambda = cv1$lambda.1se, alpha = 1)
round(coef(md1), 3)

# Make predictions
lasso.pred <- predict(md1, newX, type = "response")
lasso.predicted.classes <- ifelse(lasso.pred > 0.5, 1, 0)

# Model accuracy - 0.6976422
lasso.observed.classes <- test.data$two_year_recid
mean(lasso.predicted.classes == lasso.observed.classes)



# RIDGE WITH ALPHA = 0
cv2 <- cv.glmnet(mdlX, mdlY, family = "binomial", nfold = 10, type.measure = "deviance", paralle = TRUE, alpha = 0)
md2 <- glmnet(mdlX, mdlY, family = "binomial", lambda = cv2$lambda.1se, alpha = 0)
round(coef(md2), 3)

# Make predictions
ridge.pred <- predict(md2, newX, type = "response")
ridge.predicted.classes <- ifelse(ridge.pred > 0.5, 1, 0)

# Model accuracy - 0.6962552
ridge.observed.classes <- test.data$two_year_recid
mean(ridge.predicted.classes == ridge.observed.classes)



# ELASTIC NET WITH 0 < ALPHA < 1
a <- seq(0.1, 0.9, 0.05)
search <- foreach(i = a, .combine = rbind) %dopar% {
  cv <- cv.glmnet(mdlX, mdlY, family = "binomial", nfold = 10, type.measure = "deviance", paralle = TRUE, alpha = i)
  data.frame(cvm = cv$cvm[cv$lambda == cv$lambda.1se], lambda.1se = cv$lambda.1se, alpha = i)
}
(cv3 <- search[search$cvm == min(search$cvm), ])
md3 <- glmnet(mdlX, mdlY, family = "binomial", lambda = cv3$lambda.1se, alpha = cv3$alpha)
cv3.1 <- cv.glmnet(mdlX, mdlY, family = "binomial", nfold = 10, type.measure = "deviance", paralle = TRUE, alpha = cv3$alpha)
round(coef(md3), 3)

# Make predictions
en.pred <- predict(md3, newX, type = "response")
en.predicted.classes <- ifelse(en.pred > 0.5, 1, 0)

# Model accuracy - 0.6997226
en.observed.classes <- test.data$two_year_recid
mean(en.predicted.classes == en.observed.classes)



## CONFUSION MATRICES 
table(lasso.predicted.classes, lasso.observed.classes)
table(ridge.predicted.classes, ridge.observed.classes)
table(en.predicted.classes, en.observed.classes)



## PLOTS
library(foreach)
library(ggfortify)
library(ggpubr)

attach(data)

lassotune <- autoplot(cv1$glmnet.fit, "lambda", label = TRUE, main = "LASSO (alpha = 1)") + 
  theme(legend.position="right") + 
  scale_colour_discrete(name = "Variables", 
                        labels = c("Age", "Charge degree", "Juvenile felony count",
                                   "Juvenile misdemeanor count", "Juvenile other count",
                                   "Priors count", "Race", "Sex")) + 
  theme(legend.title = element_text(size=10)) + 
  theme(legend.text = element_text(size = 8)) + 
  theme_bw() +
  geom_vline(data = NULL, 
             xintercept = log(cv1$lambda.1se), 
             na.rm = FALSE, show.legend = TRUE)

ridgetune <- autoplot(cv2$glmnet.fit, "lambda", label = TRUE, main = "Ridge (alpha = 0)") + 
  theme(legend.position="right") + 
  scale_colour_discrete(name = "Variables", 
                        labels = c("Age", "Charge degree", "Juvenile felony count",
                                   "Juvenile misdemeanor count", "Juvenile other count",
                                   "Priors count", "Race", "Sex")) + 
  theme(legend.title = element_text(size=10)) + 
  theme(legend.text = element_text(size = 8)) + 
  theme_bw() +
  geom_vline(data = NULL, 
             xintercept = log(cv2$lambda.1se), 
             na.rm = FALSE, show.legend = TRUE)

elasticnettune <- autoplot(cv3.1$glmnet.fit, "lambda", label = TRUE, main = "Elastic Net (alpha = 0.3)") + 
  theme(legend.position="right") +
  scale_colour_discrete(name = "Variables", 
                        labels = c("Age", "Charge degree", "Juvenile felony count",
                                   "Juvenile misdemeanor count", "Juvenile other count",
                                   "Priors count", "Race", "Sex")) + 
  theme(legend.title = element_text(size=10)) + 
  theme(legend.text = element_text(size = 8)) + 
  theme_bw() +
  geom_vline(data = NULL, 
             xintercept = log(cv3$lambda.1se), 
             na.rm = FALSE, 
             show.legend = TRUE)

main_figures <- ggarrange(lassotune, ridgetune, elasticnettune,
                          ncol = 1, nrow = 3)
main_figures


# ERROR BAR PLOTS
lassoerror <- autoplot(cv1, label = TRUE, 
                       main = "Search for Lambda: LASSO") +
  labs(subtitle = "At a range of values for lambda",
       caption = "Vertical dashed line shows selected value of lambda (1 SD above minimum)") +
  theme(plot.caption = element_text(hjust = 0, face = "italic"))+
  theme_bw()

ridgeerror <- autoplot(cv2, label = TRUE, 
                       main = "Search for Lambda: Ridge") +
  labs(subtitle = "At a range of values for lambda",
       caption = "Vertical dashed line shows selected value of lambda (1 SD above minimum)") +
  theme(plot.caption = element_text(hjust = 0, face = "italic"))+
  theme_bw()

elasticneterror <- autoplot(cv3.1, label = TRUE, 
                            main = "Search for Lambda: Elastic Net") +
  labs(subtitle = "At a range of values for lambda",
       caption = "Vertical dashed line shows value of lambda (1 SD above minimum)") +
  theme(plot.caption = element_text(hjust = 0, face = "italic")) +
  theme_bw()

lambda_figures <- ggarrange(lassoerror, ridgeerror, elasticneterror,
                            ncol = 1, nrow = 3)
lambda_figures





## ROC
# Calculate predicted responses
pred1 <- as.numeric(predict(md1, newX, type = "response")) # lasso
pred2 <- as.numeric(predict(md2, newX, type = "response")) # ridge
pred3 <- as.numeric(predict(md3, newX, type = "response")) # elastic-net

## VIZ
plot.roc(newY, pred1, col="red", lwd = 1)
plot.roc(newY, pred2, col="blue", lwd = 1, add=TRUE)
plot.roc(newY, pred3, col="green", lwd = 1, add=TRUE)
grid()
legend(0.35, 0.3, legend=c("LASSO", "Ridge", "Elastic-net"),
       col=c("red", "blue", "green"), lwd = 1, cex = 1)

## NUMERICAL (AUC)
(lasso_roc <- roc(newY, as.numeric(predict(md1, newX, type = "response"))))
(ridge_roc <- roc(newY, as.numeric(predict(md2, newX, type = "response"))))
(en_roc <- roc(newY, as.numeric(predict(md3, newX, type = "response"))))








## APPENDIX: 
#   Replicate Dressel (Logit 1000 times)

# Train logit on 80% and test on reamining 20% 1000 times
test.logit <- rep(NA, 1000)

for(i in 1:1000){
  training.samples <- data_sub$two_year_recid %>% 
    createDataPartition(p = 0.8, list = FALSE)
  
  train.data <- data_sub[training.samples, ]
  test.data <- data_sub[-training.samples, ]
  
  # Fit the model
  full.model <- glm(two_year_recid ~., data = train.data, family = binomial)
  
  # Make predictions
  p <- full.model %>% 
    predict(test.data, type = "response")
  
  predicted.classes <- ifelse(p > 0.5, 1, 0)
  
  # Model accuracy
  observed.classes <- test.data$two_year_recid
  
  # store accuracy in cell i
  test.logit[i] <- mean(predicted.classes == observed.classes)
}
logit_mean <- mean(test.logit) # 0.6758232
logit_mean



# MSE
library(stargazer)

y <- train.data$two_year_recid

for (i in 0:10) {
  assign(paste("fit", i, sep=""), cv.glmnet(mdlX, y, type.measure="mse", 
                                            alpha=i/10,family="gaussian"))
}

yhat0 <- predict(fit0, s=fit0$lambda.1se, newx=mdlX)
yhat1 <- predict(fit1, s=fit1$lambda.1se, newx=mdlX)
yhat2 <- predict(fit2, s=fit2$lambda.1se, newx=mdlX)
yhat3 <- predict(fit3, s=fit3$lambda.1se, newx=mdlX)
yhat4 <- predict(fit4, s=fit4$lambda.1se, newx=mdlX)
yhat5 <- predict(fit5, s=fit5$lambda.1se, newx=mdlX)
yhat6 <- predict(fit6, s=fit6$lambda.1se, newx=mdlX)
yhat7 <- predict(fit7, s=fit7$lambda.1se, newx=mdlX)
yhat8 <- predict(fit8, s=fit8$lambda.1se, newx=mdlX)
yhat9 <- predict(fit9, s=fit9$lambda.1se, newx=mdlX)
yhat10 <- predict(fit10, s=fit10$lambda.1se, newx=mdlX)

mse0 <- mean((y - yhat0)^2)
mse1 <- mean((y - yhat1)^2)
mse2 <- mean((y - yhat2)^2)
mse3 <- mean((y - yhat3)^2)
mse4 <- mean((y - yhat4)^2)
mse5 <- mean((y - yhat5)^2)
mse6 <- mean((y - yhat6)^2)
mse7 <- mean((y - yhat7)^2)
mse8 <- mean((y - yhat8)^2)
mse9 <- mean((y - yhat9)^2)
mse10 <- mean((y - yhat10)^2)

mses <- matrix(c(mse0, mse1, mse2, mse3, mse4, 
                 mse5, mse6, mse7, mse8, mse9,
                 mse10), ncol = 1, byrow = FALSE)
colnames(mses) <- c("MSE")
rownames(mses) <- c("Alpha=0 (Ridge)","Alpha=0.1", "Alpha=0.2", "Alpha=0.3 (Elastic Net)", "Alpha=0.4", "Alpha=0.5", "Alpha=0.6", 
                    "Alpha=0.7", "Alpha=0.8", "Alpha=0.9", "Alpha=1 (LASSO)")

MSEtab <- as.table(mses); MSEtab

stargazer(MSEtab, type = "html", title = "Mean Squared Errors (MSEs) for levels of Alpha", 
          colnames = FALSE, rownames = FALSE, digits = 3)

#
