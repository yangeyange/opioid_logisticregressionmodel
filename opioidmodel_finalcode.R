
######################################################################################

#   Project: Summer Institute in Biostatistics Capstone Project: Predictive Modeling of Opioid Usage
#   Program Name: opioidmodel_finalcode.R
#   Author: Eileen Yang
#   Date created: 2019-07-14
#   Purpose: building and visualizing a logistic regression model to predict opioid usage based on demographic, personality, and drug use data
#   Revision history:
#   DateAuthorRef(*)Revision
#   searchable reference phrase: ### [*]###

######################################################################################
#   Notes: 
#   all data used for this project is from this site: https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29
#   a cleaned/recoded dataset of all relevant variables was used for this model ("opioidmodeldataset.csv")

######################################################################################

#install.packages("leaps")
#install.packages("dummies")
#install.packages("car")
library(leaps)
library(dummies)
library(caret)
library(car)
library(pROC)

#--reading in "opioidmodeldataset.csv" to build the model with
drug <- read.csv("/Users/eileenyang/Desktop/opioidmodeldataset.csv") #Replace with your custom filepath to the csv file included in the zip file
names(drug)

#--making dummy variables from the categorical variables
aged <- dummy(drug$Age, sep = ".")
genderd <- dummy(drug$Gender, sep = ".")
educationd <- dummy(drug$Education, sep = ".")
countryd <- dummy(drug$Country, sep = ".")
ethnicityd <- dummy(drug$Ethnicity, sep = ".")


#--combining dummy variables into a new dataset dummydrug, which is going to be all possible covariates
dummydrug <- cbind(aged, genderd, educationd, countryd, ethnicityd, drug$Nscore2, drug$Ascore2, drug$Cscore2, drug$Oscore2, drug$Escore2,
                   drug$Alcohol2, drug$Cannabis2, drug$Nicotine2, drug$Opioid)

#--making dummydrug into a data.frame
dummydrug<-data.frame(dummydrug)

#--renaming dummydrug variables 
dummydrug$Nscored<-dummydrug$V32
dummydrug$Ascored<-dummydrug$V33
dummydrug$Cscored<-dummydrug$V34
dummydrug$Oscored<-dummydrug$V35
dummydrug$Escored<-dummydrug$V36
dummydrug$alcd<-dummydrug$V37
dummydrug$cand<-dummydrug$V38
dummydrug$nicd<-dummydrug$V39
dummydrug$opioid<-dummydrug$V40
dummydrug

#--recode two-factor variables into 1s and 0s
recodeopioid2<-recode(dummydrug$opioid, "1='A'; 2='B'")
recodeopioid3<-recode(recodeopioid2,"'A'=0;'B'=1")

recodealcd2<-recode(dummydrug$alcd, "1='A'; 2='B'")
recodealcd3<-recode(recodealcd2,"'A'=0;'B'=1")

recodecand2<-recode(dummydrug$cand, "1='A'; 2='B'")
recodecand3<-recode(recodecand2,"'A'=0;'B'=1")

recodenicd2<-recode(dummydrug$nicd, "1='A'; 2='B'")
recodenicd3<-recode(recodenicd2,"'A'=0;'B'=1")


#--recombine the renamed variables with original dummydrug dataset
dummydrug2 <- cbind(dummydrug,recodeopioid3,recodealcd3,recodecand3,recodenicd3)
names(dummydrug2)

#--removing old unnamed dummydrug variables and variables with undesirable names
dummydrug2=within(dummydrug2, rm(V32, V33, V34, V35, V36, V37, V38, V39, V40, alcd, nicd, cand, opioid))
names(dummydrug2)

#--renaming recodenicd,alcd,cand,opioid variables
alcd<-dummydrug2$recodealcd3
nicd<-dummydrug2$recodenicd3
cand<-dummydrug2$recodecand3
opioid<-dummydrug2$recodeopioid3

#--final recombination to get the dataset we want to use
dummydrug3 <- cbind(dummydrug2,alcd,nicd,cand,opioid)
dummydrug3 <- within(dummydrug3, rm(recodealcd3, recodenicd3, recodecand3, recodeopioid3))

#rename to dummydrug for simplicity
dummydrug<-dummydrug3
names(dummydrug)

# ------------------------------------------------------------------------------------
#--creating a training and testing dataset from the overall dataset
set.seed(123)
dummytrain.index <- createDataPartition(dummydrug$opioid, p=.7, list = FALSE)
dummytrain <- dummydrug[dummytrain.index, ]
dummytest <- dummydrug[-dummytrain.index, ]

# ------------------------------------------------------------------------------------
#BEGIN CREATION OF MODEL
#--selecting the covariates with stepwise selection
selection <- regsubsets(dummytrain$opioid~., data = dummytrain, method = "seqrep",nbest=1)
summary(selection)

#(removed dummydrug$Ethnicity.White because not significant--don't worry about this it's just scratch)

#--binary logistic regression--fitting the model
glm.fit_step1 <- glm(opioid ~  Country.USA, 
                     data = dummytrain, family = binomial(link=logit))
summary(glm.fit_step1) #AIC=1201.7

glm.fit_step2 <- glm(opioid ~ Country.USA 
                     + cand, 
                     data = dummytrain, family = binomial(link=logit))
summary(glm.fit_step2) #AIC=1152

glm.fit_step3 <- glm(opioid ~ Country.USA 
                     + cand + Nscored, 
                     data = dummytrain, family = binomial(link=logit))
summary(glm.fit_step3) #AIC= 1142.1

glm.fit_step4 <- glm(opioid ~ Country.USA 
                     + nicd + Nscored + Gender.Female, 
                     data = dummytrain, family = binomial(link=logit))
summary(glm.fit_step4) #AIC= 1141.7

glm.fit_step5 <- glm(opioid ~ Gender.Female
                     + Country.USA + Country.UK
                     + Nscored + nicd, 
                     data = dummytrain, family = binomial(link=logit))
summary(glm.fit_step5) #AIC= 1130.1

glm.fit_step6 <- glm(opioid ~ Gender.Female + Education.Left.school.before.16.years
                     + Country.UK + Country.USA
                     + Nscored  + nicd, 
                     data = dummytrain, family = binomial(link=logit))
summary(glm.fit_step6) #AIC= 1125.4

glm.fit_step7 <- glm(opioid ~ Gender.Female + Education.Left.school.before.16.years
                     + Country.UK + Country.USA
                     + Nscored + Oscored
                     + nicd, 
                     data = dummytrain, family = binomial(link=logit))
summary(glm.fit_step7) #AIC= 1118.8

glm.fit_step8 <- glm(opioid ~ Gender.Female + Education.Left.school.before.16.years
                     + Country.UK + Country.USA
                     + Ethnicity.White + Nscored + Oscored
                     + nicd, 
                     data = dummytrain, family = binomial(link=logit))
summary(glm.fit_step8) #AIC= 1113.3

glm.fit_step9 <- glm(opioid ~ Gender.Female + Education.Left.school.before.16.years
                     + Country.UK + Country.USA
                     + Ethnicity.White + Nscored + Oscored + Ascored
                     + nicd, 
                     data = dummytrain, family = binomial(link=logit))
summary(glm.fit_step9) #AIC= 1109.2

# --- To get estimates on odds scale, must exponentiate
exp(glm.fit_step9$coef)
exp(confint.default(glm.fit_step9))

selectionwhole <- regsubsets(opioid~., data = dummydrug, method = "seqrep",nbest=1, nvmax=9)
summary(selectionwhole)

glm.fit_whole <- glm(opioid ~ Gender.Male + Education.Left.school.before.16.years + Education.Masters.degree
                     + Country.USA + Country.UK
                     + Nscored + Ascored + Oscored
                     + cand + nicd, 
                     data = dummydrug, family = binomial(link=logit))
summary(glm.fit_whole) #AIC= 1584.7

# ------------------------------------------------------------------------------------
# - Get predicted probabilities of Opioid

trainprobs <- predict(glm.fit_whole, newdata=dummytrain, type='response')
trainprobs
trainprobs.classes <- ifelse(trainprobs > 0.5, 1, 0)
trainprobs.classes

trainresults <- dummytrain$opioid == trainprobs.classes
table(trainresults)
table(dummytrain$opioid)
#1051/1314=0.7998478

testprobs <- predict(glm.fit_whole, newdata=dummytest, type='response')
testprobs
testprobs.classes <- ifelse(testprobs > 0.5, 1, 0)
testprobs.classes

testresults <- dummytest$opioid == testprobs.classes
table(testresults)
table(dummytest$opioid)
#448/563=0.7957371

totalprobs <- predict(glm.fit_whole, newdata=dummydrug, type='response')
totalprobs
totalprobs.classes <- ifelse(totalprobs > 0.5, 1, 0)
totalprobs.classes

totalresults <- dummydrug$opioid == totalprobs.classes
table(totalresults)
table(dummydrug$opioid)
#1499/1877=0.7986148

# ------------------------------------------------------------------------------------
# - Estimate calibration-in-the-large: values should be similar

mean(trainprobs)        # Mean P(Y=1|X)
mean(dummytrain$opioid)          # Observed P(Y=1)

mean(testprobs)
mean(dummytest$opioid)

# ------------------------------------------------------------------------------------
# CREATE CALIBRATION PLOT

# --- Summarize predicted probabilities
summary(trainprobs)
hist(trainprobs)

summary(testprobs)
hist(testprobs)

# --- Compute deciles of predicted probabilities
traindec <- quantile(trainprobs,            # Variable to be summarized
                     probs=seq(0,1,by=0.1), # Vector of percentile values 
                     type=3)                # Use same algorithm as SAS
traindec 

testdec <- quantile(testprobs,            # Variable to be summarized
                    probs=seq(0,1,by=0.1), # Vector of percentile values 
                    type=3)   
testdec

# --- Create decile group variable 
traindec_grp <- cut(trainprobs,         # Predicted probabilities (var to group)
                    breaks = 10,       # Cut points (group intervals)
                    include.lowest = T, # Include smallest value 
                    labels = 1:10)      # Labels for groups
traindec_grp

testdec_grp <- cut(testprobs,         # Predicted probabilities (var to group)
                   breaks = 10,       # Cut points (group intervals)
                   include.lowest = T, # Include smallest value 
                   labels = 1:10)      # Labels for groups
testdec_grp

# ----- Check that decile groups created correctly 
table(traindec_grp)                     # Number of observations in each decile group
prop.table(table(traindec_grp))         # Proportion of observations in each decile group

table(testdec_grp)                     # Number of observations in each decile group
prop.table(table(testdec_grp))        # Proportion of observations in each decile group

# --- Compute mean predicted probability and event rate by decile group
trainagg <- aggregate(cbind(opioid,trainprobs) ~ traindec_grp, # Aggregate A,B by C
                      data = dummytrain,                  # From data set
                      FUN = 'mean')                # Using this summary function 
trainagg

testagg <- aggregate(cbind(opioid,testprobs) ~ testdec_grp, # Aggregate A,B by C
                     data = dummytest,                  # From data set
                     FUN = 'mean')                # Using this summary function 
testagg

# ----- Check computations for decile group 5
#mean(drug$Opioid[drug$dec_grp2 == 5])
#mean(drug$p.hats2[drug$dec_grp2 == 5])

# --- Create calibration plot 
plot(trainagg$trainprobs,                         # x-coor = mean pred prob in dec group
     trainagg$opioid,                            # y-coor = obs event rate in dec group
     main = 'Calibration Plot',          # Add main title 
     ylab = 'Observed Event Rate',       # Add y-axis label
     xlab = 'Predicted Probabilities',   # Add x-axis label
     pch = 19,                           # Plotting character = solid dot
     col = 'orangered',                  # Color of plotting character 
     cex = 2)                            # Size of plotting character (base = 1)

plot(testagg$testprobs,                         # x-coor = mean pred prob in dec group
     testagg$opioid,                            # y-coor = obs event rate in dec group
     main = 'Calibration Plot',          # Add main title 
     ylab = 'Observed Event Rate',       # Add y-axis label
     xlab = 'Predicted Probabilities',   # Add x-axis label
     pch = 19,                           # Plotting character = solid dot
     col = 'orangered',                  # Color of plotting character 
     cex = 2)                            # Size of plotting character (base = 1)

# Add identity line
abline(a = 0,                            # a = intercept 
       b = 1)                            # b = slope

# Add fitted regression line
traincal.fit <- lm(opioid ~ trainprobs, data = trainagg) # Fit linear model to plot data (agg set, not raw set)
abline(traincal.fit,                         # Using intercept and slope from linear model fit
       lty = 2,                         # Dashed line
       col = 'royalblue',               # Color of plotting line
       lwd = 3)                         # Thickness of plotting line (base = 1)

testcal.fit <- lm(opioid ~ testprobs, data = testagg)
abline(testcal.fit,                         # Using intercept and slope from linear model fit
       lty = 2,                         # Dashed line
       col = 'royalblue',               # Color of plotting line
       lwd = 3)                         # Thickness of plotting line (base = 1)



summary(traincal.fit)                        # Compute calibration intercept and slope
confint(traincal.fit)                        # Compute  95% confidence intervals

summary(testcal.fit)                        # Compute calibration intercept and slope
confint(testcal.fit)                        # Compute  95% confidence intervals

# ------------------------------------------------------------------------------------
# - Plot density of predicted probabilities by event status - we want a small overlap between the two curves

ggplot(dummytrain,                           # Data set to pull variables from 
       aes(trainprobs,                    # Variable density to plot 
           fill=as.factor(opioid))) +    # Variable to stratify by (has to be a factor)
  geom_density(alpha = 0.2) +           # Transparency of plotting colors 
  scale_fill_manual(                    # Set plotting colors
    values=c("orangered", "royalblue"))

ggplot(dummytest,                           # Data set to pull variables from 
       aes(testprobs,                    # Variable density to plot 
           fill=as.factor(opioid))) +    # Variable to stratify by (has to be a factor)
  geom_density(alpha = 0.2) +           # Transparency of plotting colors 
  scale_fill_manual(                    # Set plotting colors
    values=c("orangered", "royalblue"))

# ------------------------------------------------------------------------------------
# - Create ROC curve plot and compute AUC 

trainroc.mod <- roc(dummytrain$opioid,       # Observed outcome variable (Y) 
                    trainprobs)    # Predicted probabilities (P.hat(Y=1|X))
plot.roc(trainroc.mod)             # Plot ROC curve
auc(trainroc.mod)                  # Compute AUC: the closer to 1 the better 
ci.auc(trainroc.mod)               # Compute 95% confidence interval for AUC
#AUC: 0.8187; 95% CI: 0.808-0.8565 (DeLong)

testroc.mod <- roc(dummytest$opioid,       # Observed outcome variable (Y) 
                   testprobs)    # Predicted probabilities (P.hat(Y=1|X))
plot.roc(testroc.mod)             # Plot ROC curve
auc(testroc.mod)                  # Compute AUC: the closer to 1 the better 
ci.auc(testroc.mod)              #Compute 95% confidence interval for AUC
#AUC: 0.8238; 95% CI: 0.7874-0.8602 (DeLong)

# ------------------------------------------------------------------------------------
# End of Program
