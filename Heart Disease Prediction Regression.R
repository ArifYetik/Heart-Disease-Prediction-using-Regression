---
  title: "Heart Disease Prediction using Regression"
author: "Arif Yetik"
date: "1/31/2022"
output:
  word_document: default
---
  
  
  ```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```


#### Abstract

Cardiovascular diseases and chronic respiratory diseases are a global threat. Due to such high deaths, there is needed to tackle the reasons behind these diseases. Most Coronary Heart Diseases can be prevented by addressing behavioral risk factors. It is important to detect cardiovascular disease as early as possible so that management with counseling and medicines can begin. 
Recent advances in the field of artificial intelligence and softwares have led to the emergence of expert systems for medical applications. Moreover, in the last few decades computational tools have been designed to improve the experiences and abilities of expertss for making decisions about their patients. 

In this study we used R program with some packages. Research intends to pinpoint the most relevant/risk factors of heart disease as well as predict the overall risk using regression models. We will use our data to create a model which tries predict if a patient has this disease or not.

The dataset is publically available on the Kaggle website. The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD).The dataset provides the patients' information. It includes over 4,000 records and 15 attributes.




##### Keywords 
Coronary Heart Disease (CHD), Heart Disease Prediction, Logistic Regression, R Program, Artificial Intelligence



### 1. Introduction

#### 1.1.   Motivation 
Heart Diseases are the leading cause of death globally. World Health Organization has estimated 17.9 million people died from Heart Diseases in 2019, representing 32% of all global deaths. Of these deaths, 85% were due to heart attack and stroke. About 659,000 people in the United States die from heart disease each year, that's 1 in every 4 deaths.
While many people with heart disease have symptoms such as chest pain and fatigue, as many as 50% have no symptoms until a heart attack occurs. According to the American heart association (AHA), CHD is the leading killer of American men and women, responsible for more than one of every five deaths in 2001 (http://www.americanheart.org, 2008). Many statistics show CHD as the leading cause of premature and permanent disability among American workers.



#### 1.2.   Objectives 

In this data set, my ultimate goal is to find out the factors that will increase the chances of heart failure or heart disease and create a model that can accurately (hopefully) predict whether a person has the risk of having heart failure or a heart disease based on the given variables. Since this is a classification challenge (high risk or low risk), I will be experimenting with Simple Linear Regressions and Generalized Linear Regression. We will predict 10-year risk of Coronary Heart Diseases.


#### 1.3.   Significance

The early prognosis of heart diseases can aid in making decisions on lifestyle changes in high risk patients and in turn reduce the complications. This research intends to pinpoint the most relevant/risk factors of heart disease as well as predict the overall risk using regressions.
Use of risk prediction model to estimate total heart disease risk is a major advance on the  older research of identifying and treating individual risk factors, such as gender, age, smoking, BPMeds, stroke, sysBP, and glucose.


### 2. Relevant Literature

Heart disease also called as coronary heart disease (CHD), is a deposition of fats inside the tubes which supplies blood to the heart muscles. Heart disease actually starts as early as 18 years and patients only came to know about heart disease when the blockage exceeds about 70 %. Theses blockages develop over the years and lead to rupture of the membrane covering the blockage due to pressure increases. If the chemicals released by broken membrane mixed with blood and lead to a blood clot, results to heart disease. The reasons which increase blockage are called as risk factors

Researchers expressed their efforts in finding the best model for predicting cardiovascular disease. In the meantime, various studies give only a glimpse into predicting heart disease using machine learning techniques and fuzzy logic systems.

Logistic regression(LR) is a generalized linear regression model. Therefore, it is similar with multiple linear regression in many aspects. Usually, LR is used for binary classification problems where the predictive variable???????[0,1], 0 is negative class and 1 is positive class. But it can also be used for multi-classification. Logistic regression is mainly used to for prediction and also calculating the probability of success.

Regression analysis explores the relationship between a quantitative response variable and explanatory variables. Regression model has two main objectives. Firstly, identify the statically significant relationship between these two variables. Secondly, forecast the new observations on response variable based on explanatory variables. In short, these variable are of two types i.e. dependent variable and independent variable. The dependent variable is the one whose value is required to be forecasted (i.e. vital signs values in our case) whereas the independent variable is used to explain dependent variable as input. 


### 3. Methodology

#### 3.1.The Data 

In late 1940s, U.S. Government set out to better understand cardiovascular disease (CVD). They track large cohort of initially health patients over time
City of Framingham, MA selected as site for study. Patients aged 30-59 enrolled. Patients given questionnaire and exams every 2 years. Exams and questions expanded over time. We will build models using the Framingham data to predict and prevent Coronary Heart Diseases.

This data was downloaded from kaggle. We can describe each variables as follows.

1- Sex: male or female(Nominal)
2- Age: Age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)
3- Current Smoker: whether or not the patient is a current smoker (Nominal)
4-  Cigs Per Day: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can have any number of cigarettes, even half a cigarette.)
5- BP Meds: whether or not the patient was on blood pressure medication (Nominal)
6- Prevalent Stroke: whether or not the patient had previously had a stroke (Nominal)
7- Prevalent Hyp: whether or not the patient was hypertensive (Nominal)
8- Diabetes: whether or not the patient had diabetes (Nominal)
9- Tot Chol: total cholesterol level (Continuous)
10- Sys BP: systolic blood pressure (Continuous)
11- Dia BP: diastolic blood pressure (Continuous)
12- BMI: Body Mass Index (Continuous)
13- Heart Rate: heart rate (Continuous - In medical research, variables such as heart rate though in fact discrete, yet are considered continuous because of large number of possible values.)
14- Glucose: glucose level (Continuous)

Predict variable (desired target)
15- 10 year risk of coronary heart disease CHD (binary: "1", means "Yes", "0" means "No")


In this study we used R program and some packages.

```{r}
library(tidyverse)
library(dplyr)
library(cowplot)
library(pROC)
library(caTools)
library(vtree)
library(corrgram)
library(caret)
```


##### 3.1.1 Data Cleaning and Exploratory Data Analysis

```{r}
heart_disease <- read.csv("framingham.csv")
View(heart_disease)
str(heart_disease)
```


```{r}
summary(heart_disease)
```


First, we can remove Duplicate Observations and Clean Null Observations.
```{r}
heart_disease <- heart_disease %>% distinct()
colSums(is.na(heart_disease))
heart_disease <- na.omit(heart_disease)
head(heart_disease)
```

Convert binary variables tonumeric for better visualization
```{r}
heart_disease$currentSmoker <- as.numeric(as.character(heart_disease$currentSmoker))
heart_disease$prevalentHyp <- as.numeric(as.character(heart_disease$prevalentHyp))
heart_disease$diabetes <- as.numeric(as.character(heart_disease$diabetes))
heart_disease$TenYearCHD <- as.numeric(as.character(heart_disease$TenYearCHD ))
```

Data structure:
  demographic risk factors: male, age, and education
behavioral risk factors : currentSmoker and cigsPerDay
medical history factors : BPmeds, prevalentStroke, prevalentHyp, diabetes
physical exam risk      : totChol, sysBP, diaBP, BMI, heartRate, glucose

dependent/outcome variable: CHD in 10 years


##### 3.1.2. Data Visualization

```{r}
vtree(heart_disease, c("TenYearCHD", "male"), 
      fillcolor = c(TenYearCHD = "#e7d4e8", male = "#99d8c9"),
      horiz = TRUE)
```
In this plot, there are 3656 patients and 557 of them had heart disease in ten years. Among who had heart disease, 307 (55%) of them were male and 250 (45%) were female. More males than females have CHD. 


```{r}
vtree(heart_disease, c("TenYearCHD", "diabetes"), 
      fillcolor = c(TenYearCHD = "#e7d4e8", male = "#99d8c9"),
      horiz = TRUE)
```
In this plot, among 557 patients had heart disease,  just 35 (6%) patients have diabetes. More non-diabetics have CHD compared to diabetics.



Now, we can look into our variables distributions and efeect of heart disease.

```{r}
plot_1<- ggplot(heart_disease, aes(age)) + geom_density(fill = "blue") + labs(x="",title = "Age Distrubition") + theme_minimal()

plot_12 <- ggplot(data = heart_disease, mapping = aes(x = as.factor(TenYearCHD), y = age, color = TenYearCHD, fill = TenYearCHD)) +
  geom_boxplot() + labs(x="",title = "Heart Diseas ~ Age")
plot_2 <- ggplot(heart_disease, aes(totChol)) + geom_density(fill = "blue") + labs(x="",title = "Cholesterol Distrubition") + theme_minimal()

plot_22 <- ggplot(data = heart_disease, mapping = aes(x = as.factor(TenYearCHD), y = totChol, color = TenYearCHD)) +
  geom_boxplot() + labs(x="",title = "Heart Diseas ~ Total Chol") 

plot_grid(plot_1, plot_12, plot_2, plot_22) 
```
In this plot, many patients are 35-55 years old. Among them, who are above 55 years old had more heart failure. Many patients have average 250 cholesterol. Total Cholesterol has effect on heart disease. 



```{r}
plot_3 <- ggplot(heart_disease, aes(sysBP)) + geom_density(fill = "blue") + labs(x="",title = "sysB Distrubition") + theme_minimal()
plot_33 <- ggplot(data = heart_disease, mapping = aes(x = as.factor(TenYearCHD), y = sysBP, fill = TenYearCHD)) +
  geom_boxplot() + labs(x="",title = "Heart Disease ~ sysBP")
plot_4 <- ggplot(heart_disease, aes(diaBP)) + geom_density(fill = "blue") + labs(x="", title = "diaBP Distrubition") + theme_minimal()
plot_44 <- ggplot(data = heart_disease, mapping = aes(x = as.factor(TenYearCHD), y = diaBP, color = TenYearCHD)) +
  geom_boxplot() + labs(x="", title = "Heart Disease ~ diaBP") 
plot_grid(plot_3, plot_33, plot_4, plot_44) 
```
In this plot, many patients have average 140 systolic blood pressure. People with CHD have higher mean systolic blood pressures. People with CHD have higher mean diastolic blood pressures.




```{r}
plot_5 <- ggplot(heart_disease, aes(diaBP)) + geom_density(fill = "blue") + labs(x="", title = "BMI Distrubition") + theme_minimal()
plot_55 <- ggplot(data = heart_disease, mapping = aes(x = as.factor(TenYearCHD), y = BMI, fill = TenYearCHD)) +
  geom_boxplot() +labs(x="", title = "Heart Disease ~ BMI")
plot_6 <- ggplot(heart_disease, aes(heartRate)) + geom_density(fill = "blue") + labs(x="", title = "Heart Rate Distrubition") + theme_minimal()
plot_66 <- ggplot(data = heart_disease, mapping = aes(x = as.factor(TenYearCHD), y = heartRate, fill = TenYearCHD)) +
  geom_boxplot() +labs(x="", title = "Heart Disease ~  HeartRate")
plot_grid(plot_5, plot_55, plot_6, plot_66) 
```
People with CHD have a higher mean BMI. People with CHD have very similar mean heart rates as people without CHD.


```{r}
plot_7 <- ggplot(heart_disease, aes(glucose)) + geom_density(fill = "blue") + labs(x="", title = "Glucose Distrubition") + theme_minimal()
plot_77 <- ggplot(data = heart_disease, mapping = aes(x = as.factor(TenYearCHD), y = glucose, fill = TenYearCHD)) +
  geom_boxplot() +labs(x="", title = "Heart Disease ~ Glucose")
plot_8 <- ggplot(heart_disease, aes(cigsPerDay)) + geom_density(fill = "blue") + labs(x="", title = "Cigarete PerDay Distrubition") + theme_minimal()
plot_88 <- ggplot(data = heart_disease, mapping = aes(x = as.factor(TenYearCHD), y = cigsPerDay, fill = TenYearCHD)) +
  geom_boxplot() +labs(x="", title = "Heart Disease ~ Cigarete PerDay") 
plot_grid(plot_7, plot_77, plot_8, plot_88) 
```
In this plot, people with CHD and without CHD have very similar mean glucose levels



##### 3.2.1 Linear Regressions

We can compare the HeartDisease output with all the numeric variables within our data set and see whether we can find any correlations by using a correlogram.

We can check correlation between variables.
```{r}
heart_disease %>% corrgram(order=TRUE, upper.panel=panel.cor)

```

Based on the given data, we can make regression model. So, we will make prediction on the target variable coronary heart disease CHD. 

```{r}
model <- lm(TenYearCHD ~ ., data = heart_disease)
summary(model)
```
Adjusted R-squared is too low. We can remodel with significant factors of male, age, sysBP, and glucose. cigsPerDay is almost significant factor. 
We can make new model.

```{r}
re_model2 <- lm(TenYearCHD ~ male + age + sysBP+glucose, data = heart_disease)
summary(re_model2)
```

The new model only achieved 0.09 adjusted R-squared which is same as previous model. Adjusted R-squared indicate a weak fit. 

For this challenge, since heart disease is a classification problem, We will be using Generalized Linear Regression (Logistic Regression) to see which model would be have the highest accuracy. But before that, we will need to split our data into training and testing data sets.

##### 3.2.2 Generalized Linear Regression

1. step -> Randomly split patients into training and testing sets
2. step -> Logistic regression on training set to predict whether or not a patient experienced CHD within 10 years of first examination
3. step -> Evaluate predictive power on test set

Randomly split the data into training and testing sets. We may put 70% of the data in the training set. When you have more data like we do here, you can afford to put less data in the training set and more in the testing set. This will increase our confidence in the ability of the model to extend to new data since we have a larger test set, and still give us enough data in the training set to create our model.


```{r}
set.seed(1000)
split = sample.split(heart_disease$TenYearCHD, SplitRatio = 0.70)
train = subset(heart_disease, split==TRUE)
test = subset(heart_disease, split==FALSE)
```
Convert binary variables to numeric for better visualization train and test datas. 
```{r}
train$currentSmoker <- as.numeric(as.character(train$currentSmoker))
train$prevalentHyp <- as.numeric(as.character(train$prevalentHyp))
train$diabetes <- as.numeric(as.character(train$diabetes))
train$TenYearCHD <- as.numeric(as.character(train$TenYearCHD))
test$currentSmoker <- as.numeric(as.character(test$currentSmoker))
test$prevalentHyp <- as.numeric(as.character(test$prevalentHyp))
test$diabetes <- as.numeric(as.character(test$diabetes))
test$TenYearCHD <- as.numeric(as.character(test$TenYearCHD))
```


Now, we can make new generalized model as follows;

```{r}
glm_model <- glm(TenYearCHD ~ ., data=train, family=binomial, na.action=na.omit)
round(summary(glm_model)$coefficients, 3)
```


It looks like male, age, cigsPerDay, total cholesterol, systolic blood pressure, and glucose are all significant in our model. The diaBP is almost significant .

All of the significant variables have positive coefficients, meaning that higher values in these variables contribute to a higher probability of 10-year coronary heart disease.


Remove the insignificant variables and retrain the model.
```{r}
new_glm_model <- glm(TenYearCHD ~ male + age + totChol + cigsPerDay + sysBP + glucose, data=train, family=binomial)
```


#### 3.3.   Statistical Inference 

```{r}
summary(new_glm_model)
```



Our model formula can be generated as follow;

Heart Disease = -8.703 + 0.531*male1 + 0.067*age + 0.0029*totChol  + 0.019*cigsPerDay +  0.017*sysBP +  0.007*glucose

Confidence Intervals and P-values

We can check our model`s confidence interval. 
```{r}
confint(new_glm_model) 

```

We can do ANOVA also
```{r}
anova(new_glm_model, test="Chisq")
```

Looking at the ANOVA, age, male, restecg, totChol, cigsPerDay, sysBP, and glucose are significant factor for predicting heart disease. P- values are extremely close to zero.


##### Prediction
We'll call our predictions predictTest and use the predict function, which takes as arguments the name of our model, new_glm_model, then type = "response", which gives us probabilities, and lastly newdata = test, the name of our testing set.

We'll use the table function and give as the first argument, the actual values, test$TenYearCHD, and then as the second argument our predictions, predictTest > 0.5.


A confusion matrix contains information about actual and predicted classifications done by a classification system. Performance of such a system is commonly evaluated using the data in the matrix. 
Predictions on the test set and Confusion matrix with threshold of 0.5

```{r}
predictTest = predict(new_glm_model, type="response", newdata=test)
table(test$TenYearCHD, predictTest > 0.5)
```

Our model has 83.22% of accuracies.
It shows 924 NO heart diseases accurately and 150 NO death events were wrongly predicted as the death events.
Also, 6 heart diseases were wrongly predicted as NO heart diseases.
17 death events are accurately predicted.


### 4.  Results and Discussion

The study cohort accumulated 3,656 patients-years of observation with 10 years. The statistically independent predictive risk factors in our model are age, male, restecg, totChol, cigsPerDay, sysBP, and glucose. 
With every extra cigarette one smokes there is a 2% increase in the odds of CDH. Smoking and aging are more effective on heart disease.
For Total cholesterol level and glucose level there is no significant change.
There is a 1.7% increase in odds for every unit increase in systolic Blood Pressure.


### 5. Conclusions
We have an accuracy about 83.22% on our test set, which means that the model can differentiate between low risk patients and high risk patients pretty well.
Men seem to be more susceptible to heart disease than women. Increase in age, number of cigarettes smoked per day and systolic Blood Pressure also show increasing odds of having heart disease.
Total cholesterol shows no significant change in the odds of CHD. This could be due to the presence of 'good cholesterol(HDL) in the total cholesterol reading. Glucose too causes a very negligible change in odds (0.2%).


### 6.  References:
1. http://www.who.int/mediacentre/factsheets/fs317/en/
2. Wajid Shah, Cardiovascular and Chronic Respiratory Diseases Prediction System 
3. Sumit Sharma, Heart Diseases Prediction Using Hybrid Ensemble Learning,  Dublin Business School
4. Qi Zhenya & Zuoru Zhang, A hybrid cost-sensitive ensemble for heart disease prediction, Open Access Published: 25 February 2021
5. Fatma Zahra Abdeldjouad, A Hybrid Approach for Heart Disease Diagnosis and Prediction Using Machine Learning Techniques, Open Access Published: 23 June 2020
6. José A. Piniés, Fernando González-Carril, Development of a prediction model for fatal and non-fatal coronary heart disease and cardiovascular disease in patients with newly diagnosed type 2 diabetes mellitus,  Published: 12 September 2014
7. Saaol times, Monthly magazine, Modifiable risk factors of heart disease, pp. 6-10, July (2015), Google Scholar
8. M. A. Jabbar, Prediction of Heart Disease Using Random Forest and Feature Subset Selection, 15 December 2015
9. https://www.heart.org/?identifier=4726
10. https://www.cdc.gov/heartdisease/facts.htm







