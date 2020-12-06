---
author-meta:
- Tessa Clarizio
- Marcel Briguglio
- Sudheer Salana
- Xiaokai Yang
- Jane Roe
- Jane Roe
bibliography:
- content/manual-references.json
date-meta: '2020-12-06'
header-includes: '<!--

  Manubot generated metadata rendered from header-includes-template.html.

  Suggest improvements at https://github.com/manubot/manubot/blob/master/manubot/process/header-includes-template.html

  -->

  <meta name="dc.format" content="text/html" />

  <meta name="dc.title" content="Predicting NO2 concentrations" />

  <meta name="citation_title" content="Predicting NO2 concentrations" />

  <meta property="og:title" content="Predicting NO2 concentrations" />

  <meta property="twitter:title" content="Predicting NO2 concentrations" />

  <meta name="dc.date" content="2020-12-06" />

  <meta name="citation_publication_date" content="2020-12-06" />

  <meta name="dc.language" content="en-US" />

  <meta name="citation_language" content="en-US" />

  <meta name="dc.relation.ispartof" content="Manubot" />

  <meta name="dc.publisher" content="Manubot" />

  <meta name="citation_journal_title" content="Manubot" />

  <meta name="citation_technical_report_institution" content="Manubot" />

  <meta name="citation_author" content="Tessa Clarizio" />

  <meta name="citation_author" content="Marcel Briguglio" />

  <meta name="citation_author" content="Sudheer Salana" />

  <meta name="citation_author" content="Xiaokai Yang" />

  <meta name="citation_author" content="Jane Roe" />

  <meta name="citation_author" content="Jane Roe" />

  <link rel="canonical" href="https://tessac2.github.io/498_NO2_pred/" />

  <meta property="og:url" content="https://tessac2.github.io/498_NO2_pred/" />

  <meta property="twitter:url" content="https://tessac2.github.io/498_NO2_pred/" />

  <meta name="citation_fulltext_html_url" content="https://tessac2.github.io/498_NO2_pred/" />

  <meta name="citation_pdf_url" content="https://tessac2.github.io/498_NO2_pred/manuscript.pdf" />

  <link rel="alternate" type="application/pdf" href="https://tessac2.github.io/498_NO2_pred/manuscript.pdf" />

  <link rel="alternate" type="text/html" href="https://tessac2.github.io/498_NO2_pred/v/498f1a6b9773670b7632aaf0d9f0885dd7902e19/" />

  <meta name="manubot_html_url_versioned" content="https://tessac2.github.io/498_NO2_pred/v/498f1a6b9773670b7632aaf0d9f0885dd7902e19/" />

  <meta name="manubot_pdf_url_versioned" content="https://tessac2.github.io/498_NO2_pred/v/498f1a6b9773670b7632aaf0d9f0885dd7902e19/manuscript.pdf" />

  <meta property="og:type" content="article" />

  <meta property="twitter:card" content="summary_large_image" />

  <link rel="icon" type="image/png" sizes="192x192" href="https://manubot.org/favicon-192x192.png" />

  <link rel="mask-icon" href="https://manubot.org/safari-pinned-tab.svg" color="#ad1457" />

  <meta name="theme-color" content="#ad1457" />

  <!-- end Manubot generated metadata -->'
keywords:
- markdown
- publishing
- manubot
lang: en-US
manubot-clear-requests-cache: false
manubot-output-bibliography: output/references.json
manubot-output-citekeys: output/citations.tsv
manubot-requests-cache-path: ci/cache/requests-cache
title: Predicting NO2 concentrations
...






<small><em>
This manuscript
([permalink](https://tessac2.github.io/498_NO2_pred/v/498f1a6b9773670b7632aaf0d9f0885dd7902e19/))
was automatically generated
from [tessac2/498_NO2_pred@498f1a6](https://github.com/tessac2/498_NO2_pred/tree/498f1a6b9773670b7632aaf0d9f0885dd7902e19)
on December 6, 2020.
</em></small>

## Authors



+ **Tessa Clarizio**<br>
    · ![GitHub icon](images/github.svg){.inline_icon}
    [tessac2](https://github.com/tessac2)<br>
  <small>
  </small>

+ **Marcel Briguglio**<br>
    · ![GitHub icon](images/github.svg){.inline_icon}
    [marcelbriguglio](https://github.com/marcelbriguglio)<br>
  <small>
  </small>

+ **Sudheer Salana**<br>
    · ![GitHub icon](images/github.svg){.inline_icon}
    [sudheersalana1](https://github.com/sudheersalana1)<br>
  <small>
  </small>

+ **Xiaokai Yang**<br>
    · ![GitHub icon](images/github.svg){.inline_icon}
    [xiaoky97](https://github.com/xiaoky97)<br>
  <small>
  </small>

+ **Jane Roe**<br>
    · ![GitHub icon](images/github.svg){.inline_icon}
    [janeroe](https://github.com/janeroe)<br>
  <small>
  </small>

+ **Jane Roe**<br>
    · ![GitHub icon](images/github.svg){.inline_icon}
    [janeroe](https://github.com/janeroe)<br>
  <small>
  </small>



## Abstract {.page_break_before}


test commit on abstract


# 1. Introduction

NO<sub>2</sub> is defined by the U.S. Environmental Protection Agency (EPA) as a criteria air pollutant, meaning it poses a risk to human and environmental health.  The primary National Ambient Air Quality Standard (NAAQS) for NO<sub>2</sub> is set at a 53 ppb annual average [@https://www.epa.gov/no2-pollution/primary-national-ambient-air-quality-standards-naaqs-nitrogen-dioxide]. NO<sub>2</sub> can cause respiratory irritation and can aggravate respiratory diseases such as asthma (US EPA, n.d., B). NO<sub>2</sub> can also react with other chemicals in the atmosphere to form both particulate matter (PM) and tropospheric ozone (O<sub>3</sub>) (US EPA, n.d., B). PM and O<sub>3</sub> are also criteria air pollutants and are harmful to human health. NOO<sub>2</sub> also contributes to the formation of acid rain, smog, and nutrient pollution in coastal waters (US EPA, n.d., B). The primary source of NO2 emissions is fossil fuel combustion, particularly from traffic and power plants (US EPA, n.d., B). 

Therefore, understanding and predicting the spatial variability of NO<sub>2</sub> emissions is of great importance to public health. However, prediction of air quality can be complicated due to the number of factors that affect local air quality, ranging from meteorology to land use. Machine learning models are a useful tool to interpret and find relationships in complex data. 

[introduce Bechle study…] Bechle et al (2015) explores the impact of.. [Grace please add here]

This report proposes a machine learning model to predict NO<sub>2</sub> concentrations spatially. First, a literature review was undertaken to understand what machine learning models have typically performed well in predicting air quality. Next, an exploratory data analysis (EDA) was performed on the Bechle et al (2015) dataset. Finally, multiple linear regression, neural network and random forest models were built and results were compared to see which method had the lowest mean-squared error (MSE). 


# 2. Methods
## 2.1 Literature Review
There are a number of studies examining how machine learning models can be used to predict air quality. Seven studies were examined as part of this literature review, and can be broadly categorized into 2 areas: predicting PM<sub>2.5</sub> and predicting the Air Quality Index (AQI)/ Air Pollution Index (API). One exception is that one of the studies examining AQI also predicted NO<sub>x</sub> concentrations. 

### 2.1.1 PM2.5
Chen et al (2018) explored the use of random forest (RF) models to predict PM<sub>2.5</sub> concentrations spatially in China and compared them to multiple linear regression (MLR) and generalized additive models. While the study began with a large number of predictors, these were narrowed down to ground-based measurements, satellite retrieved AOD data, urban cover data and  meteorological data. The random forests model had the greatest predictive power of all the models considered, with a root mean squared error (RMSE) of 28.1 µg/m<sup>3</sup> on a daily scale (R<sup>2</sup> = 83%), improving to 10.7 µg/m<sup>3</sup> (R<sup>2</sup> = 86%) and 6.9µg/m<sup>3</sup> (R<sup>2</sup> =86%) on monthly and annual time-scales, respectively. 

Xu et al (2018) likewise considered a number of machine learning models for PM<sub>2.5</sub> prediction in British Columbia, Canada. 8 models were examined in this study:  1) MLR, 2) Bayesian Regularized Neural Networks (BRNN), 3) Support Vector Machines with Radial Basis Function Kernel (SVM), 4) Least Absolute Shrinkage and Selection Operator (LASSO), 5) Multivariate Adaptive Regression Splines (MARS), 6) RF, 7) eXtreme Gradient Boosting (XGBoost), and 8) Cubist.The predictors included humidity, temperature, albedo, normalized difference vegetation index (NDVI), height of the planetary boundary layer (HPBL), wind speed, distance to the ocean, elevation, and calendar month beside the ground level monthly averaged P<sub>2.5</sub> data collected from 63 stations between 2001 to 2014 as well as 3km resolution aerosol optical depth (AOD) data from Moderate Resolution Imaging Spectroradiometer (MODIS). This study found that the cubist model had the highest accuracy (RMSE=2.64 µg/m<sup>3</sup> and R<sup>2</sup>=0.48) and the the MLR had the lowest accuracy (MSE = 3.24 µg/m<sup>3</sup> and R<sup>2</sup>=0.22). The predictors with the most influence were monthly AOD and elevation. 
 
Enebish et al (2020) considered 6 different machine learning models for PM<sub>2.5</sub> prediction in Mongolia: 1) RF, 2) gradient boosting, 3) SVM with a radial basis kernel, 4) MARS, 5) generalized linear model with elastic net penalties (a type of MLR), and 6) generalized additive model. These models were run for annual data, cold season and warm season. Parameters considered were air pollution monitoring data, meteorology, land use and population. Across all time periods, the RF had the best R<sup>2</sup> and RMSE values. Over the entire period using the hold-out test set, RF had a RMSE of 12.92 (R<sup>2</sup> = 0.96), and the cold season and warm season had RMSE of 21.23 (R<sup>2</sup> = 0.92) and 7.44 (R<sup>2</sup> = 0.84), respectively. 


### 2.1.2 AQI/API
Azid et al (2014) used a multilayered perceptron feed-forward artificial neural network model to predict API, using daily measurements of NO<sub>2</sub>, SO<sub>2</sub>, CO, PM<sub>10</sub> and O<sub>3</sub> over a period of 7 years in Malaysia. The best RMSE and R<sup>2</sup> occurred when the hidden nodes were set to 6, and were 0.618 and 10.017, respectively. 

Gu et al (2020) focuses on predicting the AQI in Shenzen, China. The dataset consists of 365 sets of daily pollution data over one calendar year (2018), and the purpose was to develop a model to predict AQI. Pollution measurements included PM<sub>2.5</sub>, PM<sub>10</sub>, SO<sub>2</sub>, CO, NO<sub>2</sub> and O<sub>3</sub>. Two SVM models were developed: smart adaptive particle swarm optimization and particle swarm optimization, SAPSO-SVM and PSO-SVM, respectively. Additionally, a back propagation (BP) neural network model was developed. SAPSO-SVM had a test set classification accuracy of 91.62%, and PSO-SVM 88.56%. For the BP-neural network model, ten iterations of the algorithm best fit the test data set, where the percent error ranged from 18.41% for PM<sub>2.5</sub> to 30.29% for SO<sub>2</sub>. While Gu et al stated that both models were a good fit for the data, by using different statistical comparisons to explain model fit, it is not clear which of the two models has a better predictive ability, although it appears to be SAPSO-SVM. The paper listed a number of limitations associated with its neural network, particularly the limited data points. 

Liu et al (2019) developed SVM and RF models to predict hourly AQI in Beijing, China and hourly NO<sub>x</sub> in an Italian city. Parameters included historical hourly averaged AQI concentrations for PM<sub>2.5</sub>, O<sub>3</sub>, SO<sub>2</sub>, PM<sub>10</sub> and NO<sub>2</sub> in Beijing (five years), and hourly averaged responses for CO, non-methane hydrocarbons, benzene, NO<sub>x</sub> and NO<sub>2</sub> in the Italian city (1 year). The SVM performed better in predicting AQI in Beijing with a RMSE 7.666 (R2=0.9776), but the RF model performed better in predicting NO<sub>x</sub> concentrations in the Italian city (RMSE = 83.67, R2 = 0.8401).  

Singh et al (2013) used ensemble learning methods to predict air quality index in Lucknow, India. They trained four different models: single decision tree (SDT), decision tree forest (DTF), decision treeboost (DTB) and SVM. While decision trees can be different from random forest, it appears in Singh’s methodology that the DTF and DTB involve randomization with replacement from the training dataset to create separate models, which are then used to predict the entire data from the subsets. This is consistent with RF models, as essentially RF are ensemble decision trees. The parameters included in the model are 5 years of data on: daily air quality measurements (SO<sub>2</sub>, NO<sub>2</sub>, suspended particulate matter and respirable suspended particulate matter) meteorology (air temperature, relative humidity, wind speed, evaporation, and daily sunshine period).  The DTF and DTB models outperformed the SVM models. DTB performed the best, with a RMSE of 4.38 (R<sup>2</sup> = 0.92).

## 2.1.3 Comparison of PM2.5 and AQI/API studies
The main difference between the PM<sub>2.5</sub> and the AQI studies is that studies examining PM<sub>2.5</sub> tended to only examine one pollutant, whereas AQI studies consisted of measuring and modeling a number of different pollutants. Therefore, some AQI models were more interested in classification than predicting a specific pollutant spatially or temporally. As a result, different parameters tended to be included in the model depending on if it was predicting PM<sub>2.5</sub> or AQI. For example, meteorological data tended to be included in PM<sub>2.5</sub> studies, but not in studies examining API/AQI.  Additionally, different types models tended to perform best depending on the target prediction. 

A common limitation of all of the studies is the volume of missing data. In Chen et al (2018), the model had only two years of ground-based measurements to train the model on (2014-2016), and then predicted PM<sub>2.5</sub> concentrations for a ten year period (2005 to 2014). Xu et al, 2018 also discussed the challenge of missing data, averaging hourly and daily measurements where available to monthly concentrations to use in model development. Enebish et al, 2020 discussed there being few air quality monitoring stations and insufficient data to well represent the high seasonal variability of PM<sub>2.5</sub> concentrations. Additionally, Gu et al (2020) only used pollution data from one year within one region in China, and Liu et al (2019) used pollution data from one Italian city over one year.

The models in each of these studies is summarized in Table 2.1 below:

**Table 2.1 Models used in literature** 

|PM<sub>2.5</sub>|Both PM<sub>2.5</sub> and AQI|
|-----|------------------|
|**MLR** (Xu et al, 2018; Enebish et al, 2020; Chen et al, 2018) | **RF** (Chen et al, 2018; Xu et al, 2018; Singh et al, 2013; Liu et al, 2019; Enebish et al, 2020)|
| **LASSO** (Xu et al, 2018) |  **Neural Network** (Azid et al, 2014; Xu et al, 2018, Gu et al, 2020) |
| **MARS** (Xu et al, 2018; Enebish et al, 2020 ) | **SVM**  (Xu et al, 2018; Gu et al, 2020; Liu et al, 2019; Enebish et al, 2020; Singh et al, 2013) | 
| **Gradient Boosting** (Xu et al, 2018; Enebish et al, 2020) | |
| **Cubist** (Xu et al, 2018) | |
| **Generalized additive model** (Enebish et al, 2020; Chen et al, 2018)| |
| **Mixed effects models** (Chen et al, 2018) |  |

As we can see above more models were used to predict PM<sub>2.5</sub> than AQI, and the ones that were used in AQI studies were also used in predicting PM<sub>2.5</sub>. The best-predicting models in each study are shown in Table 2.2, alongside their RMSE and R<sup>2</sup> values. 

**Table 2.2 Best models in each study**

| Study | Target Prediction | Best Model | RMSE | R<sup>2</sup> |
|-------|-------------------|------------|------|----|
|Chen et al (2018)|Annual average PM2.5|Random Forest|6.9|0.86|
|Xu et al (2018)|Monthly average PM2.5|Cubist|2.6|0.48|
|Enebish et al (2020)|Annual average PM2.5|Random Forest|12.9|0.96|
|Azid et al (2014)|Daily AQI|Neural network|10.0|0.62|
|Gu et al (2020)|Daily AQI|SVM|n.a.|n.a|
|Singh et al (2013)|Daily AQI|Random Forest|4.4|0.92|
|Liu et al (2019)|Hourly AQI|SVM|7.7|0.98|
|Liu et al (2019)|Houly NOx|Random Forest|83.7|0.84|

Table 2.2 demonstrates that RF models tend to provide the most accurate prediction when considering a single pollutant, with 3/4 studies looking at PM<sub>2.5</sub> or NO<sub>x</sub> having RF as the best predicting model. When examining AQI, SVM models tend to work best, with 2/4 studies finding SVM provides the best prediction. 

The RMSE and R<sup>2</sup> values vary significantly for each study. This can be attributed to the different geographic areas considered, varying spatial resolutions, amount of uncertainty in the data sources, prediction type and different parameters included in the model. 

Because the objective of this study is to predict a single variable (NO<sub>2</sub> concentrations), then the models used by PM<sub>2.5</sub> studies are the most relevant. Therefore, in our model analysis, we will use MLR, neural networks and RF. 


## 2.2 Exploratory Data Analysis
## 2.3 Model
### 2.3.1 Multiple Linear Regression
### 2.3.2 Neural Networks
Artifical neural networks are based on the design philosophy of the neural connections in our brain. They consist of a group of nodes interconnected to each other through edges. The edges transmit the signals (which in a machine learning model would be a real number) from one neuron to another just like a synapse functions in a brain.

A typical neural network has many layers. Each layer is composed of a set of neurons and each layer transforms and processes data in a different way based on the hyperparameters of the model.

There are different kinds of neural networks. The most commonly used of which are:
1. Feed Forward Neural Network
2. Convolutional Neural Network (CNN)
3. Recurrent Neural Network (RNN)
4. Long-short term memory neural network (LSTM)
5. Gated Recurrent Unit (GRU)

There are several other advanced ones such as General Adversarial Networks,Auto-encoders and Deep Belief Neural networks.

We will first discuss, briefly, the CNN, RNN, LSTM and GRU before moving on to the Feed formal neural network which is the neural network model used in this project.
#### 2.3.2.1 CNN
Largely used for classifying image and audio data, the convolutional neural network, similar to neural networks has an input and output layer with hidden layers in between them. Further, a CNN has three layers called convolutional layer, pooling layer and a fully-connected layer which is similar to the regular neural network. The convolutional layer takes the input from the input layer and convolutes the data and sends it downstream for further processing. In simpler terms, in CNNs an input image is taken and a filter is applied on it repeatedly acorss the image to create a feature map which can be used to identify and classify the input image.

#### 2.3.2.2 RNN
These neural networks are used to predict the temporal trend of the data.In these kind of models, the 'memory' of the previous inputs of is stored and used for further processing of future inputs and outputs.This relationship between input and output data could be both unidirectional (moving in the forward direction) or bidirectional where future data could also be used to improve the current inputs and outputs.These neural networks could use a variety of relationships between their inputs and outputs and could be described as one to one, one to many, many to many and many to one relationships. Sigmoid, ReLu, Tanh are the common activations used in RNNs.

#### 2.3.2.3 LSTM
LSTMs are a type of RNN which solve the common problems that RNNs face, that is their inability to efficiently handle short-term memory over a lengthy series of steps. LSTMs solve this problem using a memory cell and gating units and consists of a set of gates called input, output and forget.The input gate is responsible for monitoring and deciding the kind and quantity of data that is allowed to enter the cell, the memory gate is responsible for deciding the proportion of data that should be 'forgotten' and which information is useless and to be discarded. Finally, the output gate is responsible for deciding the amount of data that is passed as output from the cell.


### 2.3.3 Random Forest
Trees and tree algorithm is among the most widely applied machine learning algorithms. It includes random forest, gradient boosting decision trees, XGBoost, etc. The fundament of trees algorithm is decision tree, which can be divided into classification tree and regression tree. In this project, the label is the observed NO<sub>2</sub> concentration, thus, regression tree algorithm can be applied to make the prediction.

Random forest is a combination of tree predictors based on model aggregation ideas. It is realized by creating an ensemble of trees by generating random vectors that govern the growth of trees and letting them vote for the most popular label [@Breiman2001]. 

Random forest have the advantages of low overfit, low noise affect. However, for random forest regression,  a relatively large number of features are required to reduce the test set error[@Breiman2001].

Variable selection and feature engineering are very important in data preprocess of random forest. The two main objectives are to find variables highly related to the response variable, and to find a small number of variables sufficient to a good prediction of the response variable[@Genuer2010].

Based on previous exploratory data analysis, the raw dataset contains valuable land-use variables in series including impervious surfaces, population, major road length, residential road length, and total road length within different buffers. It also include elevation, distance to coast, latitude, longitude, linear combinations of satellite data and WRF-Chem output. The variables within their series are highly correlated, so a few representative ones should be selected out based on the second criteria mentioned before. Here we select one for each kind of land-use variables. It is interesting that high correlations have been found between some of these land-use variables, like road lengths. Thus, we only selected the road length having the highest relationship with the variable to be predicted. Latitude and longitude are binned and one-hot coded. Around 20% of the raw data were selected out for prediction validation.

Random forest model in this project is created with scikit-learn RandomForestRegressor library. GridSearchCV was imported to conduct hyperparameter optimization. The model generally get a MSE around 3.2 in the validation data. However, on the test data it did not show a good performance (MES = 4.2). The first reason might be that random forest algorithm perform worse on regression problems than on classification problems since it cannot give a continuous prediction. It is also weak on predicting values with a magnitudes beyond the train dataset, meaning that it cannot confidently handle noise or outliers. It prefers high dimensional, large-scale data, however, in this project the data is in a relatively low scale. 

In conclusion, random forest is possibly not a very appropriate algorithm to make prediction in regression problems from dataset in a low-scale and a low-dimension.
# 3. Results
# 4. Discussion 



[@Genuer2010]: doi:10.1016/j.patrec.2010.03.014


## 3. Results


Figure (code) shows the architecture of the neural network used that achieved the best performance. 
Many choices made regarding were governed by two main motives; reducing the complexity of the model and ensuring the model captures the entire range of concentrations in the training data. 
Both of 


The final model was further analyzed using two major parameters: squared error and absolute error.
Squared error penalizes predictions further from the observed concentrations, which is beneficial for detecting outliers and anomalies in the data. 
The squared error is also related to the root mean squared error, the measured used to determine the best model in the Kaggle competition meaning it can provide insights on the individual monitors in the training data. 
The problem is that the loss function used was the mean squared error, meaning that only using this metric can may cause us to fail to observe other possible problems while training the data.
Absolute error as a criterion for analysis fills these gaps in knowledge, such as bias in the training data or the geographic distribution of the error. 

![Figure X. Training Data Absolute Error Histogram](images/Absolute_dis.png)

Figure H (histogram) shows the distribution for the absolute error on the training data. 
The distribution is nearly gaussian, which is expected with a large sample size.
The distribution is also centered on 0 and has a very small skew, meaning that there is no bias in the when the model was training.
The major difference between the distribution and a Gaussian distribution is the kurtosis.
The kurtosis was found to be greater than 3, meaning that the data is more concentrated near the mean with some extreme outliers. 
This is interesting because since the loss function of the model was mean squared error, these outliers should be minimized.


![Figure X. Training Data Absolute Error Geographic Distribution](images/Absolute_map.png)

Figure A (Absolute value map) shows the geographic distribution of the absolute error. 
This figure shows that the absolute error is not spatially homogenous, with most of the outliers being in the West. 
Further analysis shows 75% of the outliers were in contained in the Western region mostly in suburban areas outside of major cities that were surrounded by nature. 
Another characteristic of the outliers were that they were mainly found in suburban areas near isolated major urban areas. 


All the previous analysis was done on the training data but to ensure our model is robust, the model needs to be run on other data. 
The model was run on 165 samples and the root mean squared error was found to be 2.925, which is lower than the root mean squared error on the training data (3.05). 
This means that the final model is robust and can be applicable across the United States.  


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>
