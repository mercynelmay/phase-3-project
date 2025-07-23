# CUSTOMER CHURN PREDICTION
<img width="1000" height="429" alt="image" src="https://github.com/user-attachments/assets/86e17eae-4163-404b-b650-04b3b353dcea" />

 # Project Overview

This Jupyter Notebook outlines the end-to-end workflow of a customer churn prediction project in the telecom sector. The objective is to analyze customer behavior, identify churn patterns, and build a predictive model capable of detecting customers likely to leave. This enables telecom companies like SyriaTel to take proactive steps toward improving customer retention.

## Business  Understanding 
#### Stakeholders

SyriaTel executive shareholders that’s the board of directors and senior management of the syriaTel telecommunication company.

 #####   Problem Statement
SyriaTel is experiencing increased customer churn, which impacts revenue and raises customer acquisition costs. To address this, the company seeks to leverage data analytics to uncover the root causes of churn. The goal is to build predictive models that not only identify customers at risk of leaving but also explain the key factors influencing those decisions. This insight allows SyriaTel to take proactive action, enhance customer satisfaction, and reduce attrition.
In the highly competitive telecom industry, retaining existing customers is significantly more cost effective than acquiring new ones. SyriaTel aims to reduce customer attrition by analyzing both business and customer behavior data. A thorough understanding of the domain helps identify the key drivers of churn and informs the development of a predictive model to support data-driven retention strategies.

 ##### Data Source
The data for the project was obtained from the Kaggle website: https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset.


#####  Project Objectives

1. **Understand Customer Churn** - Analyze historical data to explore patterns and trends that lead to customer churn.

2. **Identify Key Drivers** - Detect the most significant factors influencing customer churn, such as service usage, billing issues, customer support interactions, or demographics.

3. **Preprocess and Prepare Data** - Clean, transform, and engineer features from raw data to create a high-quality dataset suitable for modeling.

4. **Develop Predictive Models** - Build and evaluate machine learning models capable of accurately predicting whether a customer is likely to churn.

5. **Interpret Model Results** - Use tools like SHAP values or feature importance plots to explain predictions and support business decision-making.

6. **Support Retention Strategies** - Deliver insights and tools that allow SyriaTel to implement proactive, data-driven interventions aimed at improving customer retention.

#### 2. Data Acquisition And Preparation

This phase involves collecting relevant customer data, understanding its structure, and preparing it for analysis.Here we will explore the data to get a better understanding of its state, then decide on the steps we need to take to clean it. 
we will perform the following :
* **Data collection:** Gather relevant customer data from various sources, ensuring data privacy regulations are met.
* **Data understanding:** Explore the data structure, get the shape of the data ,get data info , identify data types, and examine data dictionaries for accurate interpretation.
* **Data cleaning:** Address missing values, outliers, inconsistencies, and other data quality issues relevant to churn prediction.
###### sample Dataset

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>account length</th>
      <th>area code</th>
      <th>phone number</th>
      <th>international plan</th>
      <th>voice mail plan</th>
      <th>number vmail messages</th>
      <th>total day minutes</th>
      <th>total day calls</th>
      <th>total day charge</th>
      <th>...</th>
      <th>total eve calls</th>
      <th>total eve charge</th>
      <th>total night minutes</th>
      <th>total night calls</th>
      <th>total night charge</th>
      <th>total intl minutes</th>
      <th>total intl calls</th>
      <th>total intl charge</th>
      <th>customer service calls</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>128</td>
      <td>415</td>
      <td>382-4657</td>
      <td>no</td>
      <td>yes</td>
      <td>25</td>
      <td>265.1</td>
      <td>110</td>
      <td>45.07</td>
      <td>...</td>
      <td>99</td>
      <td>16.78</td>
      <td>244.7</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OH</td>
      <td>107</td>
      <td>415</td>
      <td>371-7191</td>
      <td>no</td>
      <td>yes</td>
      <td>26</td>
      <td>161.6</td>
      <td>123</td>
      <td>27.47</td>
      <td>...</td>
      <td>103</td>
      <td>16.62</td>
      <td>254.4</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NJ</td>
      <td>137</td>
      <td>415</td>
      <td>358-1921</td>
      <td>no</td>
      <td>no</td>
      <td>0</td>
      <td>243.4</td>
      <td>114</td>
      <td>41.38</td>
      <td>...</td>
      <td>110</td>
      <td>10.30</td>
      <td>162.6</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>84</td>
      <td>408</td>
      <td>375-9999</td>
      <td>yes</td>
      <td>no</td>
      <td>0</td>
      <td>299.4</td>
      <td>71</td>
      <td>50.90</td>
      <td>...</td>
      <td>88</td>
      <td>5.26</td>
      <td>196.9</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OK</td>
      <td>75</td>
      <td>415</td>
      <td>330-6626</td>
      <td>yes</td>
      <td>no</td>
      <td>0</td>
      <td>166.7</td>
      <td>113</td>
      <td>28.34</td>
      <td>...</td>
      <td>122</td>
      <td>12.61</td>
      <td>186.9</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>

From the class describer, the dataset has:

3333 customers.
21 customer features: 4 string predictors, 16 numeric predictors and the target.
Various transformations will be applied on the dataset both for analysis and modeling e.g type conversions, feature seleciton etc.

#### 3. Data Preparation
Summary on data preparation.

The Data has no missing values.
The Data has no duplicates.
The unique column Phone Number has no duplicates
Outliers were not be removed. See justification in notebook.

### 4. Exploratory Data Analysis (EDA)
In this section, we will explore univariate EDA and bivariate EDA.

Overall churn Distribution.
Churn against states.
Churn against international plan.
Churn against voice mail plan.
Churn against number of customer service calls.
##### Overall churn distribution.
<img width="349" height="247" alt="image" src="https://github.com/user-attachments/assets/cddabb38-4c41-41bb-bddf-c3d51bbcab65" />

### 5. Preprocessing
In this section we will prepare the data for modeling.
Some of the preprocessing that will take place here include:
Feature selection.
Train test split.
Encoding: dummy encoding and basic replace application.




