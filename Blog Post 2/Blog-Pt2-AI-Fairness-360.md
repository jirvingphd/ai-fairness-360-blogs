# Blog Post Series - AI Fairness 360: Mitigating Bias in Machine Learning Models

## Part 2: Implementing Bias Mitigation Techniques with AI Fairness 360

![png](DALLE-banner.png)



### Introduction

Welcome back to our blog series on AI Fairness 360! In Part 1, we explored the foundational concepts of bias in AI and demonstrated how to create dataset objects and evaluate fairness metrics using IBM's AI Fairness 360 toolkit. Understanding and detecting bias in machine learning models is crucial, but the next step is equally important: mitigating this bias to ensure fair and equitable AI systems.

In this second installment, we will delve into various bias mitigation techniques provided by AI Fairness 360. These techniques are designed to address bias at different stages of the machine learning pipeline, from pre-processing the data to post-processing the model's predictions. By implementing these techniques, we can reduce bias and improve the fairness of our AI models, ultimately leading to more ethical and responsible AI applications.



### Learning Objectives

By the end of this blog post, you will be able to:
- Understand different bias mitigation techniques available in AI Fairness 360.
- Implement pre-processing, in-processing, and post-processing bias mitigation algorithms.
- Evaluate the effectiveness of these techniques using fairness metrics.



### Prerequisites

Before diving into the content, make sure you have:
- A basic understanding of AI Fairness 360 and its dataset objects, as covered in Part 1.
- Familiarity with Python and machine learning concepts.

In this post, we will provide step-by-step guides and code snippets to help you implement and understand these bias mitigation techniques. Let's get started on the journey to making our AI models more fair and equitable!



# **Overview of Bias Mitigation Techniques**



Bias mitigation is an essential step in developing fair and ethical AI systems. AI Fairness 360 offers various techniques to reduce bias at different stages of the machine learning pipeline. These techniques can be broadly categorized into three types: pre-processing, in-processing, and post-processing.

#### **A) Pre-Processing Techniques**

Pre-processing techniques aim to modify the training data to reduce bias before the model is trained. By addressing bias in the dataset, these methods ensure that the model learns from a more balanced and fair representation of the data.

- **Reweighing**: This technique assigns different weights to the examples in the training data based on the protected attributes to ensure that the model gives equal importance to all groups.
- **Optimized Preprocessing**: This technique transforms the dataset to remove bias while preserving the relationships between features and the target variable.

#### **B) In-Processing Techniques**

In-processing techniques modify the learning algorithm itself to reduce bias during model training. These methods integrate fairness considerations directly into the training process.

- **Adversarial Debiasing**: This technique uses adversarial learning to minimize bias. It trains the model to make accurate predictions while an adversary tries to detect bias based on the protected attributes.
- **Prejudice Remover**: This technique adds a regularization term to the learning algorithm's objective function to penalize biased outcomes, promoting fairness during training.

#### **C) Post-Processing Techniques**

Post-processing techniques adjust the predictions of a trained model to reduce bias. These methods are applied after the model has been trained and focus on modifying the model’s outputs to ensure fairness.

- **Equalized Odds Post-Processing**: This technique adjusts the model's predictions to achieve equalized odds, ensuring that the false positive and false negative rates are the same across different groups.
- **Calibrated Equalized Odds Post-Processing**: This technique combines calibration with equalized odds to adjust the model's predictions, maintaining both fairness and accuracy.

In the following sections, we will provide step-by-step guides to implement these techniques using AI Fairness 360. By understanding and applying these methods, you can develop AI models that are not only accurate but also fair and unbiased.



## **A) Implementing Pre-Processing Techniques**



Pre-processing techniques modify the training data to mitigate bias before the model is trained. These methods ensure that the model learns from a more balanced and fair dataset. In this section, we will focus on the Reweighing technique.

### **Step-by-Step Guide: Reweighing Technique**

The Reweighing technique assigns different weights to the examples in the training data based on the protected attributes to ensure that the model gives equal importance to all groups.



#### 1. Load the Dataset and Convert to AIF360 Dataset Object

First, we need to load the dataset and convert it into an AIF360 dataset object. We'll use the same Recidivism dataset from Part 1.



```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Import BinaryLabelDataset
from aif360.datasets import BinaryLabelDataset

# Enable pandas dataframe output
from sklearn import set_config
set_config(transform_output='pandas')


pd.set_option("display.max_columns", 100)

# Load the data 
df = pd.read_csv("data/Iowa_Prison_Recidivism_Status_20240724.csv", 
                 index_col=0, usecols=range(0, 23-7))   

## Quick Conversion of Dtypes for Clean Data
df = df.convert_dtypes(convert_string=False)

# Drop unnecessary columns
drop_cols = ['Supervising Unit','Supervision Start Date','Supervision End Date'] + [c for c in df.columns if 'Year' in c]
df = df.drop(columns=drop_cols)
df.info()

# Encode the 'race' column as binary white or non-white.
race_map = {'White': 0, 'Black': 1, 'Hispanic': 1, 'Asian or Pacific Islander': 1,
            'American Indian or Alaska Native': 1, 'Unknown': 1, 'Other':1}
df['Race'] = df['Race'].map(race_map)

# Encode the "Sex" column
sex_map = {"Male": 0, "Female":1}
df['Sex'] = df['Sex'].map(sex_map)


df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 25244 entries, 20655350 to 20999813
    Data columns (total 10 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   Race                         25244 non-null  object 
     1   Sex                          25241 non-null  object 
     2   Age                          25244 non-null  Int64  
     3   Supervision Type             25244 non-null  object 
     4   Months Supervised            25244 non-null  Int64  
     5   Supervision End Reason       25244 non-null  object 
     6   Supervision Offense Class    25244 non-null  object 
     7   Supervision Offense Type     25244 non-null  object 
     8   Supervision Offense Subtype  25244 non-null  object 
     9   Reincarcerated               25244 non-null  boolean
    dtypes: Int64(2), boolean(1), object(7)
    memory usage: 2.0+ MB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Race</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Supervision Type</th>
      <th>Months Supervised</th>
      <th>Supervision End Reason</th>
      <th>Supervision Offense Class</th>
      <th>Supervision Offense Type</th>
      <th>Supervision Offense Subtype</th>
      <th>Reincarcerated</th>
    </tr>
    <tr>
      <th>Offender Number</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20655350</th>
      <td>0</td>
      <td>0.0</td>
      <td>35</td>
      <td>Prison</td>
      <td>125</td>
      <td>Discharged - Expiration of Sentence</td>
      <td>B Felony</td>
      <td>Violent</td>
      <td>Sex</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18876932</th>
      <td>1</td>
      <td>0.0</td>
      <td>30</td>
      <td>Prison</td>
      <td>49</td>
      <td>Released to Special Sentence</td>
      <td>D Felony</td>
      <td>Other</td>
      <td>Other Criminal</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2424146</th>
      <td>0</td>
      <td>0.0</td>
      <td>40</td>
      <td>Prison</td>
      <td>8</td>
      <td>Parole Granted</td>
      <td>Aggravated Misdemeanor</td>
      <td>Public Order</td>
      <td>Other Public Order</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19088303</th>
      <td>1</td>
      <td>0.0</td>
      <td>29</td>
      <td>Work Release</td>
      <td>2</td>
      <td>Parole Granted</td>
      <td>C Felony</td>
      <td>Drug</td>
      <td>Trafficking</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20280797</th>
      <td>0</td>
      <td>0.0</td>
      <td>38</td>
      <td>Prison</td>
      <td>11</td>
      <td>Paroled w/Immediate Discharge</td>
      <td>Aggravated Misdemeanor</td>
      <td>Property</td>
      <td>Burglary</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python

## Create the Preprocessor
# Categorical Pipeline
cat_cols = df.select_dtypes(include='object').columns
cat_imputer = SimpleImputer(strategy='most_frequent')
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_pipe = make_pipeline(cat_imputer, ohe)


# Numeric Pipeline
num_cols = df.select_dtypes(include='number').columns
num_imputer = SimpleImputer(strategy='mean')
num_pipe = make_pipeline(num_imputer)



# Convert Boolean Columns to Integers
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)



# Create the column Transformer
preprocessor = ColumnTransformer(transformers=[('cat', cat_pipe, cat_cols),
                                               ('num', num_pipe, num_cols)],
                                 remainder='passthrough',
                                 verbose_feature_names_out=False)

```



### Choice: Dataset First or Train-Test-Split First

We will perform our train-test-split before applying the preprocessig steps in order to avoid data leakage between the train and test sets.  However, you may see examples and tutorials that use the BinaryLabelDataset built-in `.split()` method for creating training and test datasets. We will avoid doing so to have the most appropriate machine learning workflow possible. 

#### Train-Test Split


```python
from sklearn.model_selection import train_test_split

# Conventional train-test split
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Preprocess the training set
train_df = preprocessor.fit_transform(train_df)

# Preprocess the test set
test_df = preprocessor.transform(test_df)  # Example preprocessing step

train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Supervision Type_Prison</th>
      <th>Supervision Type_Work Release</th>
      <th>Supervision End Reason_Discharged - Expiration of Sentence</th>
      <th>Supervision End Reason_Parole Granted</th>
      <th>Supervision End Reason_Paroled to Detainer - INS</th>
      <th>Supervision End Reason_Paroled to Detainer - Iowa</th>
      <th>Supervision End Reason_Paroled to Detainer - Out of State</th>
      <th>Supervision End Reason_Paroled to Detainer - U.S. Marshall</th>
      <th>Supervision End Reason_Paroled w/Immediate Discharge</th>
      <th>Supervision End Reason_Released to Special Sentence</th>
      <th>Supervision Offense Class_A Felony</th>
      <th>Supervision Offense Class_Aggravated Misdemeanor</th>
      <th>Supervision Offense Class_B Felony</th>
      <th>Supervision Offense Class_C Felony</th>
      <th>Supervision Offense Class_D Felony</th>
      <th>Supervision Offense Class_Felony - Enhancement to Original Penalty</th>
      <th>Supervision Offense Class_Felony - Mandatory Minimum</th>
      <th>Supervision Offense Class_Other Felony</th>
      <th>Supervision Offense Class_Serious Misdemeanor</th>
      <th>Supervision Offense Class_Simple Misdemeanor</th>
      <th>Supervision Offense Class_Special Sentence 2005</th>
      <th>Supervision Offense Type_Drug</th>
      <th>Supervision Offense Type_Other</th>
      <th>Supervision Offense Type_Property</th>
      <th>Supervision Offense Type_Public Order</th>
      <th>Supervision Offense Type_Violent</th>
      <th>Supervision Offense Subtype_Alcohol</th>
      <th>Supervision Offense Subtype_Animals</th>
      <th>Supervision Offense Subtype_Arson</th>
      <th>Supervision Offense Subtype_Assault</th>
      <th>Supervision Offense Subtype_Burglary</th>
      <th>Supervision Offense Subtype_Drug Possession</th>
      <th>Supervision Offense Subtype_Flight/Escape</th>
      <th>Supervision Offense Subtype_Forgery/Fraud</th>
      <th>Supervision Offense Subtype_Kidnap</th>
      <th>Supervision Offense Subtype_Murder/Manslaughter</th>
      <th>Supervision Offense Subtype_OWI</th>
      <th>Supervision Offense Subtype_Other Criminal</th>
      <th>Supervision Offense Subtype_Other Drug</th>
      <th>Supervision Offense Subtype_Other Government</th>
      <th>Supervision Offense Subtype_Other Public Order</th>
      <th>Supervision Offense Subtype_Other Violent</th>
      <th>Supervision Offense Subtype_Prostitution/Pimping</th>
      <th>Supervision Offense Subtype_Robbery</th>
      <th>Supervision Offense Subtype_Sex</th>
      <th>Supervision Offense Subtype_Stolen Property</th>
      <th>Supervision Offense Subtype_Theft</th>
      <th>Supervision Offense Subtype_Traffic</th>
      <th>Supervision Offense Subtype_Trafficking</th>
      <th>Supervision Offense Subtype_Vandalism</th>
      <th>Supervision Offense Subtype_Weapons</th>
      <th>Race</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Months Supervised</th>
      <th>Reincarcerated</th>
    </tr>
    <tr>
      <th>Offender Number</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>614774</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>60.0</td>
      <td>236.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19154684</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>32.0</td>
      <td>4.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19189433</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>7.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>147731</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19234385</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>31.0</td>
      <td>4.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python

# Convert to BinaryLabelDatasets
train = BinaryLabelDataset(
    favorable_label=0, # Not reincarcerated
    unfavorable_label=1, # Reincarcerated
    df=train_df,
    label_names=['Reincarcerated'],
    # protected_attribute_names=['Race',"Sex"]
    protected_attribute_names = ["Sex"]
)

test = BinaryLabelDataset(
    favorable_label=0,
    unfavorable_label=1,
    df=test_df,
    label_names=['Reincarcerated'],
    # protected_attribute_names=['Race',"Sex"]
    protected_attribute_names = ["Sex"]

)

len(train.labels), len(test.labels)
```




    (17670, 7574)



####

We will use the traditional train-test-split method to split the dataset into training and testing sets. Many of the official AIF360 examples use pre-constructed datasets and do not acknowledge the data leakage issue.

### Assessing Model Bias 

Before we implement mitgation methods dataset, let' create a baseline model without bias mitigation and examine its fairness metrics.

We will use the following function to evaluate our machine learning models and to collect and compare their performance at the end of this post. 


```python

def evaluate_model(y_true, y_pred, output_dict=False, results_label=''):
    """
    Evaluate the performance of a classification model by calculating classification metrics and displaying a confusion matrix.

    Parameters:
    - y_true (array-like): The true labels.
    - y_pred (array-like): The predicted labels.
    - output_dict (bool, optional): Whether to return the classification report as a dictionary. Default is False.
    - results_label (str, optional): A label to identify the results. Default is an empty string.

    Returns:
    - If output_dict is True, a dictionary containing the classification report and the results label.
    """
    print(f"Classification Metrics: {results_label}\n\n")

    print(classification_report(y_true, y_pred))
    ax = ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize='true',cmap='Greens',
                                            display_labels=['Non-Recid','Recid'])
    ax = plt.gca()
    ax.set_title(f"Confusion Matrix: {results_label}")
    plt.show()
    
    if output_dict==True:
        report = classification_report(y_true, y_pred, output_dict=True)
        report['Label'] = results_label
        return report

```

##### Baseline Model (No mitigation applied)

We will be using a LogisticRegression model throughout this blog post for simplicity. However, this approach can be applied with any machine learning classification model.


```python
from sklearn.linear_model import LogisticRegression

# Extract features and labels from the reweighed dataset
X_train = train.features
y_train = train.labels.ravel()

X_test = test.features
y_test = test.labels.ravel()

# Train the model
model =  LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
eval_unmit = evaluate_model(y_test, model.predict(X_test), results_label='Unmitigated Model', output_dict=True) 
```

    Classification Metrics: Unmitigated Model


​    
                  precision    recall  f1-score   support
    
             0.0       0.74      0.57      0.64      4738
             1.0       0.48      0.67      0.56      2836
    
        accuracy                           0.61      7574
       macro avg       0.61      0.62      0.60      7574
    weighted avg       0.64      0.61      0.61      7574




![png](output_26_1.png)
    


We can see that we have an overall accuracy of 61%, however, when examining the recall scores we can see that the model is more confident in predicting reincarceration(recid) than Non-Recidivism.

### Fairness Metrics - Model Predictions

Evaluating the fairness of a machine learning model is crucial to ensure that the model does not exhibit bias towards any particular group. AI Fairness 360 provides the `ClassificationMetric` class, which offers a variety of fairness metrics to assess the performance of your models. In this section, we'll define key fairness metrics and demonstrate how to calculate them using AI Fairness 360.



In order to use it, we must have a dataset of true values and a dataset of predicted values. Therefore, we will make a copy of the test dataset before overwriting the `.labels` 


```python
type(test)
```




    aif360.datasets.binary_label_dataset.BinaryLabelDataset




```python
from aif360.metrics import ClassificationMetric
from aif360.explainers import MetricTextExplainer

# Get predictions
y_pred = model.predict(X_test)

# Create a new dataset with predictions
test_pred = test.copy()
test_pred.labels = y_pred


# Define privileged and unprivileged groups
privileged_groups = [{'Sex': 0}]  # Male
unprivileged_groups = [{'Sex': 1}]  # Female/Other


# Evaluate fairness
metric = ClassificationMetric(
    test,
    test_pred,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

explainer = MetricTextExplainer(metric)
```

    /opt/homebrew/Caskroom/miniforge/base/envs/fair-env/lib/python3.12/site-packages/torch/_functorch/deprecated.py:61: UserWarning: We've integrated functorch into PyTorch. As the final step of the integration, functorch.vmap is deprecated as of PyTorch 2.0 and will be deleted in a future version of PyTorch >= 2.3. Please use torch.vmap instead; see the PyTorch 2.0 release notes and/or the torch.func migration guide for more details https://pytorch.org/docs/master/func.migrating.html
      warn_deprecated('vmap', 'torch.vmap')




#### **Key Fairness Metrics**

1. **Disparate Impact**: Measures the ratio of the favorable outcome rates between the unprivileged and privileged groups. 

   $$
   \text{Disparate Impact} = \frac{\text{P(y=1|unprivileged)}}{\text{P(y=1|privileged)}}
   $$


- A value close to 1 indicates fairness.
- A value less than 1 indicates bias toward predicting positive outcomes for the privileged class.
- A value greater than 1 indicates bias towards the unpriviledged class.



```python
# Disparate Impact
disparate_impact = metric.disparate_impact()
print(f"Disparate Impact: {disparate_impact:.2f}")

print(explainer.disparate_impact())
```

    Disparate Impact: 1.56
    Disparate impact (probability of favorable outcome for unprivileged instances / probability of favorable outcome for privileged instances): 1.5575745625603374


We can see that, with a value of 1.56, which is significantly than 1, indicating that our baseline model is biased towards predicting a favorable outcome for the unprivileged class.


2. **Statistical Parity Difference**: The difference in the probability of favorable outcomes between the unprivileged and privileged groups. 
   $$
   \text{Statistical Parity Difference} = \text{P(y=1|unprivileged)} - \text{P(y=1|privileged)}
   $$
   
- Value close to 0 indicates fairness.
- Negative Value indicates that the unprivileged group has a lower probability of receiving positive outcomes compared to the privileged group.
- Positive Value indicates the privileged group has a lower probability of receiving positive outcomes compare to the unprivileged group.


```python
# Statistical Parity Difference
statistical_parity_difference = metric.statistical_parity_difference()
print(f"Statistical Parity Difference: {statistical_parity_difference:.2f}")

print(explainer.statistical_parity_difference())
```

    Statistical Parity Difference: 0.25
    Statistical parity difference (probability of favorable outcome for unprivileged instances - probability of favorable outcome for privileged instances): 0.24767723324582192


We can see that unprivileged classes are 4% less likely to be predicted to receive a positive predictio than the privileged class.


3. **Equal Opportunity Difference**: The difference in the true positive rates between the unprivileged and privileged groups. 



$$
   \text{Equal Opportunity Difference} = \text{TPR(unprivileged)} - \text{TPR(privileged)}
$$

-	Value Close to 0: A value close to 0 suggests that the model has similar true positive rates for both groups, indicating fairness in terms of equal opportunity.

-	Positive Value: A positive Equal Opportunity Difference indicates that the unprivileged group has a higher true positive rate compared to the privileged group.
-	Negative Value: A negative Equal Opportunity Difference indicates that the unprivileged group has a lower true positive rate compared to the privileged group.


```python
# Equal Opportunity Difference
equal_opportunity_difference = metric.equal_opportunity_difference()
print(f"Equal Opportunity Difference: {equal_opportunity_difference:.2f}")

print(explainer.equal_opportunity_difference())
```

    Equal Opportunity Difference: 0.19
    True positive rate difference (true positive rate on unprivileged instances - true positive rate on privileged instances): 0.19284858787621217



4. **Average Odds Difference**: The average difference in false positive rates (FPR) and true positive rates (TPR) between the unprivileged and privileged groups. 


$$
   \text{Average Odds Difference} = \frac{1}{2}[(\text{FPR(unprivileged)} - \text{FPR(privileged)}) + (\text{TPR(unprivileged)} - \text{TPR(privileged)})]
$$

- Value Close to 0: A value close to 0 suggests that the model’s TPR and FPR are similar for both groups, indicating fairness in terms of both positive and negative outcomes.
- Positive Value: A positive Average Odds Difference indicates that the unprivileged group has higher FPR and/or higher TPR compared to the privileged group.
- Negative Value: A negative Average Odds Difference indicates that the unprivileged group has lower FPR and/or lower TPR compared to the privileged group.


```python
# Average Odds Difference
average_odds_difference = metric.average_odds_difference()
print(f"Average Odds Difference: {average_odds_difference:.2f}")

print(explainer.average_odds_difference())
```

    Average Odds Difference: 0.25
    Average odds difference (average of TPR difference and FPR difference, 0 = equality of odds): 0.25029379501431104



##### **Metric Interpretation - Rules of Thumb**

| Metric                        | Definition                                                                                       | Interpretation                                                                                       | Threshold for Significant Bias                        | Bias Directionality                                            |
|-------------------------------|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------|------------------------------------------------------------------|
| **Disparate Impact**          | Ratio of favorable outcomes between unprivileged and privileged groups.                          | 1 implies perfect fairness.                                                                         | < 0.8 or > 1.25                                         | < 0.8: Against unprivileged<br> > 1.25: Against privileged       |
| **Statistical Parity Difference** | Difference in probability of favorable outcomes between unprivileged and privileged groups.     | 0 indicates similar favorable outcomes for both groups.                                              | < -0.1 or > 0.1                                         | < 0: Against unprivileged<br> > 0: Against privileged            |
| **Equal Opportunity Difference** | Difference in true positive rates between unprivileged and privileged groups.                   | 0 suggests equal positive prediction rates for both groups.                                          | < -0.1 or > 0.1                                         | < 0: Against unprivileged<br> > 0: Against privileged            |
| **Average Odds Difference**   | Average difference in false positive and true positive rates between unprivileged and privileged groups. | 0 indicates similar error rates across both groups.                                                 | < -0.1 or > 0.1                                         | < 0: Against unprivileged<br> > 0: Against privileged            |

By calculating these fairness metrics, we can assess the performance and fairness of our baseline logistic regression model. In the next section, we will apply the Reweighing technique to mitigate bias in the dataset and re-evaluate these metrics to observe the improvements in fairness.



```python
privileged_groups
```




    [{'Sex': 0}]



### Putting it All Together


```python
from aif360.metrics import ClassificationMetric

def evaluate_fairness(test, y_pred, unprivileged_groups, privileged_groups,
                      verbose=True, output_dict=False, convert_to_dataset=True, 
                      results_label=""):
    """
    Evaluates the fairness of a machine learning model's predictions.

    Parameters:
    - test (Dataset): The test dataset used for evaluation.
    - y_pred (array-like): The predicted labels for the test dataset.
    - unprivileged_groups (list of dicts): The unprivileged groups for fairness evaluation. e.g.,[{'Sex': 1}]
    - privileged_groups (list of dicts): The privileged groups for fairness evaluation.e.g., [{'Sex': 0}]
    - verbose (bool, optional): Whether to print the fairness metrics. Defaults to True.
    - output_dict (bool, optional): Whether to return the fairness metrics as a dictionary. Defaults to False.
    - convert_to_dataset (bool, optional): Whether to convert the test dataset to a new dataset with predictions. Defaults to True.
    - results_label (str, optional): A label for the fairness metrics. Defaults to "". Also used as value for "Label" key in results dict.

    Returns:
    - metrics_dict (dict, optional): A dictionary containing the fairness metrics if output_dict is True.

    """
    # Create a new dataset with predictions
    if convert_to_dataset:
        test_pred = test.copy()
        test_pred.labels = y_pred
    else:
        test_pred = y_pred

    # Evaluate fairness
    metric = ClassificationMetric(
        test,
        test_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    print(f"Fairness Metrics: {results_label}")
    
    metrics_dict = {
        "Disparate Impact": metric.disparate_impact(),
        "Statistical Parity Difference": metric.statistical_parity_difference(),
        "Equal Opportunity Difference": metric.equal_opportunity_difference(),
        "Average Odds Difference": metric.average_odds_difference()
    }
    
    if verbose:
        # Print metrics dict one by one
        for k, v in metrics_dict.items():
            print(f"- {k}: {v:.2f}")
        
    if output_dict:
        metrics_dict['Label'] = results_label
        return metrics_dict
```

#### Putting it all together - Baseline Model



```python
# Extract features and labels from the reweighed dataset
X_train = train.features
y_train = train.labels.ravel()

X_test = test.features
y_test = test.labels.ravel()

# Train the model
model =  LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Ealuate model performance
label = "Logistic Regression -  Unmitigated"
evaluate_model(y_test, model.predict(X_test), results_label=label)


# Define privileged and unprivileged groups
privileged_groups = [{'Sex': 0}]  # White
unprivileged_groups = [{'Sex': 1}]  # Non-White


# Evaluate model fairness
fairness_unmit = evaluate_fairness(test, model.predict(X_test), unprivileged_groups, privileged_groups,
                            verbose=True, output_dict=True, results_label=label)
```

    Classification Metrics: Logistic Regression -  Unmitigated


​    
                  precision    recall  f1-score   support
    
             0.0       0.74      0.57      0.64      4738
             1.0       0.48      0.67      0.56      2836
    
        accuracy                           0.61      7574
       macro avg       0.61      0.62      0.60      7574
    weighted avg       0.64      0.61      0.61      7574




​    
![png](output_48_1.png)
​    


    Fairness Metrics: Logistic Regression -  Unmitigated
    - Disparate Impact: 1.56
    - Statistical Parity Difference: 0.25
    - Equal Opportunity Difference: 0.19
    - Average Odds Difference: 0.25


##### Interpreting Our Baseline Model's Fairness


According to our rules of thumb:
- A Disparate Impact of 1.56 suggests significance bias against the privileged class.
- A Statistical Parity Difference of 0.25 indicates significant bias against the privileged class.
- An Equal Opportunity Difference of 0.19 indicated a significant bias against against the privileged class.
- An average odds difference of 0.25 indicate a significant bias against the priviledged class. 




### 2. Apply the Reweighing Algorithm to Address Bias



Next, we apply the Reweighing algorithm to the dataset. This will create a new dataset with weights assigned to each example to ensure fair representation.

`CONFIRM:` We can only mitigate the bias for one protected attribute at a time.


In part 1, we found that the dataset was more biased in terms of Sex than Race, so we will use Sex for our examples in this blog post. 


```python
from aif360.algorithms.preprocessing import Reweighing

# Define privileged and unprivileged groups
privileged_groups = [{'Sex': 0}]  # White
unprivileged_groups = [{'Sex': 1}]  # Non-White

# Apply Reweighing
RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
reweighed_train = RW.fit_transform(train)
reweighed_test = RW.transform(test)

```

We have now updated the values stored in the datasets under the `.instance_weights` methods based on the bias in the specified protected attribute.


```python
# Comparing original vs. reweighed training datasets
display(train.instance_weights[:5])
reweighed_train.instance_weights[:5]
```


    array([1., 1., 1., 1., 1.])





    array([1.01838057, 0.90648459, 1.01838057, 1.01838057, 0.97094552])



We can see that the weights have been adjusted to account for the bias in the training dataset.

#### 3. Train a Machine Learning Model

Using the reweighed dataset, we train a machine learning model. For simplicity, we'll use a logistic regression model. We will take advantage of the `sample_weight` argument in the `.fit()` method of the model to pass in the calculated weights.



##### Applying Reweighed Features


```python

# Extract features and labels from the reweighed dataset
X_train = reweighed_train.features
y_train = reweighed_train.labels.ravel()
# Extracting the reweighed sample weights
sample_weight = reweighed_train.instance_weights


X_test = reweighed_test.features
y_test = reweighed_test.labels.ravel()

# Train the model
label="Logistic Regression - Reweighed"
model =  LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
model.fit(X_train, y_train, sample_weight=sample_weight)
eval_reweigh = evaluate_model(y_test, model.predict(X_test), results_label=label, output_dict=True)


```

    Classification Metrics: Logistic Regression - Reweighed


​    
                  precision    recall  f1-score   support
    
             0.0       0.75      0.55      0.64      4738
             1.0       0.48      0.69      0.56      2836
    
        accuracy                           0.60      7574
       macro avg       0.61      0.62      0.60      7574
    weighted avg       0.65      0.60      0.61      7574




​    
![png](output_59_1.png)
​    



**4. Evaluate the Model Using Fairness Metrics**

Finally, we evaluate the model's performance and fairness using AIF360's metrics.



```python

fairness_reweigh = evaluate_fairness(reweighed_test, model.predict(X_test), unprivileged_groups, privileged_groups,
                  results_label=label, output_dict=True)
```

    Fairness Metrics: Logistic Regression - Reweighed
    - Disparate Impact: 0.88
    - Statistical Parity Difference: -0.06
    - Equal Opportunity Difference: -0.06
    - Average Odds Difference: -0.05


>Simplified Summary Table for Interpretation

| Metric                          | Interpretation                                                               | Threshold for Significant Bias |
|---------------------------------|------------------------------------------------------------------------------|--------------------------------|
| **Disparate Impact**            | 1 implies perfect fairness.                                                  | < 0.8 or > 1.25                |
| **Statistical Parity Difference** | 0 indicates similar favorable outcomes for both groups.                     | < -0.1 or > 0.1                |
| **Equal Opportunity Difference** | 0 suggests equal positive prediction rates for both groups.                 | < -0.1 or > 0.1                |
| **Average Odds Difference**     | 0 indicates similar error rates across both groups.                          | < -0.1 or > 0.1                |


After reweighing:
- Disparate impact is 0.88, which is above the 0.8 threshold, indicating that there is no significant bias.
- Statistical Parity Difference is -0.06, which is within the < -0.1 threshold, indicating no significant bias.
- Equal Opportunity Difference is -0.06, which is within the threshold, indicating no significant bias.
- Average Odds Difference is -0.05, whic is within the threshold, indicating no significant bias. 

Applying Reweighing was sufficient to eliminate the bias of our baseline model.


This example demonstrates how to apply the Reweighing technique to mitigate bias in your dataset before training a model. By ensuring fair representation in the training data, you can develop more equitable AI systems.

In the next section, we will explore in-processing techniques to mitigate bias during the training phase. Stay tuned!



## **B) Implementing In-Processing Techniques**



In-processing techniques modify the learning algorithm itself to reduce bias during model training. These methods integrate fairness considerations directly into the training process. In this section, we will focus on the Adversarial Debiasing algorithm.



### **Adversarial Debiasing**



Adversarial Debiasing uses adversarial learning to minimize bias. It trains the model to make accurate predictions while an adversary tries to detect bias based on the protected attributes. The goal is to improve the fairness of the model by making it difficult for the adversary to distinguish between different protected groups.




**3. Apply the Adversarial Debiasing Algorithm**

Next, we apply the Adversarial Debiasing algorithm to train the model.

>Note regarding Variable Scope Reuse: 
The tf.compat.v1.variable_scope is set with reuse=tf.compat.v1.AUTO_REUSE, allowing the reuse of variables within the same scope. This prevents the ValueError related to existing variables.




```python

from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow as tf


# Ensure compatibility with TensorFlow 2.x
tf.compat.v1.disable_eager_execution()

# Define the adversarial debiasing model
sess = tf.compat.v1.Session()
with tf.compat.v1.variable_scope('debiasing', reuse=tf.compat.v1.AUTO_REUSE):

    adversarial_debiasing = AdversarialDebiasing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        scope_name='debiasing',
        debias=True,
        sess=sess,
    )

# Train the model
adversarial_debiasing.fit(train)


```

    WARNING:tensorflow:From /opt/homebrew/Caskroom/miniforge/base/envs/fair-env/lib/python3.12/site-packages/tensorflow/python/util/dispatch.py:1260: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.


    2024-08-02 14:52:07.862117: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:388] MLIR V1 optimization pass is not enabled


    epoch 0; iter: 0; batch classifier loss: 1.612982; batch adversarial loss: 0.600901
    epoch 1; iter: 0; batch classifier loss: 0.775047; batch adversarial loss: 0.585992
    epoch 2; iter: 0; batch classifier loss: 0.726446; batch adversarial loss: 0.562675
    epoch 3; iter: 0; batch classifier loss: 1.033559; batch adversarial loss: 0.546327
    epoch 4; iter: 0; batch classifier loss: 0.932892; batch adversarial loss: 0.566724
    epoch 5; iter: 0; batch classifier loss: 0.902458; batch adversarial loss: 0.607745
    epoch 6; iter: 0; batch classifier loss: 0.755272; batch adversarial loss: 0.625006
    epoch 7; iter: 0; batch classifier loss: 0.741364; batch adversarial loss: 0.463662
    epoch 8; iter: 0; batch classifier loss: 0.674213; batch adversarial loss: 0.544812
    epoch 9; iter: 0; batch classifier loss: 0.639374; batch adversarial loss: 0.451837
    epoch 10; iter: 0; batch classifier loss: 0.654252; batch adversarial loss: 0.447203
    epoch 11; iter: 0; batch classifier loss: 0.639386; batch adversarial loss: 0.442813
    epoch 12; iter: 0; batch classifier loss: 0.710465; batch adversarial loss: 0.457174
    epoch 13; iter: 0; batch classifier loss: 0.675289; batch adversarial loss: 0.372009
    epoch 14; iter: 0; batch classifier loss: 0.719537; batch adversarial loss: 0.476921
    epoch 15; iter: 0; batch classifier loss: 0.656538; batch adversarial loss: 0.384876
    epoch 16; iter: 0; batch classifier loss: 0.612022; batch adversarial loss: 0.368515
    epoch 17; iter: 0; batch classifier loss: 0.620742; batch adversarial loss: 0.471665
    epoch 18; iter: 0; batch classifier loss: 0.618979; batch adversarial loss: 0.384866
    epoch 19; iter: 0; batch classifier loss: 0.722891; batch adversarial loss: 0.496701
    epoch 20; iter: 0; batch classifier loss: 0.559045; batch adversarial loss: 0.474010
    epoch 21; iter: 0; batch classifier loss: 0.562836; batch adversarial loss: 0.408687
    epoch 22; iter: 0; batch classifier loss: 0.655739; batch adversarial loss: 0.373444
    epoch 23; iter: 0; batch classifier loss: 0.663526; batch adversarial loss: 0.391315
    epoch 24; iter: 0; batch classifier loss: 0.606719; batch adversarial loss: 0.351055
    epoch 25; iter: 0; batch classifier loss: 0.714050; batch adversarial loss: 0.417550
    epoch 26; iter: 0; batch classifier loss: 0.618568; batch adversarial loss: 0.426313
    epoch 27; iter: 0; batch classifier loss: 0.642022; batch adversarial loss: 0.450222
    epoch 28; iter: 0; batch classifier loss: 0.585150; batch adversarial loss: 0.355106
    epoch 29; iter: 0; batch classifier loss: 0.620913; batch adversarial loss: 0.342492
    epoch 30; iter: 0; batch classifier loss: 0.629090; batch adversarial loss: 0.448611
    epoch 31; iter: 0; batch classifier loss: 0.588635; batch adversarial loss: 0.476278
    epoch 32; iter: 0; batch classifier loss: 0.591457; batch adversarial loss: 0.416445
    epoch 33; iter: 0; batch classifier loss: 0.628513; batch adversarial loss: 0.380825
    epoch 34; iter: 0; batch classifier loss: 0.637307; batch adversarial loss: 0.407390
    epoch 35; iter: 0; batch classifier loss: 0.661927; batch adversarial loss: 0.418905
    epoch 36; iter: 0; batch classifier loss: 0.575842; batch adversarial loss: 0.420911
    epoch 37; iter: 0; batch classifier loss: 0.623764; batch adversarial loss: 0.427711
    epoch 38; iter: 0; batch classifier loss: 0.650178; batch adversarial loss: 0.451093
    epoch 39; iter: 0; batch classifier loss: 0.575283; batch adversarial loss: 0.473690
    epoch 40; iter: 0; batch classifier loss: 0.617539; batch adversarial loss: 0.316627
    epoch 41; iter: 0; batch classifier loss: 0.666945; batch adversarial loss: 0.378264
    epoch 42; iter: 0; batch classifier loss: 0.655537; batch adversarial loss: 0.495861
    epoch 43; iter: 0; batch classifier loss: 0.594885; batch adversarial loss: 0.473660
    epoch 44; iter: 0; batch classifier loss: 0.615691; batch adversarial loss: 0.373581
    epoch 45; iter: 0; batch classifier loss: 0.577053; batch adversarial loss: 0.418978
    epoch 46; iter: 0; batch classifier loss: 0.592739; batch adversarial loss: 0.445270
    epoch 47; iter: 0; batch classifier loss: 0.609106; batch adversarial loss: 0.364682
    epoch 48; iter: 0; batch classifier loss: 0.616219; batch adversarial loss: 0.402312
    epoch 49; iter: 0; batch classifier loss: 0.590119; batch adversarial loss: 0.457990





    <aif360.algorithms.inprocessing.adversarial_debiasing.AdversarialDebiasing at 0x33ebd3080>




**4. Make Predictions and Create Predicted Dataset**

Make predictions on the test set using the trained Adversarial Debiasing model.



```python

# Make predictions on the test set
pred_dataset = adversarial_debiasing.predict(test)

# Convert predictions to labels
y_pred = pred_dataset.labels

label = "Adversarial Debiasing"
eval_adv_debias = evaluate_model(test.labels, pred_dataset.labels, results_label=label, output_dict=True)
```

    Classification Metrics: Adversarial Debiasing


​    
                  precision    recall  f1-score   support
    
             0.0       0.66      0.91      0.77      4738
             1.0       0.59      0.22      0.32      2836
    
        accuracy                           0.65      7574
       macro avg       0.63      0.56      0.54      7574
    weighted avg       0.63      0.65      0.60      7574




​    
![png](output_72_1.png)
​    



**5. Evaluate the Model Using Fairness Metrics**

Finally, we evaluate the model's performance and fairness using AIF360's metrics.



```python
fairness_adv_debias =  evaluate_fairness(test, pred_dataset, unprivileged_groups, privileged_groups,
                                         convert_to_dataset=False, output_dict=True, results_label=label)
```

    Fairness Metrics: Adversarial Debiasing
    - Disparate Impact: 1.14
    - Statistical Parity Difference: 0.12
    - Equal Opportunity Difference: 0.08
    - Average Odds Difference: 0.13


We can see that in-progressing with the Adversarial Debiasing Algorithm was not sufficient to reduce the bias to non-significant values.

## C) Implementing Post-Processing Techniques



Post-processing techniques adjust the predictions of a trained model to reduce bias. These methods are applied after the model has been trained and focus on modifying the model’s outputs to ensure fairness. In this section, we will focus on the Equalized Odds Post-Processing algorithm.



### Equalized Odds Post-Processing



The Equalized Odds Post-Processing algorithm adjusts the model’s predictions to ensure that the false positive rate and true positive rate are equal across different groups. This technique ensures that the model’s performance is consistent for both the privileged and unprivileged groups.

Step-by-Step Guide: Equalized Odds Post-Processing

1. Train a Baseline Machine Learning Model

We have already trained a baseline logistic regression model and created the train and test datasets.




2. Make Predictions on the Test Set

Make predictions on the test set using the trained model.



```python
# Get predictions for test set and convert to dataset object

y_pred = model.predict(X_test)

```


3. Convert Predictions to AIF360 BinaryLabelDataset

Convert the test set and predictions into AIF360 BinaryLabelDataset objects.


```python

# Create a new dataset with predictions
test_pred = test.copy()
test_pred.labels = y_pred.reshape(-1, 1)
```

4. Apply the Equalized Odds Post-Processing Algorithm

Apply the Equalized Odds Post-Processing algorithm to adjust the model’s predictions.


```python
from aif360.algorithms.postprocessing import EqOddsPostprocessing
# Apply Equalized Odds Post-Processing
eq_odds = EqOddsPostprocessing(
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups,
    seed=42
)

```


```python

# Fit the post-processing algorithm to the test data
eq_odds.fit(test, test_pred)


# Transform the predictions
# pred_transformed = eq_odds.transform(test_pred)
pred_transformed = eq_odds.predict(test_pred)
```


```python
# Evaluate model performance using the transformed predictions
label = "Logistic Regression - EqOdds"
eval_eqodds = evaluate_model(test.labels, pred_transformed.labels, results_label=label, output_dict=True)
```

    Classification Metrics: Logistic Regression - EqOdds


​    
                  precision    recall  f1-score   support
    
             0.0       0.75      0.51      0.60      4738
             1.0       0.46      0.71      0.56      2836
    
        accuracy                           0.58      7574
       macro avg       0.60      0.61      0.58      7574
    weighted avg       0.64      0.58      0.59      7574




​    
![png](output_87_1.png)
​    


As we can see, the model performance has decreased after transforming the model predictions. Let's asses fairness now to see if we've reduced the bias from our original model. 


```python
fairness_eqodds = evaluate_fairness(test, pred_transformed, unprivileged_groups, privileged_groups, 
                                    convert_to_dataset=False, output_dict=True, results_label=label)
```

    Fairness Metrics: Logistic Regression - EqOdds
    - Disparate Impact: 1.04
    - Statistical Parity Difference: 0.02
    - Equal Opportunity Difference: 0.01
    - Average Odds Difference: -0.00


By applying the Equalized Odds Post-Processing algorithm, we were able to adjust the model’s predictions to ensure fairness across different groups. After applying this technique, we re-evaluated the model’s fairness using key metrics to determine the effectiveness of the post-processing adjustments and confirmed that the model's fairness metrics are all inidcating no significant bias.

### **5. Comparing Mitigation Techniques**

In this section, we will compare the effectiveness of the different bias mitigation techniques we have implemented: pre-processing (Reweighing), in-processing (Adversarial Debiasing), and post-processing (Equalized Odds Post-Processing). We will use fairness metrics to evaluate each technique's impact on the model and discuss the trade-offs between accuracy and fairness.



### **Comparative Analysis**




```python
# Combine the fairness metrics into a DataFrame
df_fairness = pd.DataFrame([fairness_unmit, fairness_reweigh, fairness_adv_debias, fairness_eqodds])
df_fairness = df_fairness.set_index('Label')
df_fairness

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Disparate Impact</th>
      <th>Statistical Parity Difference</th>
      <th>Equal Opportunity Difference</th>
      <th>Average Odds Difference</th>
    </tr>
    <tr>
      <th>Label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Logistic Regression -  Unmitigated</th>
      <td>1.557575</td>
      <td>0.247677</td>
      <td>0.192849</td>
      <td>0.250294</td>
    </tr>
    <tr>
      <th>Logistic Regression - Reweighed</th>
      <td>0.882272</td>
      <td>-0.055594</td>
      <td>-0.064862</td>
      <td>-0.054568</td>
    </tr>
    <tr>
      <th>Adversarial Debiasing</th>
      <td>1.144927</td>
      <td>0.122611</td>
      <td>0.082567</td>
      <td>0.130413</td>
    </tr>
    <tr>
      <th>Logistic Regression - EqOdds</th>
      <td>1.041196</td>
      <td>0.017411</td>
      <td>0.005085</td>
      <td>-0.003514</td>
    </tr>
  </tbody>
</table>
</div>



To ease interpretation, we will use Pandas Styling to annotate any metrics that have crossed the fairness thresholds.

>*Simplified Summary Table for Interpretation*

| Metric                          | Interpretation                                                               | Threshold for Significant Bias |
|---------------------------------|------------------------------------------------------------------------------|--------------------------------|
| **Disparate Impact**            | 1 implies perfect fairness.                                                  | < 0.8 or > 1.25                |
| **Statistical Parity Difference** | 0 indicates similar favorable outcomes for both groups.                     | < -0.1 or > 0.1                |
| **Equal Opportunity Difference** | 0 suggests equal positive prediction rates for both groups.                 | < -0.1 or > 0.1                |
| **Average Odds Difference**     | 0 indicates similar error rates across both groups.                          | < -0.1 or > 0.1                |



```python
# Define styling functions
def style_di(val):
    """Style function for 'Disparate Impact' column centered around 1."""
    if (val > 1.2) or (val < 0.8):
        return 'color: red;'
    else:
        return ''

def style_diff(val):
    """Style function for difference columns centered around 0."""
    if (val > 0.1) or (val < -0.1):
        return 'color: red;'
    else:
        return ''
    
# Apply styles
styled_df = df_fairness.style.map(style_di, subset=['Disparate Impact']) \
                            .map(style_diff, subset=['Statistical Parity Difference', 
                                                          'Equal Opportunity Difference', 
                                                          'Average Odds Difference']) \
                            .format('{:.2f}') \
                            .set_caption("Comparison of Mitigation Technique Fairness Metrics")

styled_df

```




<style type="text/css">
#T_cccdd_row0_col0, #T_cccdd_row0_col1, #T_cccdd_row0_col2, #T_cccdd_row0_col3, #T_cccdd_row2_col1, #T_cccdd_row2_col3 {
  color: red;
}
</style>
<table id="T_cccdd">
  <caption>Comparison of Mitigation Technique Fairness Metrics</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_cccdd_level0_col0" class="col_heading level0 col0" >Disparate Impact</th>
      <th id="T_cccdd_level0_col1" class="col_heading level0 col1" >Statistical Parity Difference</th>
      <th id="T_cccdd_level0_col2" class="col_heading level0 col2" >Equal Opportunity Difference</th>
      <th id="T_cccdd_level0_col3" class="col_heading level0 col3" >Average Odds Difference</th>
    </tr>
    <tr>
      <th class="index_name level0" >Label</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_cccdd_level0_row0" class="row_heading level0 row0" >Logistic Regression -  Unmitigated</th>
      <td id="T_cccdd_row0_col0" class="data row0 col0" >1.56</td>
      <td id="T_cccdd_row0_col1" class="data row0 col1" >0.25</td>
      <td id="T_cccdd_row0_col2" class="data row0 col2" >0.19</td>
      <td id="T_cccdd_row0_col3" class="data row0 col3" >0.25</td>
    </tr>
    <tr>
      <th id="T_cccdd_level0_row1" class="row_heading level0 row1" >Logistic Regression - Reweighed</th>
      <td id="T_cccdd_row1_col0" class="data row1 col0" >0.88</td>
      <td id="T_cccdd_row1_col1" class="data row1 col1" >-0.06</td>
      <td id="T_cccdd_row1_col2" class="data row1 col2" >-0.06</td>
      <td id="T_cccdd_row1_col3" class="data row1 col3" >-0.05</td>
    </tr>
    <tr>
      <th id="T_cccdd_level0_row2" class="row_heading level0 row2" >Adversarial Debiasing</th>
      <td id="T_cccdd_row2_col0" class="data row2 col0" >1.14</td>
      <td id="T_cccdd_row2_col1" class="data row2 col1" >0.12</td>
      <td id="T_cccdd_row2_col2" class="data row2 col2" >0.08</td>
      <td id="T_cccdd_row2_col3" class="data row2 col3" >0.13</td>
    </tr>
    <tr>
      <th id="T_cccdd_level0_row3" class="row_heading level0 row3" >Logistic Regression - EqOdds</th>
      <td id="T_cccdd_row3_col0" class="data row3 col0" >1.04</td>
      <td id="T_cccdd_row3_col1" class="data row3 col1" >0.02</td>
      <td id="T_cccdd_row3_col2" class="data row3 col2" >0.01</td>
      <td id="T_cccdd_row3_col3" class="data row3 col3" >-0.00</td>
    </tr>
  </tbody>
</table>




- Equal Odds Postprocessing produced the most fair model predictions.
- Reweighing produced the second most fair model predictions.
- Adversarial Deibiasing was still biased.



**Trade-Offs Between Accuracy and Fairness**


Let's compare the performance metrics of the 3 mitigation approaches to see the trade-off between fairness and performance.


```python
# Demonstrating slices results from eval dicts
{'accuracy':eval_unmit['accuracy'], 
 "avg recall":eval_unmit["macro avg"]['recall'],
 "recall 1.0":eval_unmit["1.0"]['recall'],
 "recall 0.0":eval_unmit["0.0"]['recall']}
```




    {'accuracy': 0.6062846580406654,
     'avg recall': 0.6188526310399787,
     'recall 1.0': 0.6688998589562765,
     'recall 0.0': 0.5688054031236809}




```python
# Convert the evaluation results to a DataFrame
eval_dicts = [eval_unmit, eval_reweigh, eval_adv_debias, eval_eqodds]
results_eval = []
for d in eval_dicts:
    results_eval.append({"Label":d['Label'], # Model Label
                         "Recall(1.0)":d["1.0"]['recall'], # Recall for class 1
                         "Recall(0.0)":d["0.0"]['recall'], # Recall for class 0
                         "Average Recall":d["macro avg"]['recall'], # Average Recall
                         "Average Precision":d["macro avg"]['precision'], # Average Precision
                         'Average F1-Score':d["macro avg"]['f1-score'], # Average F1 Score
                        'Accuracy':d['accuracy'],  # Accuracy
                         }
                        )
df_eval = pd.DataFrame(results_eval).set_index('Label')
df_eval.style.background_gradient(#subset=['Accuracy','Recall(1.0)','Recall(0.0)', "Average F1-Score"],
                                  cmap='Greens', axis=0).format("{:.2f}").set_caption("Comparison of Model Performance")
```




<style type="text/css">
#T_b10cb_row0_col0 {
  background-color: #005f26;
  color: #f1f1f1;
}
#T_b10cb_row0_col1 {
  background-color: #def2d9;
  color: #000000;
}
#T_b10cb_row0_col2 {
  background-color: #00451c;
  color: #f1f1f1;
}
#T_b10cb_row0_col3 {
  background-color: #a7dba0;
  color: #000000;
}
#T_b10cb_row0_col4, #T_b10cb_row1_col2, #T_b10cb_row2_col1, #T_b10cb_row2_col3, #T_b10cb_row2_col5, #T_b10cb_row3_col0 {
  background-color: #00441b;
  color: #f1f1f1;
}
#T_b10cb_row0_col5 {
  background-color: #aadda4;
  color: #000000;
}
#T_b10cb_row1_col0 {
  background-color: #005522;
  color: #f1f1f1;
}
#T_b10cb_row1_col1 {
  background-color: #e7f6e2;
  color: #000000;
}
#T_b10cb_row1_col3 {
  background-color: #9bd696;
  color: #000000;
}
#T_b10cb_row1_col4 {
  background-color: #005020;
  color: #f1f1f1;
}
#T_b10cb_row1_col5 {
  background-color: #bae3b3;
  color: #000000;
}
#T_b10cb_row2_col0, #T_b10cb_row2_col2, #T_b10cb_row2_col4, #T_b10cb_row3_col1, #T_b10cb_row3_col3, #T_b10cb_row3_col5 {
  background-color: #f7fcf5;
  color: #000000;
}
#T_b10cb_row3_col2 {
  background-color: #127c39;
  color: #f1f1f1;
}
#T_b10cb_row3_col4 {
  background-color: #359e53;
  color: #f1f1f1;
}
</style>
<table id="T_b10cb">
  <caption>Comparison of Model Performance</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b10cb_level0_col0" class="col_heading level0 col0" >Recall(1.0)</th>
      <th id="T_b10cb_level0_col1" class="col_heading level0 col1" >Recall(0.0)</th>
      <th id="T_b10cb_level0_col2" class="col_heading level0 col2" >Average Recall</th>
      <th id="T_b10cb_level0_col3" class="col_heading level0 col3" >Average Precision</th>
      <th id="T_b10cb_level0_col4" class="col_heading level0 col4" >Average F1-Score</th>
      <th id="T_b10cb_level0_col5" class="col_heading level0 col5" >Accuracy</th>
    </tr>
    <tr>
      <th class="index_name level0" >Label</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b10cb_level0_row0" class="row_heading level0 row0" >Unmitigated Model</th>
      <td id="T_b10cb_row0_col0" class="data row0 col0" >0.67</td>
      <td id="T_b10cb_row0_col1" class="data row0 col1" >0.57</td>
      <td id="T_b10cb_row0_col2" class="data row0 col2" >0.62</td>
      <td id="T_b10cb_row0_col3" class="data row0 col3" >0.61</td>
      <td id="T_b10cb_row0_col4" class="data row0 col4" >0.60</td>
      <td id="T_b10cb_row0_col5" class="data row0 col5" >0.61</td>
    </tr>
    <tr>
      <th id="T_b10cb_level0_row1" class="row_heading level0 row1" >Logistic Regression - Reweighed</th>
      <td id="T_b10cb_row1_col0" class="data row1 col0" >0.69</td>
      <td id="T_b10cb_row1_col1" class="data row1 col1" >0.55</td>
      <td id="T_b10cb_row1_col2" class="data row1 col2" >0.62</td>
      <td id="T_b10cb_row1_col3" class="data row1 col3" >0.61</td>
      <td id="T_b10cb_row1_col4" class="data row1 col4" >0.60</td>
      <td id="T_b10cb_row1_col5" class="data row1 col5" >0.60</td>
    </tr>
    <tr>
      <th id="T_b10cb_level0_row2" class="row_heading level0 row2" >Adversarial Debiasing</th>
      <td id="T_b10cb_row2_col0" class="data row2 col0" >0.22</td>
      <td id="T_b10cb_row2_col1" class="data row2 col1" >0.91</td>
      <td id="T_b10cb_row2_col2" class="data row2 col2" >0.56</td>
      <td id="T_b10cb_row2_col3" class="data row2 col3" >0.63</td>
      <td id="T_b10cb_row2_col4" class="data row2 col4" >0.54</td>
      <td id="T_b10cb_row2_col5" class="data row2 col5" >0.65</td>
    </tr>
    <tr>
      <th id="T_b10cb_level0_row3" class="row_heading level0 row3" >Logistic Regression - EqOdds</th>
      <td id="T_b10cb_row3_col0" class="data row3 col0" >0.71</td>
      <td id="T_b10cb_row3_col1" class="data row3 col1" >0.51</td>
      <td id="T_b10cb_row3_col2" class="data row3 col2" >0.61</td>
      <td id="T_b10cb_row3_col3" class="data row3 col3" >0.60</td>
      <td id="T_b10cb_row3_col4" class="data row3 col4" >0.58</td>
      <td id="T_b10cb_row3_col5" class="data row3 col5" >0.58</td>
    </tr>
  </tbody>
</table>





1. **Reweighed Model**: This technique adjusts the dataset before training, which can lead to improved fairness but might slightly reduce the model's accuracy. It's a good balance for scenarios where data preprocessing is feasible. 
- Indeed, we saw a very slight decrease in accuracy from 0.61 to 0.60. However, our average recall, precision, and f1-score are unchanged.

2. **Adversarial Debiasing Model**: This technique incorporates fairness constraints directly into the training process, often achieving significant fairness improvements. However, it can be computationally intensive and might require more tuning.
- Without additional tuning Adversarial Debiasing is the most accurate model (0.65), but has terrible Recall for the 0 class (0.28),and the worse F1-score. 

3. **Equalized Odds Post-Processing Model**: This technique adjusts the model's predictions after training, which can be simpler to implement and often leads to noticeable fairness improvements. However, it might introduce a trade-off with accuracy depending on the degree of adjustment required. 
- EqOdds did indeed significantly decrease the accuracy by 3% (0.61 to 0.58), however it actually improved the model's recall for the 1.0 class. 


### **6. Conclusion**

In this series, we have explored the critical aspects of bias in machine learning models and the importance of fairness in AI systems. We started with understanding and detecting bias using AI Fairness 360, then moved on to implementing various bias mitigation techniques, including pre-processing, in-processing, and post-processing methods. Here's a brief recap and the key takeaways from our journey:

#### **Recap of Bias Mitigation Techniques**

1. **Pre-Processing Techniques (Reweighing)**
   - Modify the training data to reduce bias before model training.
   - Ensures fair representation in the dataset, leading to more balanced model outcomes.

2. **In-Processing Techniques (Adversarial Debiasing)**
   - Incorporate fairness constraints directly into the training process.
   - Aim to make the model's predictions less distinguishable based on protected attributes.

3. **Post-Processing Techniques (Equalized Odds Post-Processing)**
   - Adjust the model's predictions after training to ensure fairness.
   - Focus on balancing false positive and true positive rates across different groups.

#### **Key Takeaways**

- **Understanding Bias**: Recognizing the different types of bias (data bias, algorithmic bias, and societal bias) is the first step towards building fair AI systems.
- **Detecting Bias**: Utilizing fairness metrics such as Disparate Impact, Statistical Parity Difference, Equal Opportunity Difference, and Average Odds Difference to evaluate model fairness.
- **Mitigating Bias**: Implementing bias mitigation techniques at different stages of the machine learning pipeline to reduce unfairness and improve equity.
- **Comparing Techniques**: Assessing the effectiveness of each mitigation technique and understanding the trade-offs between accuracy and fairness.

#### **Importance of Continuous Monitoring**

Fairness in AI is not a one-time effort but a continuous process. It is crucial to regularly monitor and evaluate the fairness of AI models as new data and scenarios emerge. By doing so, we can ensure that AI systems remain equitable and do not inadvertently perpetuate biases.

#### **Future Directions**

As AI technology continues to evolve, so will the approaches to ensuring fairness. Future advancements may include more sophisticated bias detection and mitigation techniques, integration of fairness considerations into all stages of AI development, and broader adoption of fairness standards and regulations.

### **Call to Action**

- **Implement Bias Mitigation**: Apply the techniques discussed in this series to your own projects to develop fairer AI models.
- **Engage in Discussions**: Join the conversation on AI fairness by sharing your experiences, challenges, and solutions on social media or in professional forums.
- **Explore Further**: Continue learning about AI fairness through additional resources, research papers, and case studies.

By prioritizing fairness in AI, we can build systems that not only perform well but also promote equity and justice in society. Together, we can make a significant impact on the future of AI.

Thank you for following along with this series. Stay tuned for the next part, where we will delve into advanced topics or case studies to further illustrate the application of AI Fairness 360 in real-world scenarios.

# APPENDIX/CUT


```python
raise Exception("End of blog post")
```


    ---------------------------------------------------------------------------
    
    Exception                                 Traceback (most recent call last)
    
    Cell In[33], line 1
    ----> 1 raise Exception("End of blog post")


    Exception: End of blog post




## Modeling with AIF360


```python
train, test = binary_dataset.split([0.8], shuffle=True)
len(train.labels), len(test.labels)

```


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
```


```python

def evaluate_model(y_true, y_pred):
    """Minimal evaluation of a model's performance."""
    print(classification_report(y_true, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize='true',cmap='Greens',
                                            display_labels=['Non-Recid','Recid'])
```


```python
# Training Model #1
clf_rf = RandomForestClassifier(
                             max_depth=12,
                             random_state=42, 
                             class_weight='balanced')
clf_rf.fit(train.features, train.labels.ravel())

# Evaluate model
y_hat_test_rf = clf_rf.predict(test.features)
evaluate_model(test.labels, y_hat_test_rf)
```


```python
# Training Model #2
# Logistic Regression
clf_logreg = LogisticRegression(max_iter=2000, 
                                random_state=42,
                             class_weight='balanced'
                             )
clf_logreg.fit(train.features, train.labels.ravel())

# Evaluate model
y_hat_test_logreg = clf_logreg.predict(test.features)
evaluate_model(test.labels, y_hat_test_logreg)
```


```python
# Training Model #3
from sklearn.svm import SVC
clf_svc = SVC(#probability=True, 
          kernel='rbf', class_weight='balanced')
clf_svc.fit(train.features, train.labels.ravel())

# Evaluate model
y_hat_test_svc = clf_svc.predict(test.features)
evaluate_model(test.labels, y_hat_test_svc) 
```


```python
import xgboost as xgb

# Training Model #1
clf_xgb = xgb.XGBClassifier(
    random_state=42,
    scale_pos_weight=(1 - train.labels.mean()) / train.labels.mean() # Mimic class_weight='balanced'
)
clf_xgb.fit(train.features, train.labels.ravel())

# Evaluate model
y_hat_test_xgb = clf_xgb.predict(test.features)
evaluate_model(test.labels, y_hat_test_xgb)
```

### ANN


```python
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# # from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# # Define the neural network model
# def create_model():
#     model = Sequential()
#     model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
    
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model




# # Make the data compatible with the model
# X_train, y_train = train.features, train.labels.ravel()
# X_test, y_test = test.features, test.labels.ravel()


# # Wrap the model using KerasClassifier for compatibility with scikit-learn
# model = create_model()
# # Train the model
# history = model.fit(X_train, y_train, validation_split=0.2, verbose=True, epochs=50)

```


```python
# y_hat_test = model.predict(X_test)
# y_hat_test = (y_hat_test > 0.5).astype(int)
# y_hat_test = y_hat_test.ravel()

# evaluate_model(y_test, y_hat_test)
```
