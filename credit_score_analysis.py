# -*- coding: utf-8 -*-
"""DWDM2

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yCpAAGPYfG65EVO9BRDntuxjYpHxg5St
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn
import seaborn as sns
# %matplotlib inline
from scipy import stats

data = pd.read_csv("CreditScoring(After_Preprocessing).csv")
data.head()

from sklearn.preprocessing import LabelEncoder
model = LabelEncoder()
data['Status'] = model.fit_transform(data['Status'].astype('str'))
data['Home'] = model.fit_transform(data['Home'].astype('str'))
data['Marital'] = model.fit_transform(data['Marital'].astype('str'))
data['Job'] = model.fit_transform(data['Job'].astype('str'))
data['Records'] = model.fit_transform(data['Records'].astype('str'))

X = list( data.columns )
X.remove( 'Status' )
X

Y = data['Status']

credit_data = pd.get_dummies( data[X], drop_first = True )
len( credit_data.columns )

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( data[X], Y, test_size = 0.3, random_state = 42 )

import statsmodels.api as sm
logit = sm.Logit( y_train, sm.add_constant( X_train ) )
lg = logit.fit()

import pickle

pickle.dump(lg, open('model1.pkl','wb'))

model1=pickle.load(open('model1.pkl','rb'))

import numpy as np
import statsmodels.api as sm
# int_features = [1.0,3,3,60,26,3,0,0,35,83,0,0,1050,1156,90.83,7.42]
#int_features = [1.0,9,1,60,30,2,1,3,73,129,0,0,800,846,94.56264775,4.2] --> good aana chahiye 0.18 coming
#int_features = [1.0, 17,	1,	60,	58,	3,	1,	1,	48,	131,	0,	0,	1000,	1658, 60.31363088,	4.98] --> good aana chahiye, 0.631 coming
#int_features = [1.0,10,2,36,46,2,2,3,90,200,3000,0,2000,2985,67.00167504,1.98] --> bad aana chahiye, 0.026 coming
final_features = [int_features]
prediction = model1.predict( sm.add_constant(final_features))
#[0].predicted_prob.map( lambda x: 1 if x > 0.5 else 0)

prediction[0]

