import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame([
                   ['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']

size_mapping = {
                'XL': 3,
                'L': 2,
                'M' : 1}
class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
"""ohe = OneHotEncoder(categorical_features=[0])
ohe_data = ohe.fit_transform(X).toarray()
print(ohe_data)"""

print(X)
df['classlabel'] = df['classlabel'].map(class_mapping)

print(class_mapping)
print(df)
print(pd.get_dummies(df[['price', 'color', 'size']]))