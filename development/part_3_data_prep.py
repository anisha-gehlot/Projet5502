# -*- coding: utf-8 -*-
"""Part 3 - Data Prep.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XE3IEmLK5lWpOcGfZyBA7qTiWagvRFGs

# Part 3 - Data Prep

https://www.udemy.com/course/feature-engineering-for-machine-learning

* Types and characteristics of data
* Missing data imputation
* Categorical encoding
* Variable transformation
* Discretization
* Outliers
* Datetime
* Scaling
* Feature creation

## Load Data
"""

import pandas as pd

df = pd.read_csv('.../created_raw_data.csv')
print(df.shape)
print(df.info())
df.head()

import preppy.utils as utils
from preppy.version import __version__

print(__version__)

utils.report.write_report(df, thresh=.5)

import preppy.utils as preppy

consts = preppy.functions.identify_consts(df)
quasi_consts = preppy.functions.identify_quasi_consts(df)
duplicates = preppy.functions.check_col_duplicates(df)
print(duplicates)
print(consts)
print(quasi_consts)

# numeric_df = df.apply(pd.to_numeric, errors='coerce')
all_deletes = list(set(consts + quasi_consts + duplicates))
for col in all_deletes:
  print(col, df[col].dtype)
  if df[col].dtype in ['float64', 'int64']:
    df_numerical.remove(col)
  elif df[col].dtype in ['object']:
    df_object.remove(col)
    df_categorical_features.remove(col)
  else:
    df_discreet.remove(col)

"""## PrepPy Pipeline"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import preppy.utils as preppy

pipe = [
    ('constants', preppy.classes.RemoveConstants()),
    ('quasiconsts', preppy.classes.RemoveQuasiConstants(thresh=0.8)),
    ('duplicates', preppy.classes.DropDuplicates()),
    ('missing', preppy.classes.HandleMissingValues()),
    # ('encoding', HandleCatEncodeing())
]

pipe_model = Pipeline(pipe)
data = pipe_model.fit_transform(df)
cols = [col for col in df.columns if col not in consts + quasi_consts + duplicates]
nu_df = pd.DataFrame(data, columns=cols)
nu_df.info()

import pickle

# Load the pickled variable from the file
with open('.../var_types.pkl', 'rb') as f:
    var_types = pickle.load(f)

print(var_types)

df_numerical = var_types['df_numerical']
df_object = var_types['df_object']
df_discreet = var_types['df_discreet']
df_categorical_features = var_types['df_categorical_features']

# code along
nu_df[df_numerical] = nu_df[df_numerical].astype(float)
nu_df.info()

"""## Feature Engineering

### Feature Combination
"""

# create a new variable by combining two variables
df['scaling_combined'] = df['standard scaling'] + df['min max scaling']
df.drop(['standard scaling', 'min max scaling'], axis=1, inplace=True)

"""### Categorical Encoding"""

# code along
import preppy.utils as utils

df = utils.functions.do_OHE(df)
df.info()

df.to_csv('.../prepared_data.csv', index=False)