#%% initialization
import pandas as pd
import numpy as np


#%% wczytywanie danych
dataPref = '../data/predicting_response/'

covar = pd.read_csv('data/predicting_response/X_covariates.tsv', sep = '\t')
label = pd.read_csv('data/predicting_response/y.tsv', sep = '\t', header = None, names = ['label'])


#%% Podziel dataset na podgrupy

from sklearn.model_selection import train_test_split 

xTrain, xTest, yTrain, yTest = train_test_split(covar, label, test_size = 0.25, random_state = 42)

mergedKlin = pd.concat([yTrain, xTrain], axis = 1)
mergedKlinTest = pd.concat([yTest, xTest], axis = 1)

#%%
intKlin = pd.concat([mergedKlin, mergedKlinTest]).iloc[:, 0:3]

#%% [markdown]
# ## podejscie na calym 

#%% chcemy esktrapolowac wartosci z wiekszych brakow do mniejszych 

mergedFull = pd.concat([label, covar], axis = 1)

#%% start 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


#%%
intKlin.columns = ['label', 'mutation', 'neoantigen']
subTrening = intKlin.loc[(intKlin['mutation'].isna() == False) & (intKlin['neoantigen'].isna() == False), :]


#%%
imp = IterativeImputer(max_iter = 100, random_state = 42)
res = imp.fit_transform(intKlin.iloc[:, 1:])

newMut = [i[0] for i in res]
intKlin['mutation'] = newMut

intKlin.to_csv('/Users/empro/Documents/HR/contests/bioHack/analysis/final/klin.csv')
#%% 
from sklearn.linear_model import LinearRegression

compl = LinearRegression()
compl.fit(subTrening.iloc[:, 1:], subTrening.iloc[:, 0])