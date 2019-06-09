#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'analysis'))
	print(os.getcwd())
except:
	pass

#%% wczytanie paczek
import pandas as pd
import numpy as np

#%%
#useful functions

def writeList(where, what) :
    with open(where, 'w+') as file :
        for i in what :
            file.write("%s\n" % i)

def readList(where) :
    res = []
    
    with open(where, 'r') as file : 
        for i in file :
            res.append(i[:-1])
    
    return res


#%%
klin = pd.read_csv('/Users/empro/Documents/HR/contests/bioHack/analysis/final/klin.csv', index_col = 0)

merged = pd.read_csv('/Users/empro/Documents/HR/contests/bioHack/dataSets/merged.csv', index_col = 0)
mergedTest = pd.read_csv('/Users/empro/Documents/HR/contests/bioHack/dataSets/mergedTest.csv ', index_col = 0)

weka = readList('/Users/empro/Documents/HR/contests/bioHack/dataSets/weka1.txt')

#%%
klinTrain = klin.loc[list(merged.index), :]
klinTest = klin.loc[list(mergedTest.index), :]

merged = pd.concat([klinTrain['mutation'], merged], axis = 1)
mergedTest = pd.concat([klinTest['mutation'], mergedTest], axis = 1)

weka = ['mutation'] + weka


#%% tpot na szybkosci
from tpot import TPOTClassifier

hope = TPOTClassifier(generations=20, population_size=50, cv=10,
                                    random_state=42, verbosity=2)


#%%
hope.fit(merged.loc[:, weka], merged.loc[:, 'label'])

#%%
