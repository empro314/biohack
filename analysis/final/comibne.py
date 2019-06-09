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
klin = readList('/Users/empro/Documents/HR/contests/bioHack/analysis/final/newList')
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
hope.export('tmp.py')

#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

res = make_pipeline(
    StackingEstimator(estimator=MultinomialNB(alpha=1.0, fit_prior=True)),
    LogisticRegression(C=10.0, dual=False, penalty="l2")
)
#%%
res.fit(merged.loc[:, 'mutation'], merged.loc[:, 'label'])
results = hope.predict_proba(mergedTest.loc[:, weka])

#%%

results = [i[1] for i in results]


#%%auc

from sklearn.metrics import roc_auc_score

yProba =results

score = roc_auc_score(mergedTest.loc[:, 'label'], yProba)


#%%

mergedKlin = pd.read_csv('/Users/empro/Documents/HR/contests/bioHack/analysis/final/mergedKlin.csv', index_col = 0)
mergedKlinTest = pd.read_csv('/Users/empro/Documents/HR/contests/bioHack/analysis/final/mergedKlinTest.csv', index_col = 0)

finalList = weka + klin

merged = pd.concat([merged, mergedKlin], axis = 1)
mergedTest = pd.concat([mergedTest, mergedKlinTest], axis = 1)

#%% new weka
from tpot import TPOTClassifier

hope = TPOTClassifier(generations=20, population_size=50, cv=10,
                                    random_state=42, verbosity=2)


#%%
hope.fit(merged.loc[:, finalList], merged.loc[:, 'label'].iloc[:, 0])

#%%list2
list2 = weka + ['klin1']

from tpot import TPOTClassifier

hope = TPOTClassifier(generations=20, population_size=50, cv=10,
                                    random_state=42, verbosity=2)

