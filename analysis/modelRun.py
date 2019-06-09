#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'analysis'))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd
import h2o


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

zapiszListyPref = '../dataSets/'


#%%
dataDir = '../dataSets/'
merged = pd.read_csv(dataDir + 'merged.csv', index_col = 0)

istotneSkorel = readList(dataDir + 'istotneSkorel.txt')
poKorel05 = readList(dataDir + 'poKorel05')
poKorel07 = readList(dataDir + 'poKorel07')
weka = readList(dataDir + 'weka1')


#%%
#puszczamy na poczatek TPOTa


#%%
from tpot import TPOTClassifier 


#%%
opt1 = TPOTClassifier(generations=20, population_size=50, cv=10,
                                    random_state=42, verbosity=2)


#%%
opt1.fit(merged.loc[:, weka], merged.iloc[:, 0])


#%%
opt1.export('dupa.py')


#%%
mergedTest = pd.read_csv('mergedTest.csv', index_col=0)
from sklearn.ensemble import GradientBoostingClassifier

# Average CV score on the training set was:0.8327380952380953
optimTree1 = GradientBoostingClassifier(learning_rate=0.1, max_depth=4, max_features=0.1, min_samples_leaf=4, min_samples_split=4, n_estimators=1000, subsample=0.6500000000000001)


#%%
optimTree1.fit(merged.loc[:, poKorel07], merged.iloc[:, 0])
results = optimTree1.predict(mergedTest.loc[:, poKorel07])


#%%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
cm = confusion_matrix(mergedTest.iloc[:, 0], results)


#%%
cm


#%%
from sklearn.metrics import roc_auc_score

yProba = optimTree1.predict_proba(mergedTest.loc[:,poKorel07])
yProba = [i[1] for i in yProba]

roc_auc_score(mergedTest.iloc[:,0], yProba)

#%% [markdown]
# ## h2o based classifier

#%%
import h2o
h2o.init()


#%%
trainingFrame = h2o.H2OFrame(merged)
validationFrame = h2o.H2OFrame(mergedTest)

trainingFrame['label'] = trainingFrame['label'].asfactor()
validationFrame['label'] = validationFrame['label'].asfactor()


#%%
from h2o.automl import H2OAutoML

#%% [markdown] 
## trenowanie na skorelowanych 
#%%
amlKorel = H2OAutoML(max_models = 10000, seed = 42, max_runtime_secs=3600, nfolds = 5)


#%%
amlKorel.train(x = poKorel07, y = 'label', training_frame = trainingFrame)


#%%
aml.leaderboard


#%%
topmodel = aml.leader


#%%
topmodel.model_performance(test_data = validationFrame, valid = True)


#%%
topmodel.plot


#%%
h2o.save_model(topmodel, 'topModl2')


#%% [markdown] 
## trenowanie na wece


#%%
amlWeka = H2OAutoML(max_models = 10000, seed = 42, max_runtime_secs=3600, nfolds = 5)

#%%
amlWeka.train(x = weka, y = 'label', training_frame = trainingFrame)

#%%
