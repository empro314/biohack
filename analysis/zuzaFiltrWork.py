#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'analysis'))
	print(os.getcwd())
except:
	pass


#%%Useful functions definitions

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
import pandas as pd
import numpy as np
import sklearn as sk
import scipy as sp
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt


#%%
prePath = '../data/predicting_response/'

covar = pd.read_csv(prePath + 'X_covariates.tsv', sep='\t', engine='python')
geneexp = pd.read_csv(prePath + 'X_genes.tsv', sep='\t', engine='python')
y = pd.read_csv(prePath + 'y.tsv', header = None)


#%%
cov_missing = covar.isna().mean()*100

#%%
cov_missing


#%%
xTrain, xTest, yTrain, yTest = train_test_split(geneexp, y, test_size = 0.25, random_state = 42)


#%%
geneFilt = xTrain.loc[:, xTrain.median() > 10]
genelog = np.log10(geneFilt + 1)


#%%
sns.distplot(genelog.median())


#%%
fig = plt.figure(figsize = (20, 20))

clusterOpttions = {
    'col_cluster' : False,
    'method' : 'average',
    'metric' : 'euclidean',
    'figsize' : (15, 15),
    'robust' : True,
    'cmap' : 'RdBu',
    'robust' : True,
}

lut = dict(zip(list(yTrain.iloc[:, 0].unique()), 'rb'))
rowCols = yTrain.iloc[:, 0].map(lut)

fig = sns.clustermap(data = genelog, **clusterOpttions, row_colors = rowCols)

for label in yTrain.iloc[:, 0].unique() :
    fig.ax_col_dendrogram.bar(0, 0, color = lut[label], label = label, linewidth = 0)
fig.ax_col_dendrogram.legend(title = 'dupaaa', loc = 3, ncol = 1)


#%% ## Analiza statystyczna 


#%%
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy.stats.mstats import gmean

#%% zlozenie datasetu z labelami na 1 pozycji 

merged = pd.concat([yTrain, genelog], axis = 1)


#%% Zmien wkurzajace nazwy genow

pres = list(merged.columns)

nowe = [i[9:] for i in pres[1:]]
nowe = ['label'] + nowe

merged.columns = nowe
#%%
pvals = []
FCs = []


for i in list(merged.columns[1: ]) :
    aSet = merged.loc[merged['label'] == 1, i]
    bSet = merged.loc[merged['label'] == 0, i]
    act = ttest_ind(a = aSet, b = bSet)
    pvals.append(act.pvalue)
    
    fc = gmean(10 ** merged.loc[merged['label'] == 1, i]) / gmean(10 ** merged.loc[merged['label'] == 0, i])
    FCs.append(fc)
    

pCor = multipletests(pvals, alpha = 0.05, method = 'fdr_bh')[1]

statDF = pd.DataFrame({'p' : pvals, 'pAdj' : pCor, 'FC' : FCs})
statDF.index = list(merged.columns[1:])

statDF = statDF.sort_values(by = 'p')

#%%
#volcano plot

volPs = -np.log10(statDF.loc[:, 'pAdj'])
volSig = statDF.loc[:, 'pAdj'] < 0.05
volFold = np.log2(statDF.loc[:, 'FC'])

volSig = volSig.rename('p Adjusted < 0.05')

fig = plt.figure( figsize = (20, 10) )
ax = sns.scatterplot(x = volFold, y = volPs, hue = volSig, s = 80)#, palette=['copper', 'ocean'])
ax.set_facecolor('white')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.tick_params(axis = 'both', labelsize=16)

plt.xlabel('log2(foldChange)', fontsize=20)
plt.ylabel('-log10(p Adjsuted)', fontsize=20)
plt.legend(fontsize='x-large', title_fontsize='80')

#%% wez zmienne istotne bez poprawki

almostSig = statDF.loc[statDF['p'] < 0.05, :]
almostSig = almostSig.sort_values(axis = 0, by = 'FC')


#%% wybieranie genow istotnych bez poprawki majacy fold change in (0, 0.66> u <1.5, inf)

istFold = almostSig.loc[(almostSig['FC'] >= 1.5) | (almostSig['FC'] <= (1 / 1.5) ), :]

istotneFold = istFold.index
#%%zapisz pierwsza liste do pliku 

#writeList(zapiszListyPref + 'istotneSkorel.txt', istotneFold)

#%% - -- - - -- - - - - - - - - -- - - - - - - -- - - - -


#%% plot=t correlation

corr = merged.loc[:, istotneFold].corr()

sns.set(font_scale=2)
fig = plt.figure(figsize = (20, 20))
fig = sns.heatmap(data = corr, annot = False, cmap="RdYlGn", annot_kws={"size": 20} )


#%% Plot clusterisation

fig = plt.figure(figsize = (20, 20))

clusterOpttions = {
    'col_cluster' : False,
    'method' : 'average',
    'metric' : 'euclidean',
    'figsize' : (15, 15),
    'robust' : True,
    'cmap' : 'RdBu',
    'robust' : True,
}

lut = dict(zip(list(merged.iloc[:, 0].unique()), 'rb'))
rowCols = merged.iloc[:, 0].map(lut)

fig = sns.clustermap(data = merged.loc[:, istotneFold], **clusterOpttions, row_colors = rowCols)

for label in merged.iloc[:, 0].unique() :
    fig.ax_col_dendrogram.bar(0, 0, color = lut[label], label = label, linewidth = 0)
fig.ax_col_dendrogram.legend(title = 'dupaaa', loc = 3, ncol = 1)
#%% Wywyal skorelowane parami (analogicznie do poprzednie workflowu)
from scipy.stats import spearmanr

res = [True for i in istotneFold]

for i in range(0, len(istotneFold)) :
    
    if res[i] == False : 
        continue
    
    for j in range(i + 1, len(istotneFold) ) :
        
        if res[j] == False :
            continue
        
        if abs(spearmanr(merged.loc[:, istotneFold[i]], merged.loc[:, istotneFold[j]])[0]) > 0.7 :
            res[j] = False
        
from itertools import compress

poKorel = list(compress(istotneFold, res))


#%%
poKorel05 = poKorel
poKorel07 = poKorel

#%% zapisz srednie do pliku 
#writeList(zapiszListyPref + 'poKorel05', poKorel05)
#writeList(zapiszListyPref + 'poKorel07', poKorel07)

#%% Prepare mergeTest set

mergedTest = pd.concat([yTest, xTest], axis = 1)

pres = list(mergedTest.columns)
nowe = [i[9:] for i in pres[1:]]
nowe = ['label'] + nowe
mergedTest.columns = nowe

mergedTest.iloc[:, 1:] = np.log10(mergedTest.iloc[:, 1:] + 1)
#%%
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000, random_state=42)


rf.fit(merged.loc[:, poKorel07], merged.iloc[:, 0])

yPredicted = rf_predict = rf.predict(mergedTest.loc[:, poKorel07])


#%% Evaluate model performance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

cm = confusion_matrix(mergedTest.iloc[:, 0], yPredicted,  labels=[0, 1])
cr = classification_report(mergedTest.iloc[:, 0], yPredicted)
ac = accuracy_score(mergedTest.iloc[:, 0], yPredicted])

#%% ocena AUC
from sklearn.metrics import roc_auc_score

yProba = rf.predict_proba(mergedTest.loc[:, poKorel07])
yProba = [i[1] for i in yProba]

roc_auc_score(mergedTest.iloc[:, 0], yProba)

#%%
