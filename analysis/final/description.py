#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'analysis'))
	print(os.getcwd())
except:
	pass


#%% importowanie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%% 
data = pd.read_csv('/Users/empro/Documents/HR/contests/bioHack/data/predicting_response/X_genes.tsv', sep = '\t')
y = pd.read_csv('/Users/empro/Documents/HR/contests/bioHack/data/predicting_response/y.tsv', sep = '\t', header = None, names = ['labels'])

comb = pd.concat([y, data], axis = 1)
#%%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

#PCA

pca = PCA(n_components=2)
scaled = StandardScaler().fit_transform(comb.iloc[:, 1:])
reduced = pca.fit_transform(scaled)

pcaDF = pd.DataFrame(data = reduced, columns = ['PC1', 'PC2'])
pcaDF['label'] = list(comb.iloc[:, 0])

sns.set_style('white')
fig = plt.figure(figsize = (20, 10))
fig = sns.scatterplot(data = pcaDF, x = 'PC1', y = 'PC2', hue = 'label', palette='deep')
plt.xlabel('PC1', fontsize=20)
plt.ylabel('PC2', fontsize=20)
plt.legend(fontsize='x-large', title_fontsize='40', loc = 'upper left')


fig = sns.despine()

#UMAP

u = umap.UMAP(random_state = 42)
reduced = u.fit_transform(scaled)
uDF = pd.DataFrame(data = reduced, columns = ['1st emb', '2nd emb'])
uDF['label']= list(comb.iloc[:, 0])

fig = plt.figure(figsize = (20, 10) ) 
fig = sns.scatterplot(data = uDF, x = '1st emb', y = '2nd emb', hue = 'label', palette='deep')
plt.xlabel('1st emb', fontsize=20)
plt.ylabel('2nd emb', fontsize=20)
plt.legend(fontsize='x-large', title_fontsize='40', loc = 'upper left')

fig = sns.despine()
