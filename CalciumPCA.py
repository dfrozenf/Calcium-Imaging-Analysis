import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pca import pca

nFeatures = 12
features = 'C:/Users/User/Desktop/calciumdump.csv'
feature_names = ['F{}'.format(i+1) for i in range(nFeatures)]
id = []

with open(features) as f: #Read in features extracted from our preprocessing script
    features = f.readlines()[1:]
    features = [i.rstrip().split(sep=',') for i in features]
    for i in range(len(features)):
        for j in range(len(features[i])):
            features[i][j] = float(features[i][j])

for i in range(len(features)): #If a feature was determined to be invalid previously, mark it with a NaN
    for j in range(len(features[i])):
        if features[i][j] == -999:
            features[i][j] = np.nan

mask = np.any(np.isnan(features), axis=1) #Mask and remove NaN containing signals from the feature array (row-wise)
features = np.array(features)[~mask].tolist()

for i in range(len(features)): #Build an ID table for color coding in the PCA
    id.append(features[i][-2])
    features[i] = features[i][0:-2]

nsamples = len(features) #Find the total number of signals fed into the PCA

features = np.array(features).reshape((nsamples,nFeatures)) #Reshape the trimmed feature matrix

scaler = StandardScaler().fit(features) #Apply a feature-(column-)wise scaler to normalize values before fitting PCA
sfeatures = scaler.transform(features)

sfeatures = pd.DataFrame(sfeatures, index = id, columns=feature_names) #Convert to a Pandas dataframe, so that we can place feature names on loadings in our biplot

model = pca(n_components=0.95) #Set a 95% variance threshold for our PCA
results = model.fit_transform(sfeatures, col_labels=feature_names) #Calcualte principle component coordinates 
fig, ax = model.plot() #Build a variance plot
fig, ax = model.scatter3d(legend=False)
fig, ax = model.biplot3d(n_feat=nFeatures, legend=False) #Build a biplot containing flattened signals and loadings

#TODO: Use all cells and subset outliers for another pca
