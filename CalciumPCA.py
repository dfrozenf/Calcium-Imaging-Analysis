import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pca import pca

nFeatures = 12
features = 'C:/Users/User/Desktop/calciumdump.csv'
feature_names = ['F{}'.format(i+1) for i in range(nFeatures)]
id = []

with open(features) as f:
    features = f.readlines()[1:]
    features = [i.rstrip().split(sep=',') for i in features]
    for i in range(len(features)):
        for j in range(len(features[i])):
            features[i][j] = float(features[i][j])

for i in range(len(features)):
    for j in range(len(features[i])):
        if features[i][j] == -999:
            features[i][j] = np.nan

mask = np.any(np.isnan(features), axis=1)
features = np.array(features)[~mask].tolist()

for i in range(len(features)):
    id.append(features[i][-2])
    features[i] = features[i][0:-2]

nsamples = len(features)

features = np.array(features).reshape((nsamples,nFeatures))

scaler = StandardScaler().fit(features)
sfeatures = scaler.transform(features)

sfeatures = pd.DataFrame(sfeatures, index = id, columns=feature_names)

model = pca(n_components=0.95)
results = model.fit_transform(sfeatures, col_labels=feature_names)
fig, ax = model.plot()
fig, ax = model.scatter3d(legend=False)
fig, ax = model.biplot3d(n_feat=nFeatures, legend=False)

#TODO: Use all cells and subset outliers for another pca