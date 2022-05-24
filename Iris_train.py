import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump


test_size=0.2 
random_state=42

def load_data():
    return pd.read_csv("iris.csv")


iris = load_data()
iris.drop('Id',inplace=True, axis=1)


X=iris.drop(['Species'],axis=1)
y=iris['Species']
encoder=LabelEncoder()
y=encoder.fit_transform(y)

dump(encoder,'encoder')


pipeline_KNeighbors = Pipeline([('pca1', PCA(n_components=3)),
    ('KNeighbors',KNeighborsClassifier(n_neighbors=3))
])
pipeline_KNeighbors.fit(X,y)

dump(pipeline_KNeighbors,'my_model')




