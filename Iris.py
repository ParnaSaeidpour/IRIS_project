import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
from joblib import load

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


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=random_state)



pipeline_KNeighbors = Pipeline([('pca1', PCA(n_components=3)),
    ('KNeighbors',KNeighborsClassifier(n_neighbors=3))
])
pipeline_KNeighbors.fit(X_train,y_train)
print("Before save:",pipeline_KNeighbors.predict(X_test))


dump(pipeline_KNeighbors,'my_model')




knn_from_joblib=load("my_model")
print(knn_from_joblib.predict(X_test))


print("Enter your Numbers:")
SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm=input().split(',')
X_input= pd.DataFrame([ [SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm] ],columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
predict=knn_from_joblib.predict(X_input)
predict_name=encoder.inverse_transform(predict)
print('predict name is:',predict_name)