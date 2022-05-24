
from joblib import load
import pandas as pd

knn_from_joblib=load("my_model")
encoder=load("encoder")

print("Enter your Numbers:")
SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm=input().split(',')
X_input= pd.DataFrame([ [SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm] ],columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
predict=knn_from_joblib.predict(X_input)
predict_name=encoder.inverse_transform(predict)
print('predict name is:',predict_name)
