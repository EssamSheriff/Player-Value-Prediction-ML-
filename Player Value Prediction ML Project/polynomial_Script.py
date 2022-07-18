import joblib
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.metrics import r2_score
from Pre_processing import *
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv(r'E:\code\PycharmProjects\PythonProject1\Player Value Prediction\player-tas-regression-test.csv')

K= joblib.load("FillingData.pkl")

X, Y, data, S_median, S_median_val = FillingData(data, K[0], K[1], 1, "reg")
pd.set_option('display.max_columns', None)

cols = ('name', 'full_name', 'work_rate1', 'club_position', 'positions', 'nationality', 'preferred_foot',
        'work_rate', 'body_type', 'club_team', 'national_team',
        'national_team_position', 'tags', 'traits', 'positions1', 'positions2', 'positions3')
Z=X
print(Z)
X, lbls = Feature_Encoder(X,cols)

X['work_rate'] = X['work_rate'] + X['work_rate1']
X.drop('work_rate1', inplace=True, axis=1)

Name = joblib.load("Top_Feature.pkl")
X = X[Name]
#print(Name)
#X = featureScaling(X, 0, 1)
scal = sklearn.preprocessing.MinMaxScaler()
X=scal.fit_transform(X)
#print(X)
print('*********************************************************************************************************************************************')
print("---------------------------------Polynomial Model-------------------------------------")
poly_model = joblib.load("Polynomial_model.pkl")
#print("*****************************")
#print(X)
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X)
#print("Size= ",X_train_poly.shape)
result = poly_model.predict(X_train_poly)
#result = joblib_model.score(X, Y)
#print(result)
print('Mean Square Error', metrics.mean_squared_error(result,Y))
print("The achieved Accuracy is :  " + str(r2_score(result,Y)*100))
print('*********************************************************************************************************************************************')
print('*********************************************************************************************************************************************')
##################################Multipli Linear Regression Model####################################################
print("---------------------------------Multi Regression Model-------------------------------------")
Multi_Regression_model = joblib.load("Multi Regression_model.pkl")
p=Multi_Regression_model.predict(X)
print('Mean Square Error', metrics.mean_squared_error(p,Y))
print("The achieved Accuracy is :  " + str(r2_score(p,Y)*100))
print('*********************************************************************************************************************************************')
















'''
P= Ypredict.reshape(-1,1)
P=np.exp(P)
y_test=np.exp(y_test)
print("The achieved Accuracy is :  " + str(r2_score(P, y_test)))
print('Mean Square Error', metrics.mean_squared_error(P, y_test)/1000000)

pkl1_filename = "model_poly_features.pkl"
with open(pkl1_filename, 'rb') as file:
    Feature_P = pickle.load(file)
X_test_poly_model= Feature_P.transform(X_test)

pkl1_filename = "Polynomial_model.pkl"
with open(pkl1_filename, 'rb') as file:
    pickle_model = pickle.load(file)
Ypredict = pickle_model.predict(X_test_poly_model)
print(Ypredict)
P= Ypredict.reshape(-1,1)
P=np.exp(P)
y_test=np.exp(y_test)
print("The achieved Accuracy is :  " + str(r2_score(P, y_test)))
print('Mean Square Error', metrics.mean_squared_error(P, y_test)/1000000)
'''