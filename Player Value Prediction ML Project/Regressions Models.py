from sklearn import linear_model
from sklearn import metrics
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *
from sklearn.metrics import r2_score
import time
import joblib

#Load players data
data = pd.read_csv('player-value-prediction.csv')
X ,Y , data,S_median,S_median_val = FillingData(data,0,0,1,"reg")
#pickle_model = open("FillingData", "wb")
#pickle.dump(S_median, pickle_model)
#pickle.dump(S_median_val, pickle_model)
#pickle_model.close()
s=[S_median,S_median_val]
joblib.dump(s,"FillingData.pkl")
#joblib.dump(S_median_val,"FillingData1.pkl",)
cols = ('name', 'full_name', 'work_rate1', 'club_position', 'positions', 'nationality', 'preferred_foot',
        'work_rate', 'body_type', 'club_team', 'national_team',
        'national_team_position', 'tags', 'traits', 'positions1', 'positions2','positions3')

X ,lbls= Feature_Encoder(X,cols)
X['work_rate'] = X['work_rate'] + X['work_rate1']
X.drop('work_rate1', inplace=True, axis=1)
#pickle_model = open("Feature_Encoder", "wb")
#pickle.dump(lbls, pickle_model)
#pickle_model.close()
#lbls[1].fit(list(X[1].values))
#Z=lbls[1].transform(list(X[1].values))
#print("L=  ",lbls)
joblib.dump(lbls,"Feature_Encoder.pkl")
for n in X:
    data[n]= X[n]

values=data['value']
data.drop('value', inplace=True, axis=1)
data.insert(len(data.columns),'value',values)
#print(data)

top_feature_num=MakeCorrelation(data,0.55,'value')

Name=[]
for n in top_feature_num:
  Name.append(n)
#print(Name)

X = X[Name]

joblib.dump(Name,"Top_Feature.pkl")


#OR Normalization
#X = featureScaling(X,0,1)
scal = sklearn.preprocessing.MinMaxScaler()
X=scal.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,shuffle=True,random_state=9900)
#X_train=X
#y_train=Y
'''
#POLY######################################################################################################################
E_data = pd.read_csv(r'E:\code\PycharmProjects\PythonProject1\ProjectTestSamples\Milestone 1\player-test-samples.csv')
X_E_Test, Y_E_Test, E_data, S_mediannn, S_median_vallllll = FillingData(E_data, S_median, S_median_val, 1)
cols = ('name', 'full_name', 'work_rate1', 'club_position', 'positions', 'nationality', 'preferred_foot',
        'work_rate', 'body_type', 'club_team', 'national_team','national_team_position', 'tags', 'traits', 'positions1', 'positions2', 'positions3')

X_E_Test, lbls = Feature_Encoder(X_E_Test,cols)
X_E_Test['work_rate'] = X_E_Test['work_rate'] + X_E_Test['work_rate1']
X_E_Test.drop('work_rate1', inplace=True, axis=1)
X_E_Test = X_E_Test[Name]
scal = sklearn.preprocessing.MinMaxScaler()
X_E_Test=scal.fit_transform(X_E_Test)
#print("Size new X= ",X_E_Test.shape)
'''
#POLY######################################################################################################################
print('***********************************************')
print('***********************************************')
print("---------------------------------Polynomial Model-------------------------------------")
poly_features = PolynomialFeatures(degree=3)
# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)
#print("Size1= ",X_train.shape)
#print("Size2= ",X_train_poly.shape)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
start = time.time()
poly_model.fit(X_train_poly, y_train)
stop = time.time()
# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
#ypred=poly_model.predict(poly_features.transform(X_E_Test))
ypred=poly_model.predict(poly_features.transform(X_test))

# predicting on test data-set
#prediction = poly_model.predict(poly_features.fit_transform(X_E_Test))
prediction = poly_model.predict(poly_features.fit_transform(X_test))

print(f"Training Time : {stop-start}s")
print('Co-efficient of linear regression',poly_model.coef_)
print('Intercept of linear regression model',poly_model.intercept_)
#print('Mean Square Error', metrics.mean_squared_error(Y_E_Test, prediction))
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))

true_player_value=np.asarray(y_test)[0]

predicted_player_value=prediction[0]
print('True value for the first player in the test set in millions is : ' + str(true_player_value))
print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))
print("The achieved Accuracy is :  " + str(r2_score(y_test, prediction)*100))


#pkl_filename = "Polynomial_model.pkl"
#with open(pkl_filename, 'wb') as file:
#  pickle.dump(poly_model, file)

joblib.dump(poly_model,"Polynomial_model.pkl")
print('***********************************************')
print('***********************************************')
##################################Multipli Linear Regression Model####################################################
print("---------------------------------Multi Regression Model-------------------------------------")
cls = linear_model.LinearRegression()
start_time = time.time()
cls.fit(X_train,y_train)
stop = time.time()
prediction= cls.predict(X_test)

print(f"Training Time : {stop-start_time}s")
print('Multi Co-efficient of linear regression',cls.coef_)
print('Multi Intercept of linear regression model',cls.intercept_)
print('Multi Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))

true_player_value=np.asarray(y_test)[0]
predicted_player_value=prediction[0]
print('True value for the first player in the test set in millions is : ' + str(true_player_value))
print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))
print("---%s seconds ---"%(time.time()-start_time))
print("The achieved Accuracy is: " + str(r2_score(y_test,prediction)*100))
joblib.dump(cls,"Multi Regression_model.pkl")

'''
print('***********************************************')
print('***********************************************')

cls = linear_model.LinearRegression()
start_time = time.time()
cls.fit(X_train,y_train)
stop = time.time()
prediction= cls.predict(X_E_Test)
sco= cls.score(X_E_Test,Y_E_Test)
print(f"Training Time : {stop-start_time}s")
print('Multi Co-efficient of linear regression',cls.coef_)
print('Multi Intercept of linear regression model',cls.intercept_)
print('Multi Mean Square Error', metrics.mean_squared_error(np.asarray(Y_E_Test), prediction))

true_player_value=np.asarray(Y_E_Test)[0]
predicted_player_value=prediction[0]
print('True value for the first player in the test set in millions is : ' + str(true_player_value))
print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))
print("---%s seconds ---"%(time.time()-start_time))
print("The achieved Accuracy is: " + str(r2_score(Y_E_Test,prediction)*100))
print("The achieved Accuracy issss: " + str(sco))

'''