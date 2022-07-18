import joblib
import sklearn.preprocessing
from Pre_processing import *
from sklearn import *
from sklearn.linear_model import LogisticRegression
from Pre_processing import *
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import plotly.express as px
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import sklearn.preprocessing
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier

data = pd.read_csv(r'E:\code\PycharmProjects\PythonProject1\Player Value Prediction\player-tas-classification-test.csv')

K= joblib.load("FillingData_Classification.pkl")

X, Y, data, S_median, S_median_val = FillingData(data, K[0], K[1], 1, "")

cols = ('name', 'full_name', 'work_rate1', 'club_position', 'positions', 'nationality', 'preferred_foot',
        'work_rate', 'body_type', 'club_team', 'national_team',
        'national_team_position', 'tags', 'traits', 'positions1', 'positions2', 'positions3')

X, lbls = Feature_Encoder(X,cols)
X['work_rate'] = X['work_rate'] + X['work_rate1']
X.drop('work_rate1', inplace=True, axis=1)

Name = joblib.load("Top_Feature_classification.pkl")
X = X[Name]
#print(Name)
pd.set_option('display.max_columns', None)
#X = featureScaling(X, 0, 1)
scal = sklearn.preprocessing.MinMaxScaler()
X=scal.fit_transform(X)

print('*********************************************************************************************************************************************')
print("--------------------------------- AdaBoost Model -------------------------------------")
AdaBoost_Model=joblib.load("AdaBoost Model.pkl")
result = AdaBoost_Model.predict(X)
accuracy = np.mean(result == Y)*100
print("The acc is "+str(accuracy))
print('*********************************************************************************************************************************************')
print("--------------------------------- Decision tree Model -------------------------------------")
Decision_tree_Model=joblib.load("Decision tree Model.pkl")
Decision_tree_Model_result = Decision_tree_Model.predict(X)
accuracy_DT = np.mean(Decision_tree_Model_result == Y)*100
print("The acc is "+str(accuracy_DT))
print('*********************************************************************************************************************************************')
print("--------------------------------- SVM Models -------------------------------------")
SVM_Linear_Model=joblib.load("SVC_with_linear_kernel_Model.pkl")
SVM_Linear_Model_result = SVM_Linear_Model.predict(X)
accuracy_SVM_linear = np.mean(SVM_Linear_Model_result == Y)*100
print("SVC with linear kernel:  " + str(accuracy_SVM_linear))
print("______________________________________________________________")
SVM_RBF_Model=joblib.load("SVC_with_rbf_kernel_Model.pkl")
SVM_RBF_Model_result = SVM_RBF_Model.predict(X)
accuracy_SVM_RBF = np.mean(SVM_RBF_Model_result == Y)*100
print("SVC with RBF kernel:  " + str(accuracy_SVM_RBF))
print("______________________________________________________________")
SVM_poly_Model=joblib.load("SVC_with_polynomial (degree 5)_kernel_Model.pkl")
SVM_poly_Model_result = SVM_RBF_Model.predict(X)
accuracy_SVM_poly = np.mean(SVM_poly_Model_result == Y)*100
print("SVC with polynomial (degree 4)_kernel_Model:  " + str(accuracy_SVM_poly))
print("______________________________________________________________")
Linear_SVM_Model=joblib.load("LinearSVC_(linear kernel)_Model.pkl")
Linear_SVM_Model_result = Linear_SVM_Model.predict(X)
accuracy_SVM = np.mean(Linear_SVM_Model_result == Y)*100
print("LinearSVC (linear kernel) Model:  " + str(accuracy_SVM))
print('*********************************************************************************************************************************************')
print('*********************************************************************************************************************************************')
Logistic_Model=joblib.load("Logistic Regression Model.pkl")
Logistic_Model_result = Logistic_Model.predict(X)
accuracy_Logistic = np.mean(Linear_SVM_Model_result == Y)*100
print("Logistic Regression Model:  " + str(accuracy_Logistic))

