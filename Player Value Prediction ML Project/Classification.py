import time
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



data = pd.read_csv('player-classification.csv')

X, Y, data, S_median, S_median_val = FillingData(data, 0, 0, 1, "")

s=[S_median,S_median_val]
joblib.dump(s,"FillingData_Classification.pkl")
cols = ('name', 'full_name', 'work_rate1', 'club_position', 'positions', 'nationality', 'preferred_foot',
        'work_rate', 'body_type', 'club_team', 'national_team',
        'national_team_position', 'tags', 'traits', 'positions1', 'positions2','positions3')

X ,lbls= Feature_Encoder(X,cols)
joblib.dump(lbls,"Feature_Encoder_class.pkl")

X['work_rate'] = X['work_rate'] + X['work_rate1']
X.drop('work_rate1', inplace=True, axis=1)

for n in X:
    data[n]= X[n]

values=data['PlayerLevel']
data.drop('PlayerLevel', inplace=True, axis=1)
data.insert(len(data.columns),'PlayerLevel',values)
#print(data)

top_feature_num=MakeCorrelation(data,0.026,'PlayerLevel')
Name=[]
for n in top_feature_num:
  Name.append(n)
#print(Name)

X = X[Name]

joblib.dump(Name,"Top_Feature_classification.pkl")


#OR Normalization
#X = featureScaling(X,0,1)
scal = sklearn.preprocessing.MinMaxScaler()
X=scal.fit_transform(X)
#scale = StandardScaler()
#scale.fit(X_train)
#X_train = scale.transform(X_train)
#X_test = scale.transform(X_test)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,shuffle=True,random_state=100)



print("\n")
print("--------------------------------- AdaBoost Model -------------------------------------")
##AdaBoost
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=12), algorithm="SAMME", n_estimators=100)
start_time = time.time()
bdt.fit(X_train,y_train)
stop = time.time()
print(f"Training Time : {stop-start_time}s")

start_time = time.time()
y_pred1 = bdt.predict(X_test)
acc = np.mean(y_pred1 == y_test)*100
stop = time.time()
print(f"Testing Time : {stop-start_time}s")
print("acc = "+ str(acc))
joblib.dump(bdt,"AdaBoost Model.pkl")

print('*********************************************************************************************************************************************')
print("--------------------------------- Decision tree Model -------------------------------------")
###Decision
clf = tree.DecisionTreeClassifier(max_depth = 12,criterion="entropy")
start_time = time.time()
clf.fit(X_train,y_train)
stop = time.time()
print(f"Training Time : {stop-start_time}s")
start_time = time.time()
y_pred = clf.predict(X_test)
accuracy = np.mean(y_pred == y_test)*100
stop = time.time()
print(f"Testing Time : {stop-start_time}s")
print("The acc is "+str(accuracy))
Decision_tree_accuracy=accuracy
joblib.dump(clf,"Decision tree Model.pkl")

print('*********************************************************************************************************************************************')
print("--------------------------------- pipeline adaBoost Model -------------------------------------")
#using pipeline
X_pipeline, Y_pipeline = make_classification(random_state=42)
X_train_pipeline, X_test_pipeline, y_train_pipeline, y_test_pipeline = train_test_split(X_pipeline, Y_pipeline, random_state=100)
pipe = make_pipeline(StandardScaler(), LogisticRegression())
start_time = time.time()
pipe.fit(X_train_pipeline, y_train_pipeline)  # apply scaling on training data
stop = time.time()
print(f"Training Time : {stop-start_time}s")
start_time = time.time()
print('Score = ',pipe.score(X_test_pipeline, y_test_pipeline)*100)  # apply scaling on testing data, without leaking training data.
stop = time.time()
print(f"Testing Time : {stop-start_time}s")

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
joblib.dump(pipe,"pipeline adaBoost Model.pkl")

print('*********************************************************************************************************************************************')
print('*********************************************************************************************************************************************')
print("\n")
print("--------------------------------- SVM Models -------------------------------------")

C = 0.95  # SVM regularization parameter
'''
svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(X_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)

for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    predictions = clf.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(accuracy*100)
'''
start_time = time.time()
svm_kernel_ovo = SVC(kernel='linear', C=C).fit(X, Y)
stop = time.time()
print(f"Training Time : {stop-start_time}s")
accuracy1 = svm_kernel_ovo.score(X, Y)
start_time = time.time()
accuracy = svm_kernel_ovo.score(X_test, y_test)
stop = time.time()
print(f"Testing Time : {stop-start_time}s")
print("Trainning data SVC with linear kernel:  " + str(accuracy1))
print("SVC with linear kernel:  " + str(accuracy))
joblib.dump(svm_kernel_ovo,"SVC_with_linear_kernel_Model.pkl")
print("______________________________________________________________")
start_time = time.time()
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(X, Y)
stop = time.time()
print(f"Training Time : {stop-start_time}s")
start_time = time.time()
predicition =rbf_svc.predict(X_test)
accurcy=np.mean(predicition == y_test)
stop = time.time()
print(f"Testing Time : {stop-start_time}s")
print("SVC with rbf kernel:  " + str(accuracy))
joblib.dump(rbf_svc,"SVC_with_rbf_kernel_Model.pkl")

print("______________________________________________________________")
start_time = time.time()
poly_svc = svm.SVC(kernel='poly', degree=4, C=C).fit(X, Y)
stop = time.time()
print(f"Training Time : {stop-start_time}s")
accuracy1 = poly_svc.score(X, Y)
start_time = time.time()
accuracy = poly_svc.score(X_test, y_test)
stop = time.time()
print(f"Testing Time : {stop-start_time}s")
print("Training data SVC with polynomial  kernel:  " + str(accuracy1))
print("SVC with polynomial kernel:  " + str(accuracy))
svm_accuracy=accuracy1
joblib.dump(poly_svc,"SVC_with_polynomial (degree 5)_kernel_Model.pkl")
print("______________________________________________________________")

start_time = time.time()
svm_linear_ovo = OneVsOneClassifier(LinearSVC(dual=False)).fit(X, Y)
stop = time.time()
print(f"Training Time : {stop-start_time}s")
accuracy1 = svm_linear_ovo.score(X, Y)
start_time = time.time()
accuracy = svm_linear_ovo.score(X_test, y_test)
stop = time.time()
print(f"Testing Time : {stop-start_time}s")
print("Training data LinearSVC (linear kernel):  " + str(accuracy1))
print("LinearSVC (linear kernel):  " + str(accuracy))
joblib.dump(svm_linear_ovo,"LinearSVC_(linear kernel)_Model.pkl")

print('*********************************************************************************************************************************************')
print('*********************************************************************************************************************************************')
print("\n")
print("--------------------------------- Logistic Regression Model -------------------------------------")
log_reg = linear_model.LogisticRegression(solver='liblinear', C=7500)
start_time = time.time()
log_reg.fit(X_train, y_train)
stop = time.time()
print(f"Training Time : {stop-start_time}s")

start_time = time.time()
log_reg.score(X_test, y_test)
stop = time.time()

prediction = log_reg.predict(X_test)

true_player_level = np.asarray(y_test)[0]
predicted_player_level = prediction[0]
#print('Co-efficient of linear regression', log_reg.coef_)
#print('Intercept of linear regression model', log_reg.intercept_)
print(f"Testing Time : {stop-start_time}s")
print('True player level in the test set is : ' + str(true_player_level))
print('Predicted player level player in the test set is : ' + str(predicted_player_level))
#print("The achieved Accuracy is :  " + str(r2_score(y_test, prediction)*100))
accuracy = np.mean(prediction == y_test)*100
print("The acc is "+str(accuracy))
logistic_regression_accuracy=accuracy

joblib.dump(log_reg,"Logistic Regression Model.pkl")
print('*********************************************************************************************************************************************')
print('*********************************************************************************************************************************************')
print("--------------------------------- Visualization -------------------------------------")

'''49
Models = ['polynomial SVM','logistic regression','Decision tree']
acc = [svm_accuracy,logistic_regression_accuracy,Decision_tree_accuracy]
print("The acc is "+str(svm_accuracy))
New_Colors = ['Blue','Yellow','Green']
plt.bar(Models, acc, color=New_Colors)
plt.title('Models accuracy', fontsize=14)
plt.xlabel('models', fontsize=14)
plt.ylabel('accuracy', fontsize=14)
plt.grid(True)
plt.show()


#print("The achieved Accuracy is: " + str(r2_score(y_test,y_pred)*100))
sns.histplot(x='age', data=data, kde=False, hue='PlayerLevel')
plt.show()
'''
'''def avg_of_attr(attr):
    avgSkills = []
    avgSkills.append(data[data['club_team'] == 'Paris Saint-Germain'][attr].mean())
    avgSkills.append(data[data['club_team'] == 'Real Madrid'][attr].mean())
    avgSkills.append(data[data['club_team'] == 'FC Barcelona'][attr].mean())
    avgSkills.append(data[data['club_team'] == 'Liverpool'][attr].mean())
    avgSkills.append(data[data['club_team'] == 'Manchester City'][attr].mean())
    avgSkills.append(data[data['club_team'] == 'Milan'][attr].mean())
    avgSkills.append(data[data['club_team'] == 'Chelsea'][attr].mean())
    avgSkillsNames = ['Paris Saint-Germain','Real Madrid','FC Barcelona','Liverpool','Manchester City','Milan','Chelsea']

    fig = px.bar(y=avgSkillsNames, x=avgSkills, orientation='h',title=attr,labels ={'x': attr, 'y':'Clubs'})
    fig.show()

avg_of_attr(attr='age')
avg_of_attr(attr='skill_moves(1-5)')
avg_of_attr(attr='dribbling')
avg_of_attr(attr='finishing')
avg_of_attr(attr='freekick_accuracy')
avg_of_attr(attr='short_passing')

'''

# class1 = data.loc[Y == 0]
# class2 = data.loc[Y == 1]
# class3 = data.loc[Y == 2]
# class4 = data.loc[Y == 3]
# class5 = data.loc[Y == 4]

# plt.scatter(class1.iloc[:, 0], class1.iloc[:, 1], s=10, label='class A')
# plt.scatter(class2.iloc[:, 0], class2.iloc[:, 1], s=10, label='class B')
# plt.scatter(class3.iloc[:, 0], class3.iloc[:, 1], s=10, label='class C')
# plt.scatter(class4.iloc[:, 0], class4.iloc[:, 1], s=10, label='class D')
# plt.scatter(class5.iloc[:, 0], class5.iloc[:, 1], s=10, label='class S')
# plt.show()