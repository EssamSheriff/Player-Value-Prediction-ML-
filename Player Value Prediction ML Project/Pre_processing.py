from sklearn.preprocessing import LabelEncoder
import numpy as np
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import pandas as pd


def Feature_Encoder(X,cols):
    L=[]
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
        #print("L= ",lbl.fit_transform(list(X[c].values)))
        L.append(lbl.fit_transform(list(X[c].values)))
    return X,L

def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X


def FillingData(data,S_median,S_median_val,flag,model):
    X = data.iloc[:, 0:91]  # Features
    if model =="reg":
        if S_median_val == 0:
            S_median_val = data['value'].median()
        data['value'].replace([numpy.NaN], S_median_val, inplace=True)
        Y = data['value'].astype(int)  # Label
    else:
        lab = LabelEncoder()
        lab.fit(list(data['PlayerLevel'].values))
        data['PlayerLevel'] = lab.transform(list(data['PlayerLevel'].values))
        data['PlayerLevel'].replace([np.NaN], data['PlayerLevel'].median(), inplace=True)
        Y = data['PlayerLevel']  # Label
        #print(Y)

    ColsString = ['club_team', 'club_position', 'national_team', 'national_team_position', 'tags', 'traits']
    ColsNumber = ['wage', 'release_clause_euro', 'club_rating', 'club_jersey_number', 'national_rating',
                  'national_jersey_number']
    ColsDate = ['club_join_date']
    ColsLS = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM',
              'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
    #pd.set_option("display.max_columns",None)
    #print("old X= ",X)
    for n in ColsLS:
        # X[n]=X[n].replace([numpy.NaN], 'NaN', inplace=True)
        data[n].replace([numpy.NaN], '0+0', inplace=True)
        X[n] = data[n]
        X[[n, 't']] = X[n].str.split('+', expand=True)
        X[n] = X[n].astype(int)
        data[n]=X[n].astype(int)

    for n in ColsString:
        # X[n]=X[n].replace([numpy.NaN], 'NaN', inplace=True)
        data[n].replace([numpy.NaN], 'No', inplace=True)
        X[n] = data[n]
        data[n]=X[n]


    for n in ColsNumber:
        # X[n]=X[n].replace([numpy.NaN], '0', inplace=True)
        if S_median==0:
            S_median = data[n].median()
        data[n].replace([numpy.NaN],S_median , inplace=True)
        X[n] = data[n]
        data[n]=X[n]


    for n in ColsDate:
        # X[n]=X[n].replace([numpy.NaN], '1/1/2022', inplace=True)
        data[n].replace([numpy.NaN], '1/1/1990', inplace=True)
        X[n] = data[n]
        data[n]=X[n]


    X[['work_rate', 'work_rate1']] = X['work_rate'].str.split('/', expand=True)


    if flag==1:
        X[['positions', 'positions1', 'positions2', 'positions3']] = X['positions'].str.split(',', expand=True)
    else:
        X[['positions', 'positions1', 'positions2']] = X['positions'].str.split(',', expand=True)


    cols = ('name', 'full_name', 'work_rate1', 'club_position', 'positions', 'nationality', 'preferred_foot',
            'international_reputation(1-5)', 'work_rate', 'body_type', 'club_team', 'national_team',
            'national_team_position', 'tags', 'traits', 'positions1', 'positions2', 'positions3')


    X[['t','t1','birth_date']] = X['birth_date'].str.split('/', expand=True)
    X[['t','t1','club_join_date']] = X['club_join_date'].str.split('/', expand=True)
    X['club_join_date']=X['club_join_date'].astype(int)
    data['club_join_date'] = X['club_join_date'].astype(int)
    #z=X['birth_date']
    #print(z)
    data['birth_date'] = X['birth_date'].astype(int)
    X['birth_date']=X['birth_date'].astype(int)

    X.drop('contract_end_year', inplace=True, axis=1)
    data.drop('contract_end_year', inplace=True, axis=1)
    X.drop('t', inplace=True, axis=1)
    X.drop('t1', inplace=True, axis=1)
  #  print("new X= ",X['international_reputation(1-5)'])

    return X,Y,data,S_median,S_median_val


def Make_Corr_cat(X,data):
    cols = ('name', 'full_name', 'work_rate1', 'club_position', 'positions', 'nationality', 'preferred_foot',
           'work_rate', 'body_type', 'club_team', 'national_team',
            'national_team_position', 'tags', 'traits', 'positions1', 'positions2', 'positions3')
    T=[]
    for n in X :
        contingency = pd.crosstab(X[n], data['value'])
        stat, p, dof, expected = chi2_contingency(contingency)
        # interpret p-value
        alpha = 0.005
        print("p value is " + str(p))
        if p <= alpha:
            print('Dependent (reject H0)')
            T.append(n)
        else:
            print('Independent (H0 holds true)')
        #print(data)
        #print(expected)

    print('NEW LISt  ',T)
    return T

def MakeCorrelation(X, val,name):
    # Get the correlation between the features
    corr = X.corr()
    # Top 50% Correlation training features with the Value
    top_feature = corr.index[abs(corr[name]) > val]
    # Correlation plot
    plt.subplots(figsize=(12, 8))

    top_corr = X[top_feature].corr()

    sns.heatmap(top_corr, annot=True)
    plt.show()
    top_feature = top_feature.delete(-1)
    #print(len(top_feature))
    return top_feature