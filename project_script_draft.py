# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 18:23:32 2022

@author: Rayan
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


# 1. Data exploration: a complete review and analysis of the dataset including:

# Load and describe data elements (columns), provide descriptions & types, 
# ranges and values of elements as aproppriate. – use pandas, numpy and any other python packages.

# def load_data(file_name, path):
#     '''
#     Function to load csv data and convert to dataframe
#     parameters include file name and file path
#     '''
#     full_path = os.path.join(path, file_name)
#     return pd.read_csv(full_path)

# Load dataset
df = pd.read_csv("Bicycle_Thefts.csv")#, r"C:\Users\rayan\Documents\SchoolSem4\Supervised Learning\Project")

#print 1st 5 rows of dataset
df.head()


# Data exploration

def describe_dataframe(df):
    print(f'Dataframe Columns: \n\n {df.columns.tolist()}\n\n')
    print(f'More Info on dataframe: \n')
    print({df.info()})
    print(f'\nBasic statistics on dataframe: \n\n {df.describe()}')
    print('\nRange on numerical columns: \n')
    cat_cols = [col for col in df.columns if df[col].dtype.name=="object"]
    df_num = df.drop(cat_cols, axis = 1)
    print(df_num.max() - df_num.min())
    print(f"\nPrint first 5 rows of dataframe: \n\n {df.head()}")

# call function to describe dataset  
describe_dataframe(df)

# Statistical assessments including means, averages, correlations
print('\nMean:')
print(df.mean())
print('\nMedian:')
print(df.median())
print('\nMin:')
print(df.min())
print('\nMax:')
print(df.max())
print('\nCount:')
print(df.count())
print('\nNumber of distinct categorical values:')
cat_cols = [col for col in df.columns if df[col].dtype.name=="object"]
print(df[cat_cols].nunique())


# Correlation
print(f"Correlation: {df.corr()}")


# Missing data evaluations – use pandas, numpy and any other python packages
print('\nNumber of missing values:')
print(df.isnull().sum())


# Graphs and visualizations – use pandas, matplotlib, seaborn, numpy and any other python packages,
# you also can use power BI desktop.

# 1. Histogram
df.hist(figsize =(15, 10))

# 2. Correlation Heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr())
plt.title("Correlation Plot of Features")

# TABLEAU PLOTS
# 3. Line Plot
# 4. Bar Plot
# 5. Count Plot
# 6. Buble Chart



# 2. Data modelling:

# Data transformations – includes handling missing data, categorical data management, 
#data normalization and standardizations as needed.

def notuseful_col(df, base=[], missing=1, uniq=-1):
    '''
    Function to extract columns not likely to contribute much value in 
    terms of model performance.
    
    Parameters:
        df -- dataframe object
        base -- a list to contain initial columns manually selected
        missing -- max % of missing values the columns should contain
        uniq -- max number of unique/distint values an object-datatype column
                should have
        '''
    if uniq == -1: uniq = df.shape[0]
    cols = base
    cols = cols + [c for c in df.columns if df[c].isnull().sum() > df.shape[0] * missing]
    cols = cols + [c for c in df.columns if df[c].nunique() > uniq and df[c].dtype.name=="object"]
    return list(set(cols))

# Drop not useful columns
df.drop(columns=notuseful_col(df, 
                              ['OBJECTID', 'Report_Year', 'Report_Month',
                               'Report_DayOfWeek', 'Report_DayOfMonth', 'Report_DayOfYear', 'Report_Hour',
                               'X','Y','Division','City'], 
                              missing=0.3, 
                              uniq=20), axis=1, inplace = True)




# confirm distinct classes before transformation
df['Status'].unique()

# convert target data type to numeric and binerize
df['Status'] = df['Status'].map(lambda x: 1 if x == "RECOVERED" else 0)

# Confirm number of distinct classes after transformation
df['Status'].unique()
df['Status'].value_counts()

# Class composition of Target
print(f" Percentage composition of classes: \
      \n{round(df['Status'].value_counts(normalize = True),4) * 100}")


# Managing imbalanced classes: Up-Sampling
def up_sample(df):
    # define majority and minority records
    df_majority = df[df.Status==0]
    df_minority = df[df.Status==1]
    
    # apply Resample function from sklearn.utils class
    df_minority_upsampled = resample(df_minority, 
                                 replace=True,
                                 n_samples=df_majority.shape[0],
                                 random_state=111)
    
    # merge majority records with upsampled minority
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    return df_upsampled


X = df.drop('Status', axis=1) # features
Y = df.Status # target

# Apply up_sample function to dataframe

X
# Construct pipelines
# For numerical features
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

# For categorical features
cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ("1hot", OneHotEncoder()),
    ])

# Make list of Categorical values
cat_cols_prep = [col for col in X.columns if X[col].dtype.name=="object"]

# Make list of Numerical values
num_cols = list(X.drop(X[cat_cols_prep].columns, axis=1))

# Make full pipeline by combining both numerical and categorical pipelines
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols_prep),
    ])
#X_train.columns
# Split the data set into train/test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=111) # 70% training and 30% test
test_df = pd.concat([X_test,y_test], axis=1)
test_df.to_csv("test_Data.csv")
sample_train_df = pd.concat([X_train,y_train], axis=1)
 #-- Do ONLY transform on test data; do not fit_transform
X_train_sampled = up_sample(sample_train_df)

# Confirm status of up-sampled df
X_train_sampled['Status'].value_counts()

# Define features and target

X_train_resampled = X_train_sampled.drop('Status', axis=1) # features
Y_train_resampled = X_train_sampled.Status 
# Logisitic Regression
from sklearn.linear_model import LogisticRegression
pipe_logistic = Pipeline([("Transformer", full_pipeline),("LG",LogisticRegression(max_iter = 1000))])

# Apply full pipeline for LR
X_train_lg = pipe_logistic.fit(X_train_resampled,Y_train_resampled)

# parameter grid and grid search
param_grid_logistic = [
    {'LG__solver' : ['lbfgs', 'saga','liblinear'],
     'LG__C': [0.01,0.1,1,10,100],
     }]

grid_search_logistic = GridSearchCV(estimator = pipe_logistic, param_grid = param_grid_logistic, scoring = 'accuracy', refit = True, verbose = 3)

#randomized search for logistic regression

parameters= parameters={'LG__C': range(1,100,10)}
    
search = RandomizedSearchCV(estimator=pipe_logistic, scoring='accuracy', param_distributions=parameters,cv=5, n_iter=7,refit=True,verbose=3 )

search.fit(X_train_resampled,Y_train_resampled)

print('Best Parameters are:',search.best_params_)
lr_model = search.best_estimator_
preds = lr_model.predict(X_test)
print("RF Accuracy Score -> ",accuracy_score(preds, y_test)*100)
# DECISION TREE
from sklearn.tree import DecisionTreeClassifier
pipe_decision = Pipeline([('Transformer', full_pipeline), ('DT', DecisionTreeClassifier())])

# Apply full pipeline for DT

X_train_dt= pipe_decision.fit(X_train_resampled,Y_train_resampled)

# parameter grid and grid search
param_grid_dt = [
    {'DT__criterion' : ['gini', 'entropy'],
     'DT__splitter': ['best', 'random'],
     'DT__max_depth': [4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],
     'DT__min_samples_split' : [2,3,4],
     'DT__max_leaf_nodes' : [2,5,10,15,20,30,50,70,80]
     }]


grid_search_dt = GridSearchCV(estimator = pipe_decision, param_grid = param_grid_dt, scoring = 'accuracy', refit = True, verbose = 3)


#randomized search for decision tree

parameters= parameters={'DT__max_depth' : range(10,300,20),'DT__max_leaf_nodes':
range(1,30,2),'DT__min_samples_leaf':range(1,15,3)}
    
dt_search = RandomizedSearchCV(estimator=pipe_decision, scoring='accuracy', param_distributions=parameters,cv=5, n_iter=7,refit=True,verbose=3 )

dt_search.fit(X_train_resampled,Y_train_resampled)

print('Best Parameters are:',dt_search.best_params_)

dt_model = dt_search.best_estimator_
preds = dt_model.predict(X_test)
print("RF Accuracy Score -> ",accuracy_score(preds, y_test)*100)
import joblib
filename = 'dt_model.sav'
joblib.dump(dt_model, filename)
joblib.dump(lr_model, 'lr_model.sav')

#SVM
from sklearn import svm
pipe_svm = Pipeline([('Transformer', full_pipeline), ('SVM', svm.SVC(probability=True))])

# Apply full pipeline for SVM

X_train_svm = pipe_svm.fit(X_train,y_train)

# parameter grid and grid search
param_grid_svm = [
    {'SVM__kernel' : ['linear', 'rbf', 'poly'],
      'SVM__C': [0.01,0.1,1,10,100],
      'SVM__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
      'SVM__degree' : [2,3]}]


grid_search_svm = GridSearchCV(estimator = pipe_svm, param_grid = param_grid_svm, scoring = 'accuracy', refit = True, verbose = 3)

#randomized search for SVM

parameters= parameters={'SVM__C' : np.arange(0.1,10,0.1),'SVM__gamma':
np.arange(0.01,3,0.01),'SVM__degree':range(1,5,1), 'SVM__kernel': ['linear','rbf','poly']}
    
search = RandomizedSearchCV(estimator=pipe_svm, scoring='accuracy', param_distributions=parameters,cv=5, n_iter=7,refit=True,verbose=3 )

search.fit(X_train, y_train)

print('Best Parameters are:',search.best_params_)


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
pipe_Forest = Pipeline([('Transformer', full_pipeline), ('RFC', RandomForestClassifier())])

X_train_randForest = pipe_Forest.fit(X_train_resampled,Y_train_resampled)


# parameter grid and grid search
param_grid_forest = [
    {'RFC__criterion': ['gini', 'entropy'],
     'RFC__C': [0.01,0.1,1,10,100],
     'RFC__max_depth': [4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],
     'RFC__min_sample_split': [2,3,4],
     'RFC__max_leaf_nodes': [2,5,10,15,20,30,50,70,80]
     }]

grid_search_forest = GridSearchCV(estimator = pipe_Forest, param_grid = param_grid_forest, scoring = 'accuracy', refit = True, verbose = 3)

#randomized search for Random Forest Classifier

parameters= parameters={'RFC__criterion': ['gini', 'entropy'],'RFC__max_depth':np.arange(2,150,2),'RFC__min_samples_split': range(1,5,1), 'RFC__max_leaf_nodes': range(2,80,2)}
    
r_search = RandomizedSearchCV(estimator=pipe_Forest, scoring='accuracy', param_distributions=parameters,cv=5, n_iter=7,refit=True,verbose=3 )
r_search.get_params()
r_search.fit(X_train_resampled,Y_train_resampled)

print('Best Parameters are:',r_search.best_params_)
rf_model = r_search.best_estimator_
preds = rf_model.predict(X_test)
joblib.dump(rf_model, 'rf_model.sav')
print("RF Accuracy Score -> ",accuracy_score(preds, y_test)*100)

# Neural Network
from sklearn.neural_network import MLPClassifier
pipe_nn = Pipeline([('Transformer', full_pipeline), ('nn', MLPClassifier(max_iter=1000))])


X_train_neural = pipe_nn.fit(X_train_resampled,Y_train_resampled)


# parameter grid and grid search
param_grid_nn = [
    {'nn__activation': ['tanh', 'relu'],
     'nn__solver': ['lbfgs', 'sgd','adam']
     }]

grid_search_nn = GridSearchCV(estimator = pipe_nn, param_grid = param_grid_nn, scoring = 'accuracy', refit = True, verbose = 3)


#randomized search for Random Forest Classifier
parameters= parameters={'nn__activation' : ['tanh', 'relu'], 'nn__solver': ['lbfgs', 'sgd','adam']}
    
search = RandomizedSearchCV(estimator=pipe_nn, scoring='accuracy', param_distributions=parameters,cv=5, n_iter=7,refit=True,verbose=3 )

search.fit(X_train_resampled,Y_train_resampled)

print('Best Parameters are:',search.best_params_)

nn_model = search.best_estimator_
preds = nn_model.predict(X_test)
joblib.dump(nn_model, 'nn_model.sav')
print("NN Accuracy Score -> ",accuracy_score(preds, y_test)*100)

### Evaluating model
def evaluate(pp, X, Y):
    # Prediction
    y_pred = pp.predict(X)
    # Accuracy, Precision, Recall, F1
    accuracy = accuracy_score(Y, y_pred)
    precision = precision_score(Y, y_pred)
    recall = recall_score(Y, y_pred)
    f1 = f1_score(Y, y_pred)
    print(f'\nAccuracy ({pp.steps[1][0]}):')
    print(accuracy)
    print(f'\nPrecision ({pp.steps[1][0]}):')
    print(precision)
    print(f'\nRecall ({pp.steps[1][0]}):')
    print(recall)
    print(f'\nF1 ({pp.steps[1][0]}):')
    print(f1)
    # Confusion Matrix
    cm = confusion_matrix(Y, y_pred)
    sns.heatmap(cm, cmap= 'coolwarm', annot=True, annot_kws= {'size':20})
    plt.xlabel('predicted', fontsize=16)
    plt.ylabel('actual', fontsize=16)
    plt.title(f'{pp.steps[1][0]}', fontsize=18)
    plt.show()
    # ROC curve
    y_prob = pp.predict_proba(X)
    fpr, tpr, _ = roc_curve(Y,  y_prob[:,1])
    plt.style.use('seaborn')
    plt.plot([0, 1], [0, 1], linestyle='--', color='blue')
    plt.plot(fpr, tpr, linestyle='--',color='orange', label=f'{pp.steps[1][0]}')
    plt.title(f'ROC curve ({pp.steps[1][0]})')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    # Return AUC
    return auc(fpr, tpr)

### Recommand the best model (pipeline) based on AUC
def recommand_model(pipelines, X, Y):
    best_pp = pipelines[0]
    best_auc = 0
    for pp in pipelines:
        crnt_auc = evaluate(pp, X, Y)
        if best_auc < crnt_auc:
            best_auc = crnt_auc
            best_pp = pp
    return best_pp

### Run the function to find the best model
pipe_best = recommand_model([lr_model, dt_model, rf_model, nn_model], X_test, y_test)
####################################################################################
# After you create all the models, add them into the list of pipelines right above #
# For example, [pipe_logistic, pipe_decision, pipe_svm, pipe_forest, pipe_nn]      #
# Delete these comments when you are done                                          #
####################################################################################
joblib.dump(pipe_best,'best_model.sav')
### Print the result
print('\nBest Model')
print(pipe_best[1].__class__.__name__)

nn = joblib.load("nn_model.sav")

nn.predict(X_test)



