
# coding: utf-8

# This is classification modeling excercise on Titanic data from Kaggle using python. 
# This is based on the excellent kernel by Manav Sehgal https://www.kaggle.com/startupsci/titanic-data-science-solutions
# I have made a lot of changes to the original kernel to improve the predictions and to learn different techniques.

# In[265]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
get_ipython().magic('matplotlib inline')


# In[218]:

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# In[219]:

train_df.head()


# In[220]:

train_df.describe()


# In[221]:

train_df.isnull().any()


# Age, Cabin and Embarked columns have some nulls in the training data.

# In[222]:

test_df.isnull().any()


# Age, Fare and Cabin columns have some nulls in the test data

# In[223]:

train_df.info()


# Looks like Cabin columns has lot of nulls so we will remove this column later on as this should not affect our results. Next lets see some frequency distributions in the data

# In[224]:

train_df['Sex'].value_counts()


# In[225]:

train_df['SibSp'].value_counts()


# In[226]:

train_df['Parch'].value_counts()


# In[227]:

train_df['Survived'].value_counts()


# Check categorical column distributions

# In[228]:

train_df.describe(include=['O'])


# Check the number of nulls. This could be deduced from train_df.info() command above but this is just to learn 

# In[229]:

train_df['Age'].isnull().sum()


# In[230]:

train_df['Cabin'].isnull().sum()


# In[231]:

train_df['Embarked'].isnull().sum()


# Create a correlation matrix for all the columns. 

# In[232]:

train_df.corr()


# Pclass and Fare shows some linear correlation with Survived. I was expecting Age to be correlated as well but probably correlation is not linear. 
# Let's analyze the relationship with some of the features with survive. 

# In[233]:

train_df[['Survived','Pclass']].groupby('Pclass').mean().sort_values(by='Survived', ascending = False)


# In[234]:

train_df[['Survived','Sex']].groupby('Sex').mean().sort_values(by='Survived', ascending = False)


# In[235]:

train_df[['Survived','SibSp']].groupby('SibSp').mean().sort_values(by='Survived', ascending = False)


# In[236]:

train_df[['Survived','Parch']].groupby('Parch').mean().sort_values(by='Survived', ascending = False)


# In[237]:

train_df[['Survived','Embarked']].groupby('Embarked').mean().sort_values(by='Survived', ascending = False)


# We will need to fill null fare value in test dataset so let us see if there is any relationship with ticket class. Sure enough there is a relationship so we can use mean for each ticket class to fill missing fare values.

# In[269]:

train_df[['Fare','Pclass']].groupby('Pclass').mean().sort_values(by='Fare', ascending = False)


# Let us split the training data into survived and pershed dataframes to do some analysis

# In[238]:

survived = train_df[train_df['Survived'] == 1]
perished = train_df[train_df['Survived'] == 0]  
survived.tail()                    


# In[239]:

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
axes = plt.gca()
axes.set_xlim([0,80])
axes.set_ylim([0,60])


ax1.hist(survived['Age'].dropna(),20,normed=False,facecolor = 'Green')
plt.xlabel('Age')
ax1.grid(True)
ax1.set_xlabel('Age')
ax1.set_ylabel('Count')
ax1.set_title('Survived')


ax2.hist(perished['Age'].dropna(),20,normed=False,facecolor = 'Green')
plt.xlabel('Age')
ax2.grid(True)
ax1.set_xlabel('Age')
ax2.set_title('Perished')

plt.show()


# Younger passengers had higher chances of survival

# In[240]:

survived['Sex'].value_counts().plot(kind='bar')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Survived')
plt.show()


# In[241]:

perished['Sex'].value_counts().plot(kind='bar')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Perished')
plt.show()


# Female passengers had higher chances of survival. 
# 
# Let us check the fare distribution next. Some passengers paid higher price for their tickets.

# In[242]:

train_df.boxplot(column='Fare')


# Check any relationship between class and the port of embarkation. Looks like most passengers boarded at port 'S' and were in ticket class 2.

# In[243]:

emb_counts = pd.crosstab(train_df.Pclass, train_df.Embarked)
emb_pcts = emb_counts.div(emb_counts.sum(1).astype(float), axis=0)
emb_pcts.plot(kind='bar', stacked=True)


# We need to fill the missing values in Age. For this let us extract title from the name column and see the median age for each title.

# In[244]:

train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[245]:

train_df[['Age','Title']].groupby('Title').median().sort_values(by='Age', ascending = False)


# Based on the above and using some assumptions, let us standardize the titles and then map them to integers

# In[246]:

train_df['Title'] = train_df['Title'].replace(['Mlle','Ms'],'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
train_df['Title'] = train_df['Title'].replace('Countess', 'Mrs')
train_df['Title'] = train_df['Title'].replace(['Don','Jonkheer','Rev','Dr','Sir','Major'], 'Sr')

test_df['Title'] = test_df['Title'].replace(['Mlle','Ms'],'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')
test_df['Title'] = test_df['Title'].replace('Countess', 'Mrs')
test_df['Title'] = test_df['Title'].replace(['Don','Jonkheer','Rev','Dr','Sir','Major'], 'Sr')


# In[247]:

title_mapping = {"Master": 1, "Miss": 2, "Mr":3, "Mrs": 4, "Sr":5,"Lady": 6, "Col": 7,"Capt":8}

train_df['Title'] = train_df['Title'].map(title_mapping)
test_df['Title'] = test_df['Title'].map(title_mapping)
train_df.head()


# Find the median age in each title group and pass that to a dictionary object for mapping. Fill the missing age values in train and test datasets using this mapping.

# In[248]:

title_age = train_df.groupby('Title').Age.median().to_dict()

title_age


# In[249]:

train_df['Age'] = train_df.Age.fillna(train_df.Title.map(title_age))
test_df['Age'] = test_df.Age.fillna(train_df.Title.map(title_age))


# In[272]:

class_fare = train_df.groupby('Pclass').Fare.mean().to_dict()
test_df['Fare'] = test_df.Fare.fillna(train_df.Pclass.map(class_fare))


# Drop Ticket, Name and Cabin columns from the datasets

# In[250]:

train_df = train_df.drop(['Ticket','Name','Cabin'], axis = 1)
test_df = test_df.drop(['Ticket','Name','Cabin'], axis = 1)


# Create age groups and assign them to integers

# In[251]:

def modif(row):
    if row['Age'] <= 6:
        return 1
    elif (row['Age'] > 6) & (row['Age'] <= 12):
        return 2 
    elif (row['Age'] > 12) & (row['Age'] <= 18):
        return 3 
    elif (row['Age'] > 18) & (row['Age'] <= 48):
        return 4 
    elif (row['Age'] > 48) & (row['Age'] <= 56):
        return 5     
    elif (row['Age'] > 56) & (row['Age'] <= 62):
        return 6   
    elif (row['Age'] > 62) & (row['Age'] <= 72):
        return 6
    else:
        return 7
    
train_df['Age'] = train_df.apply(modif, axis = 1)
test_df['Age'] = test_df.apply(modif, axis = 1)

train_df.head()


# Fill missing port value in train and test dataset from frequent port value 

# In[252]:

freq_port = train_df.Embarked.dropna().mode()[0]
train_df['Embarked'] = train_df['Embarked'].fillna(freq_port)
test_df['Embarked'] = test_df['Embarked'].fillna(freq_port)


# Since Embarked is not ordinal, let us get dummy values

# In[253]:

train_df = pd.get_dummies(train_df)
train_df.head()


# Prepare the data for classification but first selecting the features and then splitting into train and test by 33%

# In[255]:

features = ['Pclass','Sex_female','Sex_male','Age','SibSp','Parch','Embarked_C','Embarked_Q','Embarked_S','Title','Fare']
X = train_df[features].copy()
Y = train_df.iloc[:,1].copy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=324)


# Let us first build a decition tree model

# In[256]:

survival_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
survival_classifier.fit(X_train, Y_train)


# In[257]:

predictions = survival_classifier.predict(X_test)


# In[258]:

predictions[:10]


# In[259]:

Y_test[:10]


# In[260]:

accuracy_score(y_true = Y_test, y_pred = predictions)


# Let us check XGBoost

# In[261]:

gbm = xgb.XGBClassifier(max_depth=10, n_estimators=300, learning_rate=0.001).fit(X_train, Y_train)
predictions = gbm.predict(X_test)
accuracy_score(y_true = Y_test, y_pred = predictions)


# Logistic Rregression

# In[262]:

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
predictions = logreg.predict(X_test)
accuracy_score(y_true = Y_test, y_pred = predictions)


# Random Forest

# In[264]:

forest = RandomForestClassifier(criterion='entropy',n_estimators = 100, random_state = 1, n_jobs=2)
forest.fit(X_train,Y_train)
predictions = forest.predict(X_test)
accuracy_score(y_true = Y_test, y_pred = predictions)


# Support vector machine

# In[266]:

svm_classifier = svm.SVC(kernel='rbf', random_state=0)
svm_classifier.fit(X_train,Y_train)
predictions = svm_classifier.predict(X_test)
accuracy_score(y_true = Y_test, y_pred = predictions)


# Let us check SVM with hyperparamter tunning

# In[267]:

pipe_svc = Pipeline([('scl', StandardScaler()),
                    ('clf', SVC(random_state=1))])
param_range = [0.0001,1,1000.0]
param_grid = [{'clf__C': param_range,
             'clf__gamma': param_range,
             'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                 param_grid=param_grid,
                 scoring='accuracy',
                 cv=10,
                 n_jobs = -1)
gs = gs.fit(X_train,Y_train)
print(gs.best_score_)
print(gs.best_params_)


# Looks like XGboost has given the best results so far
