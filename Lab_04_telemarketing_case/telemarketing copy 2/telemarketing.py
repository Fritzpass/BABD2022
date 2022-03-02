#!/usr/bin/env python
# coding: utf-8

# # Telemarketing example

# ## Data import

# In[37]:


import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt

col_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
df = pd.read_csv('ataset/crxdata.csv', header=None, names=col_names, na_values="?")


# In[38]:


df.head(3)


# In[39]:


df.isnull().sum()


# In[40]:


df.tail()


# ## Split categorical/continuos variables

# In[44]:


df.dtypes


# In[45]:


df_categorical=df.select_dtypes(include=['object'])
df_categorical.head()


# In[46]:


df_numerical=df.select_dtypes(include=['int64','float64'])
df_numerical.head()


# In[47]:


df_numerical.columns


# In[49]:


df[['A2', 'A3', 'A8', 'A11', 'A14', 'A15', 'A16']]


# ## Categorical data

# In[56]:


#Visualize Class Counts
sns.countplot(y=df.A16 ,data=df) #"target" is the name of the target column, change it accordingly to your dataset
plt.xlabel("count of each class")
plt.ylabel("classes")
plt.show()


# In[57]:


df_categorical.shape


# In[53]:


df.A11


# In[ ]:


sns.catplot(x="size", y="total_bill", data=tips)


# In[55]:


df_0=df_categorical[df['A16']==0] # records wih target==1
df_1=df_categorical[df['A16']==1] # records wih target==0


fig, axes = plt.subplots(4, 3,figsize=[12,12])
axes = axes.flatten()
i=0
for x in df_categorical.columns:
    plt.sca(axes[i]) # set the current Axes
    plt.hist([df_0[x],df_1[x]],density=True)
    plt.title(x)
    i+=1
plt.show()


# In[92]:


df_categorical.columns


# In[93]:


dummies = pd.get_dummies(df_categorical[['marital', 'education', 'default','housing', 'loan', 'contact',
       'month', 'poutcome']],drop_first=True) 

dummies.tail()


# ## Numerical data

# In[94]:


df_numerical.hist(figsize=(10,10))


# In[95]:


df_numerical.previous


# In[96]:


import math
pd.options.mode.chained_assignment = None

df_numerical['logcampaign']=df_numerical['campaign'].apply(math.log)
df_numerical['logduration']=df_numerical['duration'].apply(math.log)
df_numerical['logprevious']=df_numerical['previous'].apply(lambda x: math.log(x+1))


# In[97]:


df_numerical.head(3)


# In[98]:


df_numerical.hist(figsize=(10,10))


# In[99]:


sns.pairplot(df_numerical[['age','duration','logduration','campaign','logcampaign','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','target']], hue='target')


# ## Standarize

# In[100]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(df_numerical)
scaled_df = pd.DataFrame(scaler.transform(df_numerical))
scaled_df.columns = df_numerical.columns

scaled_df.head()


# In[27]:


scaled_df.boxplot()


# In[28]:


scaled_df.tail()


# In[29]:


X_numerical=scaled_df[['age','logduration','logprevious','logcampaign','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']]


# In[30]:


dummies.tail(3)


# In[31]:


print(dummies.shape)
print(X_numerical.shape)

dummies.tail()


# In[32]:


X=pd.concat([dummies,X_numerical], axis = 1)
X.tail()


# ### Separate Train/Test sets
# 

# In[33]:


y=df['target']


# In[34]:


y.shape


# In[35]:


from sklearn.model_selection import train_test_split

#SPLIT DATA INTO TRAIN AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size =0.30, #by default is 75%-25%
                                                    stratify=y, #preserve target propotions 
                                                    random_state= 123) #fix random seed for replicability

print(X_train.shape, X_test.shape)


# ## Models

# In[64]:


import sklearn as sk
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import f1_score


# In[37]:


# KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
parameters = {'n_neighbors':np.arange(10,1000,100)}

def hyperp_search(classifier, parameters):
    gs = GridSearchCV(classifier, parameters, cv=3, scoring = 'f1', verbose=0, n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print("f1_train: %f using %s" % (gs.best_score_, gs.best_params_))

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)

    print("f1_test: ", f1_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# In[38]:


hyperp_search(classifier,parameters)


# In[39]:


model_knn = KNeighborsClassifier(n_neighbors=10)

def roc(model,X_train,y_train,X_test,y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_probs = model.predict_proba(X_test) #predict_proba gives the probabilities for the target (0 and 1 in your case) 

    fpr, tpr, thresholds1=metrics.roc_curve(y_test,  y_probs[:,1])

    import matplotlib.pyplot as plt
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    auc = metrics.roc_auc_score(y_test, y_probs[:,1])
    print('AUC: %.2f' % auc)
    return (fpr, tpr)

fpr1,tpr1=roc(model_knn,X_train,y_train,X_test,y_test)


# In[40]:


#Tree

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
parameters = {'criterion': ['entropy','gini'], 
              'max_depth': [4,5,10],
              'min_samples_split': [20],
              'min_samples_leaf': [10]}

hyperp_search(classifier,parameters)


# In[41]:


model_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10, min_samples_split=20)

fpr2,tpr2=roc(model_tree,X_train,y_train,X_test,y_test)


# ## Plotting the tree 

# In[42]:


from sklearn import tree
r = tree.export_text(model_tree,feature_names=X_test.columns.tolist(),max_depth=3)
print(r)


# In[43]:


# Naive Bayes
from sklearn.naive_bayes import GaussianNB #or alternative NB implementations

model = GaussianNB()

model.fit(X_train, y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import f1_score
print("f1_score: ", f1_score(y_test, y_pred))

print("f1_test: ", f1_score(y_test, y_pred))


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[44]:


y_probs = model.predict_proba(X_test) #predict_proba gives the probabilities for the target (0 and 1 in your case) 

fpr3,tpr3=roc(model,X_train,y_train,X_test,y_test)


# In[45]:


# Logistic

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
parameters = {"C":[1e-4,1e-3,1e-2,1e-1,1,10], "max_iter":[1000] }


hyperp_search(classifier,parameters)


# In[46]:


model = LogisticRegression(C=1, max_iter=1000)

fpr4,tpr4=roc(model,X_train,y_train,X_test,y_test)


# In[47]:


#SVM

from sklearn.svm import SVC

classifier = SVC()
parameters = {"kernel":['linear','rbf'], "C":[0.1,100]}

hyperp_search(classifier,parameters)


# In[48]:


model = SVC(C=100, kernel='linear',probability=True)

fpr5,tpr5=roc(model,X_train,y_train,X_test,y_test)


# In[52]:


# Multi-layer Perceptron classifier

from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier()
parameters = {"hidden_layer_sizes":[(10, 5),(100,20,5)],  "max_iter": [2000], "alpha": [0.001,0.1]}

hyperp_search(classifier,parameters)


# In[53]:


model_MLP=MLPClassifier(hidden_layer_sizes=(10,5), alpha=0.1, max_iter=2000)

fpr6,tpr6=roc(model_MLP,X_train,y_train,X_test,y_test)


# In[54]:


plt.plot(fpr1, tpr1, label= "KNN")
plt.plot(fpr2, tpr2, label= "Tree")
plt.plot(fpr3, tpr3, label= "NB")
plt.plot(fpr4, tpr4, label= "Logistic")    
plt.plot(fpr5, tpr5, label= "SVM")
plt.plot(fpr6, tpr6, label= "NeuralNet")
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# ## Making predictions

# In[59]:


df2 = pd.read_csv('telemarketing_pred_withtarget.csv')


# In[60]:


#pd.options.mode.chained_assignment = None  # default='warn'

df2_categorical=df2.select_dtypes(include=['object'])

# Categorical
dummies2 = pd.get_dummies(df2_categorical[['marital', 'education', 'default','housing', 'loan', 'contact',
       'month', 'poutcome']],drop_first=True) 

# Numerical
df2_numerical=df2.select_dtypes(include=['int','float'])
df2_numerical['logcampaign']=df2_numerical['campaign'].apply(math.log)
df2_numerical['logduration']=df2_numerical['duration'].apply(math.log)
df2_numerical['logprevious']=df2_numerical['previous'].apply(lambda x: math.log(x+1))

# Scaling - WE MUST USE THE SAME SCALING OF THE TRAIN!
scaled_df2 = pd.DataFrame(scaler.transform(df2_numerical))
scaled_df2.columns = df2_numerical.columns

# Feature selection
X2_numerical=scaled_df2[['age','logduration','logprevious','logcampaign','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']]
X2=pd.concat([dummies2,X2_numerical], axis = 1)


# In[61]:


X.columns


# In[62]:


y2_pred = model_tree.predict(X2)


# In[101]:


X2.columns


# In[102]:


X.columns


# In[103]:


list(set(X) - set(X2))


# In[104]:


X2['default_yes']=0


# In[105]:


#y2_pred = model_MLP.predict(X2)

model_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10, min_samples_split=20)
model_tree.fit(X, y)

y2_pred = model_tree.predict(X2)

y2=df2.target
print("f1_test: ", f1_score(y2, y2_pred))
print(confusion_matrix(y2, y2_pred))
print(classification_report(y2, y2_pred))


# In[106]:


X2


# ## Let fix the variables

# In[107]:


X2.columns


# In[108]:


dummies = pd.get_dummies(df_categorical[['marital', 'education','housing', 'loan', 'contact',
       'month', 'poutcome']],drop_first=True) 
dummies2 = pd.get_dummies(df2_categorical[['marital', 'education','housing', 'loan', 'contact',
       'month', 'poutcome']],drop_first=True) 


X=pd.concat([dummies,X_numerical], axis=1)
X2=pd.concat([dummies2,X2_numerical], axis=1)


# In[109]:


print(X.columns)
print(X2.columns)


# In[110]:


# Retrain the model (with the entire dataset)

model_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10, min_samples_split=20)
model_tree.fit(X, y)

y_tree = model_tree.predict(X2)

y2=df2.target
print("f1_test: ", f1_score(y2,y_tree))
print(confusion_matrix(y2,y_tree))
print(classification_report(y2,y_tree))


# In[111]:


from sklearn.neural_network import MLPClassifier

model_MLP=MLPClassifier(hidden_layer_sizes=(10,5), alpha=0.1, max_iter=2000)
model_MLP.fit(X, y)

y_MLP = model_MLP.predict(X2)

print("f1_test: ", f1_score(y2,y_MLP))
print(confusion_matrix(y2,y_MLP))
print(classification_report(y2,y_MLP))


# In[112]:


predictions=pd.DataFrame()

predictions['tree']=y_tree
predictions['MLP']=y_MLP

predictions.to_csv('telemarketing_predictions.csv')


# In[ ]:




