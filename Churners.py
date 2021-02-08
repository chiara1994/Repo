#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# # import and check the dataset

# In[2]:


churners = pd.read_csv("train.csv", skipinitialspace=True)
pd.set_option("display.max_rows", 40, "display.max_columns", 50)


# In[204]:


churners.head()


# In[206]:


display(churners.info(verbose = True, null_counts=False))


# In[5]:


churners.shape


# In[6]:


churners.describe()


# In[7]:


churners.isnull().sum() # no missing values


# # Data exploration

# In[8]:


churners


# Data fields
# 
# state, string. 2-letter code of the US state of customer residence
# 
# account_length, numerical. Number of months the customer has been with the current telco provider
# 
# area_code, string="area_code_AAA" where AAA = 3 digit area code.
# 
# international_plan, (yes/no). The customer has international plan.
# 
# voice_mail_plan, (yes/no). The customer has voice mail plan.
# 
# number_vmail_messages, numerical. Number of voice-mail messages.
# 
# total_day_minutes, numerical. Total minutes of day calls.
# 
# total_day_calls, numerical. Total number of day calls.
# 
# total_day_charge, numerical. Total charge of day calls.
# 
# total_eve_minutes, numerical. Total minutes of evening calls.
# 
# total_eve_calls, numerical. Total number of evening calls.
# 
# total_eve_charge, numerical. Total charge of evening calls.
# 
# total_night_minutes, numerical. Total minutes of night calls.
# 
# total_night_calls, numerical. Total number of night calls.
# 
# total_night_charge, numerical. Total charge of night calls.
# 
# total_intl_minutes, numerical. Total minutes of international calls.
# 
# total_intl_calls, numerical. Total number of international calls.
# 
# total_intl_charge, numerical. Total charge of international calls
# 
# number_customer_service_calls, numerical. Number of calls to customer service
# 
# churn, (yes/no). Customer churn - target variable.

# In[9]:


churners.churn.value_counts()  # definitely an imbalanced dataset


# In[10]:


labels= "no", "yes"
plt.pie(churners["churn"].value_counts(), labels=labels, autopct="%.1f%%", explode=[0.1]*2, shadow=True, pctdistance=0.5);
plt.title('Churners?')
plt.legend(loc='best');


# In[12]:


# Checking the price: it seems that daily calls are more expensive


fig,ax = plt.subplots()  
ax.plot(churners["total_day_minutes"], churners["total_day_charge"], label="day")
ax.plot(churners["total_eve_minutes"], churners["total_eve_charge"], label="evening")
ax.plot(churners["total_night_minutes"], churners["total_night_charge"], label="night")
ax.set_ylabel('charge') 
ax.set_xlabel('minutes')
ax.set_title("Charge/minutes")
plt.legend()
fig.set_size_inches(15,8)


# In[207]:


tot_min = churners["total_day_minutes"]+churners["total_eve_minutes"]+churners["total_night_minutes"]
tot_charge=churners["total_day_charge"]+churners["total_eve_charge"]+churners["total_night_charge"]
sns.relplot(x=tot_min, y=tot_charge, hue="churn", data=churners, kind="scatter")

# It seems that those who churn are those who spend more minutes on the phone.


# In[14]:


#Those who churn make more calls to the customer service number. Are they unsatisfied of something?
fig, axes = plt.subplots(figsize=(10, 5))
sns.barplot( x=churners["churn"], y=churners["number_customer_service_calls"]);


# In[15]:


# Let's check the correlations

fig, ax = plt.subplots()
fig.set_size_inches(18,10)
sns.heatmap(churners.corr(), annot=True, linewidths=3, linecolor='black')


# In[17]:


#checking the outliers with IQR


# In[18]:


continuos_features=churners.select_dtypes(["float64"]).columns.to_list()
for figure in continuos_features:
    plt.figure()
    plt.title(figure)
    ax = sns.boxplot(churners[figure])     # perchè vengono così?


# In[211]:


Q1 = churners.quantile(0.25)
Q3 = churners.quantile(0.75)
IQR = Q3 - Q1
outliers = (churners < (Q1 - 1.5 * IQR)) |(churners > (Q3 + 1.5 * IQR))
outliers[outliers==True].sum()


# # preprocessing

# In[22]:


churners.head(2)


# In[23]:


churners.state.value_counts() 


# In[24]:


# we have 51 classes. Is onehotencoding or get_dummies appropraite in this case? we wuold have 51 more columns. Not sure, so I simply drop it


# In[25]:


churners1=churners
churners1.drop("state", axis=1, inplace=True)


# In[26]:


# we also drop some features wich show a strong correlation

churners1.drop(["total_day_charge", "total_eve_charge", "total_night_charge", "total_intl_charge"], axis=1, inplace=True)


# In[27]:


# Transform target feature 

b= {"no":0, "yes": 1}
churners1["churn"]=churners1["churn"].map(b)

# Create y and x 

y = churners1["churn"]
x = churners1.drop('churn',errors='ignore',axis=1)

#create dummies

x = pd.get_dummies(x, columns=["area_code", "international_plan", "voice_mail_plan"])


# In[28]:


# I try to solve the imbalance problem with imblearn


rus= RandomUnderSampler()
x_rus, y_rus = rus.fit_resample(x, y)
x_rus.shape, y_rus.shape


# In[29]:


# Scaling continuos features


mms = MinMaxScaler()
mms.fit_transform(x_rus, y_rus)


# # Logistic Regression

# In[30]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import scikitplot as skplt


# In[31]:


l_reg=LogisticRegression()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=123)
distributions = {"solver": ['newton-cg', 'lbfgs', 'liblinear'], "penalty":['l2', 'l1'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter':[400,500,600]}
scoring = ["precision",'accuracy', "f1", "roc_auc"]   
log = RandomizedSearchCV(l_reg, distributions, n_iter=20, scoring=scoring, refit="roc_auc", n_jobs=-1, cv=cv, random_state=123)
log.fit(x_rus, y_rus)
print(log.best_params_, log.best_score_)


# In[32]:


y_pred= log.predict(x_rus)
prob_log= log.predict_proba(x_rus)


# In[33]:


#Curva ROC

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_rus, y_pred)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve churners')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)


# In[34]:


from sklearn.metrics import confusion_matrix, classification_report
import scikitplot as skplt


# In[35]:


skplt.metrics.plot_confusion_matrix(y_pred, y_rus)


# In[36]:


print(classification_report(y_pred, y_rus))  # too many false negatives, horrible


# # KNN

# In[38]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[104]:


param = {"n_neighbors": [i for i in range(1, 15)], "weights": ["uniform", "distance"], "algorithm":["auto", "ball_tree", "kd_tree", "brute"]}
KNN = RandomizedSearchCV(estimator=knn, param_distributions=param, n_iter=10, scoring=scoring, refit="roc_auc", n_jobs=-1, cv=cv, random_state=123)
KNN.fit(x_rus, y_rus)
print(KNN.best_params_, KNN.best_score_, KNN.best_estimator_)


# In[ ]:


ypred=KNN.predict(x_rus)


# In[212]:


skplt.metrics.plot_confusion_matrix(ypred, y_rus)   # horrible


# In[42]:


print(classification_report(ypred, y_rus))


# In[43]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_rus, ypred)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve churners')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)


# # DECISION TREE

# In[213]:


from sklearn.tree import DecisionTreeClassifier 
dtc= DecisionTreeClassifier()
parameters = {"criterion": ["gini", "entropy"], "splitter": ["best", "random"], "max_depth": [i for i in range (1,19)]}
DTC = RandomizedSearchCV(dtc, parameters, scoring=scoring, refit="roc_auc",n_jobs=-1, cv=cv, random_state=123)
DTC.fit(x_rus,y_rus)
print(DTC.best_params_, DTC.best_score_)


# In[50]:


y_pred_DTC=DTC.predict(x_rus)


# In[51]:


skplt.metrics.plot_confusion_matrix(y_pred_DTC, y_rus) 


# In[52]:


print(classification_report(y_pred_DTC, y_rus))


# In[53]:


fpr, tpr, thresholds = roc_curve(y_rus, y_pred_DTC)
plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve churners')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)


# In[ ]:


# I tried to make the predictions on the test set


# In[ ]:





# In[197]:


test=pd.read_csv("test.csv")
test1=test
test1


# In[198]:


test1.drop(["id", "state", "total_day_charge", "total_eve_charge", "total_night_charge", "total_intl_charge"], axis=1, inplace=True)


# In[199]:


test1 = pd.get_dummies(test1, columns=["area_code", "international_plan", "voice_mail_plan"])


# In[200]:


mms.fit_transform(test1)


# In[203]:


Churn_log=log.predict(test1)
Churn_KNN=KNN.predict(test1)
Churn_DTC=KNN.predict(test1)
test1["Churn_log"]=Churn_log
test1["Churn_KNN"]=Churn_KNN
test1["Churn_DTC"]=Churn_DTC
test1


# In[ ]:


# I wasn't able to make a single function 

