#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv(r"train.csv")
test_df = pd.read_csv(r"test.csv")
test_ids = test_df['id']
df


# In[ ]:


df.isna().sum()


# In[ ]:


df.dtypes


# In[ ]:


df.head(5)


# In[ ]:


df.describe()


# I did df.describe showed me that 
# ST depression - around 25% to 75% has it around 1-2 and even though max and min are potential outliers and max is large outlier,   they aren't medically impossible . so i'm keeping them as it is but i will evalute them further 
# Cholesterol- around 25% to 75% has it around 223 - 269 and even though max and min are potential outliers, they aren't medically   impossible . so i'm keeping them as it is but i will evalute them further .
# BP - around 25% to 75% has it around 120 to 140, and like cholestrol the min and max aren't medically impossible, so i'm not       treating this as ouliers, but i will evalute it .
# Max HR,Slope of ST,Number of vessels fluro - they're are also having slight outliers like above 
# 

# In[ ]:


#deleting as id is not required and the values are huge which can affect 
df.drop("id", axis=1, inplace=True)
test_df.drop("id", axis=1, inplace=True)


# In[ ]:


#Evaluating histograms to see if any of data are skewed
import matplotlib.pyplot as plt
df.hist(bins = 15, figsize=(14,10))
plt.show()


# Max HR is slightly left skewed, cholestrol  is slightly right skewed
# ST depression is extremely right skewed, so i choose log transformation

# In[ ]:


#Evaluating outliers by Boxplot and Scatterplot
import seaborn as sns
sns.boxplot( data = df, x = "Heart Disease" , y ="ST depression" )


# In[ ]:


sns.scatterplot( data = df, x ="ST depression", y = "Heart Disease" )


# In[ ]:


import seaborn as sns
sns.boxplot( data = df, x = "Max HR" , y ="Heart Disease" )


# In[ ]:


sns.scatterplot( data = df, x = "Max HR" , y ="Heart Disease" )


# In[ ]:


import seaborn as sns
sns.scatterplot( data = df, x ="Cholesterol", y = "Heart Disease"  )


# In[ ]:


#Now doing correlation matrix to see if there are features highly correlated.
df['target_numeric'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
matrix = df.corr() 
plt.figure(figsize=(8,6))
sns.heatmap(matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# There is a negative correlation between Thalium and Max HR .
# And a positive correlation of .61 between Thallium and target numeric and a correaltion of 0.46 betweenchest pain type and target numeric.
# And 0.44 between Exercise angina and target numeric

# In[ ]:


df.head(10)


# In[ ]:


df['target_numeric'].value_counts()


# In[ ]:


import numpy as np
df["Risk indicator"] = np.where((df["Thallium"] > 6) & (df["Max HR"] < 150), 1, 0) # Since they have a negative correlation, creating an indicator
test_df["Risk indicator"] = np.where((test_df["Thallium"] > 6) & (test_df["Max HR"]  <150), 1 , 0)
df["Cholesterol"] = np.log1p(df["Cholesterol"]) #Log transforming to handle skewness
test_df["Cholesterol"] = np.log1p(test_df["Cholesterol"])
df["ST depression"] = np.log1p(df["ST depression"])#Log transforming to handle skewness
test_df["ST depression"] = np.log1p(test_df["ST depression"])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[["BP", "Max HR", "Age"]] = scaler.fit_transform(df[["BP", "Max HR", "Age"]])
test_df[["BP", "Max HR", "Age"]] = scaler.transform(test_df[["BP", "Max HR", "Age"]])


# In[ ]:


df.hist(bins = 10,figsize = (14,10))
plt.show()#i did st depression log transform 
#and scaling on max hr and cholestrol instead of nth pwer or exp because it make the dataset unstable


# In[ ]:


from sklearn.model_selection import train_test_split
y = df["target_numeric"]
x = df.drop(["target_numeric","Heart Disease"], axis = 1)
x_train,x_val,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)


# In[ ]:


from sklearn.linear_model import LogisticRegression
model_log = LogisticRegression( penalty = 'l2', solver = 'newton-cg', max_iter = 50, verbose = 1, n_jobs = -1)
model_log.fit(x_train,y_train)
y_pred_log = model_log.predict_proba(x_val)[:,1]
y_pred_test_log = model_log.predict_proba(test_df)[:,1]
y_pred_test_log[:100]


# In[ ]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
x_tune = x_train.sample(frac=0.1, random_state=42)
y_tune = y_train.sample(frac=0.1, random_state=42)
param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1]
}
random_search = RandomizedSearchCV(
    XGBClassifier(tree_method='hist', eval_metric='auc'), 
    param_distributions=param_dist, 
    n_iter=5,
    cv=3, 
    n_jobs=-1
)
random_search.fit(x_tune, y_tune)
best_xgb_model = random_search.best_estimator_
best_parameters = random_search.best_params_
print("Final Parameters being used:", best_parameters)


# In[ ]:


from xgboost import XGBClassifier 
model_XGB = XGBClassifier(n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=16,
    eval_metric='auc')
model_XGB.fit(x_train,y_train)
y_pred_XGB = model_XGB.predict_proba(x_val)[:,1]
y_pred_test_XGB = model_XGB.predict_proba(test_df)[:,1]
y_pred_XGB[0]


# Evaluating model's performance

# In[ ]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
y_pred_log_ = model_log.predict(x_val)
y_pred_XGB_ = model_XGB.predict(x_val)
# Checking Accuracy
print("Accuracy of Logistic regression:", accuracy_score(y_test, y_pred_log_))
print("Accuracy of XGboost:",accuracy_score(y_test,y_pred_XGB_))

# nClassification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_log_))
print(classification_report(y_test,y_pred_XGB_))


# In[ ]:


print(set(x.columns) - set(test_df.columns))
print(set(test_df.columns) - set(x.columns))


# In[ ]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred_log)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'r--') # The diagonal line (random guessing)
plt.title('Receiver Operating Characteristic (ROC) for Logistic regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test,y_pred_XGB)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'r--') # The diagonal line (random guessing)
plt.title('Receiver Operating Characteristic (ROC) for Logistic regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred_log_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Disease'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred_XGB_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Disease'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# From the confusion matrix, it is clear that the XGBOOST model outperforms Logistic regression. 
# While logistic model got 6764 cases where it predicted Disease when the true value was No Disease, XGBOOST only got 6680 cases in the same genre.
# It is also evident in the other 3 categories as well. So I'm proceeding with XGBOOST model

# In[ ]:


import shap
from shap import LinearExplainer
explainer = shap.LinearExplainer(model_log,x_train)
shap_values_log = explainer(x_val)
shap.plots.beeswarm(shap_values_log)


# In[ ]:


from shap import TreeExplainer
explainer = shap.TreeExplainer(model_XGB,x_train)
shap_values_XGB = explainer(x_val)
shap.plots.beeswarm(shap_values_XGB)


# Logistic Regression is a Linear model.In Logistic Regression, the model assigns a single weight to each feature.
# For "Sex" it is as 0 or 1. There is no "interaction" with other features. 
# Every person with a "1" for Sex gets the exact same "push" on the SHAP scale, hence the perfectly vertical line.
# 

# XGBOOST is a non-linear model.Non-linearity means the model doesn't assume a "straight line" relationship.
# It uses a series of "If/Then" splits (trees).
# The spread represents the model's ability to find these complex, non-linear relationships.

# In[ ]:


1.The beeswarm plot indicates that Thallium, Chest pain type and Max HR were the features that impacted on output in both models.
2.Since Logistic regression is a inear model, it didn't catch the interactions betwwen features well. 
3.I can see that the Thallium, chest pain type and Sex are treated as categorical variable as they're in the Logistic plot . 
meanwhile Thallium, chest pain type and has spread in the XGB boost plot. it gives a more reliable result interacting with features.
4. Also chest pain type 1,2,3 are theones with less risk and chest pain type 4 is the one that causes prediction of 1, ie Heart Disease


# In[ ]:


shap.plots.waterfall(shap_values_XGB[2])# This waterfall can hek=lp to analyse the output of a particular row
y_pred_XGB[2]# Here I am analysing the output is 1 , means having Heart Disease


# In[ ]:


shap.plots.waterfall(shap_values_XGB[1]) 


# In[ ]:


#A dependence scatter plot shows the effect a single feature has on the predictions made by the model.
#This means there are non-linear interaction effects in the model between feature
shap.plots.scatter(shap_values_XGB[:,"Thallium"])


# In[ ]:


#To show which feature may be driving these interaction effects we can color scatter plot by another feature. 
#If we pass the entire shap_values  to the color parameter then the scatter plot attempts to pick out the feature column with the strongest interaction 
shap.plots.scatter(shap_values_XGB[:,"Thallium"], color= shap_values_XGB)


# In[ ]:


#This plot made me realise Thallium is interacting with Sex.@Thallium = 3 
#Females are at less risk than male, and when @Thallium at 7 it shows that femaile is at more risk. 
#Researching about it made me realise women generally have a lower "starting" risk for obstructive coronary artery disease compared to men of the same age.
#Given that , in female A 7 is highly unusual and often means the disease is quite advanced.

