#!/usr/bin/env python
# coding: utf-8

# In[37]:


#Writing all necessary libraries for readability
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier 
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import shap


# In[38]:


#Loading the dataset and defining x and y
original_df = pd.read_csv(r"train.csv")
test_df = pd.read_csv(r"test.csv")
test_ids = test_df['id']
original_df["target_numeric"] = original_df["Heart Disease"].map({
    "Presence": 1,
    "Absence": 0
})
y = original_df["target_numeric"]
x = original_df.drop(["target_numeric","Heart Disease"], axis = 1)
x_train, x_val, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[39]:


#Created a Risk indicator as people with thallium 7 will not be able to reach maximum heart rate above 150 while exercising.
#Since both are features that influencing the output, I created an interacting feature .
def preprocessing(df):
    df = df.copy()
    df.drop("id", axis=1, inplace=True)
    df["Risk indicator"] = np.where((df["Thallium"] > 6) & (df["Max HR"] < 150), 1, 0)
    df["Cholesterol"] = np.log1p(df["Cholesterol"])
    df["ST depression"] = np.log1p(df["ST depression"])
    return df


# In[40]:


Preprocessing = FunctionTransformer(preprocessing)


# In[41]:


columns = ["BP", "Max HR", "Age", "Cholesterol"]

scaling = ColumnTransformer(transformers=[
    ('num', StandardScaler(), columns)
], remainder='passthrough')


# In[42]:


# Got the values from Randomsearch in EDA file
model_xgb = XGBClassifier(n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=16,
    eval_metric='auc')


# In[43]:


full_pipeline = Pipeline([
    ('feature_eng', Preprocessing),
    ('scaling', scaling),
    ('xgb', model_xgb)
])


# In[44]:


full_pipeline.fit(x_train, y_train)
y_probs = full_pipeline.predict_proba(x_val)[:, 1]
y_pred_test = full_pipeline.predict_proba(test_df)[:, 1]
score = roc_auc_score(y_test, y_probs)
print (score)
# Creating submission file for Kaggle competition
submission = pd.DataFrame({
    "id": test_ids, 
    "Heart Disease": y_pred_test
})
submission.to_csv(r"C:\Users\MARIYA JOHN\OneDrive\Desktop\ML\project 2_Heart disease\submission.csv", index=False)


# Evaluating the model

# In[53]:


import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test,y_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'r--') # The diagonal line (random guessing)
plt.title('Receiver Operating Characteristic (ROC) for XGBOOST')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[46]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_predict = full_pipeline.predict(x_val)
cm = confusion_matrix(y_test,y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Disease'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# In[47]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
# Checking Accuracy
print("Accuracy of XGBOOST:", accuracy_score(y_test, y_predict))

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_predict))


# In[48]:


#By using Stratified K-Fold Cross Validation we can ensure that our machine learning model is evaluated fairly 
#and consistently leading to more accurate predictions and better real-world performance.
from sklearn.model_selection import StratifiedKFold
from statistics import mean, stdev
x_values = x.copy()
y_values = y.copy(
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
lst_accu_stratified = []

for train_index, test_index in skf.split(x_values, y_values):
    x_train_fold, x_test_fold = x_values.iloc[train_index], x_values.iloc[test_index]
    y_train_fold, y_test_fold = y_values.iloc[train_index], y_values.iloc[test_index]
    full_pipeline.fit(x_train_fold, y_train_fold)
    y_pred_fold = full_pipeline.predict(x_test_fold)
    acc = accuracy_score(y_test_fold, y_pred_fold)
    lst_accu_stratified.append(acc)

# Summarize results
print('List of fold-wise accuracies:', lst_accu_stratified)
print('\nMaximum Accuracy:', max(lst_accu_stratified)*100, '%')
print('Minimum Accuracy:', min(lst_accu_stratified)*100, '%')
print('Average Accuracy:', mean(lst_accu_stratified)*100, '%')
print('Standard Deviation:', stdev(lst_accu_stratified))


# In[ ]:



A low standard deviation proves the model has actually learned the underlying patterns of heart disease rather than just memorizing specific rows.

