import pandas as pd 
from xgboost import XGBClassifier
df=pd.read_csv("diabetes.csv")
#print(df.info)
#print(df.describe())
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for i in zero_cols:
    mean=df[i].mean()
    df[i]=df[i].replace(0,mean)
for i in zero_cols:
    zeros=(df[i]==0).sum()
    #print(f"{i} has {zeros} zero values")  

x=df.drop(columns="Outcome")
y=df["Outcome"] 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=17)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#real model teaching
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score ,confusion_matrix,classification_report
print("accuracy : ",accuracy_score(y_test,y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True,fmt='d',cmap='Reds')
plt.xlabel("ypredicted")
plt.ylabel("actual values")
plt.show()

x_scaled=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,random_state=17,test_size=0.2)
model = LogisticRegression(max_iter=1000)
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

import pickle

with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
