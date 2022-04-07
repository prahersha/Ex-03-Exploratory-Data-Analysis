import pandas as pd
import numpy as np
import seaborn as sns
#graphical package
df=pd.read_csv("C:\\Users\\praha\\OneDrive\\Desktop\\titanic_dataset.csv")
df.info()
df.head()
df.isnull().sum()
df.drop(columns=['Cabin'],inplace=True)
df.isnull().sum()
df['Age']=df['Age'].fillna(df['Age'].median())
df.boxplot()
df.isnull().sum()
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
#Non numerical data so we have used mode operation
df['Embarked'].value_counts()
df['Pclass'].value_counts()
df['Survived'].value_counts()
#analyzing the data
sns.countplot(x='Survived',data=df)
sns.countplot(x='Pclass',data=df)
sns.countplot(x='Sex',data=df)
df.info()
sns.displot(df['Fare'])
sns.countplot(x='Pclass',hue='Survived',data=df)
sns.countplot(x='Sex',hue='Survived',data=df)
sns.displot(df[df['Survived']==0]['Age'])
sns.displot(df[df['Survived']==1]['Age'])
pd.crosstab(df['Pclass'],df['Survived'])
pd.crosstab(df['Sex'],df['Survived'])
df.corr()
sns.heatmap(df.corr(),annot=True)


 
