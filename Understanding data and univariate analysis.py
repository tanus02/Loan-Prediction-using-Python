#Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

#importing train and test data
train=pd.read_csv("train_ctrUa4K.csv")
test=pd.read_csv("test_lAUu6dG.csv")

#making a copy of data sets
train_original=train.copy()
test_original=test.copy()

#basic details
train.columns
test.columns
train.dtypes
train.shape
test.shape

#Univariate analysis
#loan status
train['Loan_Status'].value_counts()
train['Loan_Status'].value_counts(normalize=True) #in terms of proportions
train['Loan_Status'].value_counts().plot.bar()  #bar plot

#visualizing each variable individually
#categorial - gender,married,self employed,credit history,loan status
#ordinal features-dependents,education,property area
#numerical- applicantincome,coapplicant income,loan amount,loan amount term

plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10),title="Gender")
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title="Married")
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title="Self employed")
plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title="Credit history")
plt.show()

#ordinal variables
plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(26,4),title="Dependents")
plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title="Education")
plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title="Property area")
plt.show()

#numerical variables
plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome'])
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))
plt.show()
train.boxplot(column='ApplicantIncome',by='Education')
plt.suptitle("")

plt.subplot(121)
sns.distplot(train['CoapplicantIncome'])
plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16,5))
plt.show()

plt.subplot(121)
df=train.dropna()
sns.distplot(train['LoanAmount'])
plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))
plt.show()



train.isna().sum()
df['Gender'].value_counts(normalize=True).plot.box(figsize=(20,10),title="Gender")




















