#Each variable is compared with loan status 
Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))

Married=pd.crosstab(train['Married'],train['Loan_Status'])
Married.div(Married.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()


Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Dependents.div(Dependents.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))

Education=pd.crosstab(train['Education'],train['Loan_Status'])
Education.div(Education.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))

Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Self_Employed.div(Self_Employed.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))

Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))

Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))

#we calc mean income of people whose loan is approved and not approved
train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
bins=[0,2500,4000,6000,81000]
groups=['low','average','high','very high']
train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=groups)

Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))

#we see income does not indicate chance of loan approval which contradicts our assumption
#that higher income would inc chance of loan approval
#we do the same with coapplicant income
train.groupby('Loan_Status')['CoapplicantIncome'].mean().plot.bar()
bins=[0,1000,3000,42000]
groups=['low','average','high']
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=groups)

Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
                                                                                    
#we see if coapp income is less chances of loan approval is high. But this is not true since many applicant dont have 
#any coapplicant and hence loan approval is not dependent on it. we create another combined income to see its effect on loan approval
train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000]
groups=['low','average','high','very high']
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=groups)

Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))

#we see that loan is getting approved for people with total income being average,high&very high than with low income
#now we see loan amount to loan status
bins=[0,100,200,700]
groups=['low','average','high']
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=groups)

LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))

#so we see the proportion of loan getting approved is high for low and average loan amount than high loan amount which supports 
#our hypothesis that chances of loan getting approved is high for less loan amount

#Now we will drop all bins and change dependents variable value of 3+ to 3 and target variable as 0 and 1.
#models like logistic regression take numerical values as input
train=train.drop(['Income_bin','Coapplicant_Income_bin','Total_Income','Total_Income_bin','LoanAmount_bin'],axis=1)
train['Dependents'].replace('3+',3,inplace=True)
test['Dependents'].replace('3+',3,inplace=True)
train['Loan_Status'].replace('N',0,inplace=True)
train['Loan_Status'].replace('Y',1,inplace=True)

#now we will look at the correlation of all numerical variables. 
#we use heat map to visualize the variables
matrix=train.corr() 
ax=plt.subplots(figsize=(9,6))
sns.heatmap(matrix,vmax=0.8,square=True,cmap="BuPu");

#we see that the variables applicant income&loan amt, credit history&loan status,loan amt&coappli income are correlated



























