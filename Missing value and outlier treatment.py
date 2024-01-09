#MISSING IMPUTE VALUES
train.isnull().sum()

#so there are missing values in gender,married,dependents,self employed,loan amount,loan amt term,credit history
#for numerical variabels - we use mean,median to impute
#for categorical variables -we use mode to impute
#now we use mode for gender,married,dependents,credit history,self employed
train['Gender'].fillna(train['Gender'].mode()[0],inplace=True)
train['Married'].fillna(train['Married'].mode()[0],inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)

#so now we fill the loan amount term also with mode values
train['Loan_Amount_Term'].value_counts()
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)

#now we fill loan amount with median values as mean wont fit in due to extreme values
train['LoanAmount'].fillna(train['LoanAmount'].median(),inplace=True)

#now we check if there are any missing values
train.isnull().sum()

#now we use same approach to test data
test['Gender'].fillna(train['Gender'].mode()[0],inplace=True)
test['Married'].fillna(train['Married'].mode()[0],inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(),inplace=True)

#OUTLIER TREATMENT
#we know that mean and sd have a strong effect due to outliers. hence we must reduce the outliers

#the outliers data is right skewed. to remove the skewness we take log transformation.
train['LoanAmount_log']=np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=10)
test['LoanAmount_log']=np.log(train['LoanAmount'])

#now the distribution looks normal and the effect of extreme values has been significantly subsided
 



















