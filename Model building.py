#WE START WITH LOGISTIC REGRESSION TO PREDICT THE BINARY OUTCOME AS Y OR N
#we see loan id does not affect the loan status so we drop that from train and test data sets
train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)

#now sklearn library requires the target varaiable in a separate dataset. so we drop 
#target var from train dataset and add another dataset 
X=train.drop('Loan_Status',1)
y=train.Loan_Status

#now we will make dummy variables. dummy variables assign 0's and 1's to categorical variables
X=pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)

#so now we divide our training data set into two parts training and validation part.
#we train the model on the train part and make predictions on validation parts
#we import sklearn function to divide our train dataset
from sklearn.model_selection import train_test_split
x_train,x_cv,y_train,y_cv=train_test_split(X,y,test_size=0.3)

#now we import logistic reg and accuracy score from sklearn and fit the model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model=LogisticRegression()
model.fit(x_train,y_train)
LogisticRegression(C=1.0,class_weight=None,dual=False,fit_intercept=True,intercept_scaling=1,
                   max_iter=100,multi_class='ovr',n_jobs=1,penalty="l2",random_state=1,
                   solver='liblinear',tol=0.0001,verbose=0,warm_start=False)

#now we predict the loan status for validation set and calculate accuracy
pred_cv=model.predict(x_cv)
accuracy_score(y_cv, pred_cv)

#so are predictions are 80% accurate the train dataset
#now we predict for test dataset
pred_test=model.predict(test)
pred_test

#now we submit the file 
submission=pd.read_csv("sample_submission_49d68Cx.csv")
submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['Loan_ID']

#we need predictions in terms of Y ,N 
submission['Loan_Status'].replace(0,'N',inplace=True)
submission['Loan_Status'].replace(1,'Y',inplace=True)

#now we submit the file in .csv format and check accuaracy in leaderboard
pd.DataFrame(submission,columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')


















