import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


up_df= pd.read_csv("train_upvotes.csv")
up_test=pd.read_csv("test_upvotes.csv")

#removing outliers for train
Q1 = up_df['Views'].quantile(0.25)
Q3 = up_df['Views'].quantile(0.75)
IQR = Q3 - Q1
print(Q1,Q3,IQR)
up_df = up_df[~((up_df.Views<(Q1-1.5*IQR))|(up_df.Views>(Q3+1.5*IQR)))]

Q1 = up_df['Reputation'].quantile(0.25)
Q3 = up_df['Reputation'].quantile(0.75)
IQR = Q3 - Q1
print(Q1,Q3,IQR)
up_df = up_df[~((up_df.Reputation<(Q1-1.5*IQR))|(up_df.Reputation>(Q3+1.5*IQR)))]

#removing outliers for test
Q1 = up_test['Views'].quantile(0.25)
Q3 = up_test['Views'].quantile(0.75)
IQR = Q3 - Q1
print(Q1,Q3,IQR)
up_test = up_test[~((up_test.Views<(Q1-1.5*IQR))|(up_test.Views>(Q3+1.5*IQR)))]

Q1 = up_test['Reputation'].quantile(0.25)
Q3 = up_test['Reputation'].quantile(0.75)
IQR = Q3 - Q1
print(Q1,Q3,IQR)
up_test = up_test[~((up_test.Reputation<(Q1-1.5*IQR))|(up_test.Reputation>(Q3+1.5*IQR)))]

#del columns train
del up_df['Username']
del up_df['ID']
#del columns test
del up_test['Username']
del up_test['ID']


#onehot train
up_df=pd.get_dummies(up_df,columns=['Tag'])

#onehot test
up_test=pd.get_dummies(up_test,columns=['Tag'])

#separating x and y
x = up_df.iloc[:,[0,1,2,4,5,6,7,8,9,10,11,12,13]]
y = up_df.iloc[:,3]

#train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3) 

#scale down
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_train = pd.DataFrame(x_train)

#applying algorithm
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)


#accuracy
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Test Variance score: %.2f' % r2_score(y_test, y_pred))
