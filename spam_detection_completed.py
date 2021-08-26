import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle

##Step1: Load Dataset
dataframe = pd.read_csv("df.csv")
print(dataframe.describe())

##Step2: Split in to Training and Test Data

x = dataframe["sentence"]
y = dataframe["type"]

# x_train,y_train = x[0:4457],y[0:4457]
# x_test,y_test = x[4457:],y[4457:]

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3)

##Step3: Extract Features
cv = CountVectorizer()  
features = cv.fit_transform(x_train)

##Step4: Build a model
tuned_parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}

model = GridSearchCV(svm.SVC(), tuned_parameters)

model.fit(features,y_train)

# model= MultinomialNB()
# model.fit(x_train,y_train)

# print(model.best_params_)
#Step5: Test Accuracy
print(model.score(cv.transform(x_test),y_test))



