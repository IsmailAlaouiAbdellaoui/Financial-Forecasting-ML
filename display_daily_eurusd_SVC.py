# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 16:51:00 2017

@author: Alaoui
"""
from __future__ import division
import numpy
import pandas as pd
from sklearn.externals import joblib
import pickle
#import plotly.plotly as py
#import plotly.figure_factory as FF
from sklearn import svm
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit, RandomizedSearchCV
from datetime import datetime
import matplotlib.pyplot as plt 
import matplotlib.finance as fin
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn import cross_validation
from sklearn import tree

from sklearn.feature_selection import RFE


data = []
output_list = []
output_list_date = []

input_list_temp = []
in_list = []
in_list_date = []

testing_output = []
testing_output_date = []

testing_input = []
testing_input_date = []

output_prediction = []
#out_test



#data = numpy.genfromtxt('eur_usd_daily.csv',delimiter=',')
#print data
#Loading data from csv file
def load_format_csv():
    global data
    global df
    df = pd.read_csv('eur_usd_daily_v2.csv', sep =',', header = None)
#    del df[1]
#    data = list(df.values)

    return df

#Function to plot ( not required)
def load_plot():   
    df = load_format_csv()
    fig,ax = plt.subplots()
    fin.candlestick2_ohlc(ax,df[1],df[2],df[3],df[4],width=0.6)
    plt.show()

#Function to generate input/output for training and testing
def generate_input_output():
    global output_list
    global output_list_date
    global input_list_temp
    global in_list
    global testing_input
    global testing_input_date
    
    global testing_output
    global testing_output_date
    
    global in_list_date
    
    df = load_format_csv()
    
#    del df[0]
#    print df.values
    data = list(df.values)#ALL values in list
#    print len(data[0])
    for item in data:
#        del item[0]
        item[1] = round(item[1],5)
        item[2] = round(item[2],5)
        item[3] = round(item[3],5)
        item[4] = round(item[4],5)
        item[5] = int(item[5])
        
#        print item
    
#    print len(data)
    
#    while(i<len(data)):
#        input_list[i] = [[data[i][1],data[i][2],data[i][3],data[i][4],data[i][5]]
#        i = i + 1
    df2 = pd.DataFrame(data,columns=['date','open','high','low','close','volume'])
    df2_date = pd.DataFrame(data,columns=['date','open','high','low','close','volume'])
#    print df2.values[2053] ###training stops at date 2015.12.31 ( 31 décembre 2015)
    #31 12 2015
#    for item in data:
#        if(item[0] == "2015.12.31"):
#            print "found 31 12 2015"
#            print i
#            i = i+1
#        else:
#            i = i+1
        
    print("length of data : " + str(len(data)))
    b1 = int(0.8*len(data))#to do a 80/20 split
#    print(b1)

    del df2['date']
    
    df_input = pd.DataFrame(data[0:b1],columns=['date','open','high','low','close','volume'])
    del df_input['date']
    del df_input['volume']
    
    df_input_date = pd.DataFrame(data[0:b1],columns=['date','open','high','low','close','volume'])
    del df_input_date['volume']
    
    input_list_temp = df_input.values.tolist()
    input_list_temp_date = df_input_date.values.tolist()
    
    
    df_input_test = pd.DataFrame(data[-(len(data)-b1):],columns=['date','open','high','low','close','volume'])
    del df_input_test['date']
    del df_input_test['volume']
    
    df_input_test_date = pd.DataFrame(data[-(len(data)-b1):],columns=['date','open','high','low','close','volume'])
    del df_input_test_date['volume']
    
    
#    print "length of input values " + str(len(df_input_test.values))
    input_test_temp = df_input_test.values.tolist()
#    
    input_test_temp_date = df_input_test_date.values.tolist()
    
    k=0
    while(k<(len(data)-b1)-6):
        testing_input.append(input_test_temp[k]+input_test_temp[k+1]+input_test_temp[k+2]+input_test_temp[k+3]+input_test_temp[k+4]+input_test_temp[k+5])
        k += 1
    print "length of testing_input"+ str(len(testing_input))
    
    k=0
    while(k<(len(data)-b1)-6):
        testing_input_date.append(input_test_temp_date[k]+input_test_temp_date[k+1]+input_test_temp_date[k+2]+input_test_temp_date[k+3]+input_test_temp_date[k+4]+input_test_temp_date[k+5])
        k += 1
    
    
#    df_test = pd.pivot_table(df_input)
    
#    print input_list_temp
#    print len(input_list_temp)
#    test = []
    j = 0
    while(j<b1-6):#342 normally
        in_list.append(input_list_temp[j]+input_list_temp[j+1]+input_list_temp[j+2]+input_list_temp[j+3]+input_list_temp[j+4]+input_list_temp[j+5])
        j += 1
#    print len(in_list)
        
    j=0
    while(j<b1-6):#342 normally
        in_list_date.append(input_list_temp_date[j]+input_list_temp_date[j+1]+input_list_temp_date[j+2]+input_list_temp_date[j+3]+input_list_temp_date[j+4]+input_list_temp_date[j+5])
        j += 1
    
    
     
    i = 0
#    print test
    print "length of df2 "+ str(len(df2.values))
    while (i<b1):
        if (df2.loc[i][3] - df2.loc[i][0]) < 0:
           output_list.append('-1')
           i += 1 
        else:
            output_list.append('1')
            i += 1
            
            
    k=0
    while (k<b1):
        if (df2_date.loc[k][4] - df2_date.loc[k][1]) < 0:
           output_list_date.append(df2_date.loc[k][0]+' -1')
           k += 1 
        else:
            output_list_date.append(df2_date.loc[k][0]+' 1')
            k += 1
    
    
    print "i : "+ str(i)
    l = i
    while(i<len(data)):
        if (df2.loc[i][3] - df2.loc[i][0]) < 0:
           testing_output.append('-1')
           i += 1 
        else:
            testing_output.append('1')
            i += 1
            
    while(l<len(data)):
        if (df2_date.loc[l][4] - df2_date.loc[l][1]) < 0:
           testing_output_date.append(df2_date.loc[l][0]+' -1')
           l += 1 
        else:
            testing_output_date.append(df2_date.loc[l][0]+' 1')
            l += 1         
            
        
    
    output_list = output_list[6:]
    output_list_date = output_list_date[6:]
    
    testing_output = testing_output[6:]
    testing_output_date = testing_output_date[6:]
#    testing_output = output_list[-295:]
#    print output_list
    print "length of testing output : " + str(len(testing_output))
#    print len(output_list)
#    print input_list_temp
    
#    for item in input_list_temp:
#        print item
#    print len(input_list_temp)
    print "done generating input/output lists"
    
    
#Function to train the set , creates a predictive model called "training"            
def train_set():
    global testing_input
    global in_list
    global output_list
    global out_test
    
    print len(in_list)
    print len(output_list)
    print len(testing_input)
    
#    print("creating the model ...")
    training = svm.SVC(C=60, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma=0.001, kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
#    training = LogisticRegression()
    x = testing_input+in_list
    y = output_list+testing_output
    parameters_svm = {'kernel':('poly','poly','poly','poly','poly','poly','poly','poly','rbf'), 'C':[100,105,110,115,120,125,130,135,150],'gamma':[0.001,0.01,0.1,0.8,0.9,1,1.1,1.2,3],'degree':[2,3,3,3,3,3,3,3,3]}
    params1 = {'kernel':('poly','poly'), 'C':[105],'gamma':[0.001,0.01],'degree':[2,3]}
    parameters_random_tree = {'max_depth':range(3,20)}
    
    my_cv = TimeSeriesSplit(n_splits=5).split(x)
    print(str(len(testing_input)))
    print(str(len(testing_output)))
    
    
#    grid = GridSearchCV(SVC(), cv=my_cv, param_grid=params1)
#    grid = RandomizedSearchCV(SVC(),parameters,cv=my_cv,scoring='accuracy',n_iter=10, random_state=9,n_jobs=3)
#    grid = RandomizedSearchCV(tree.DecisionTreeClassifier(),parameters_random_tree,cv=my_cv,n_iter=17,random_state=9)
    
#    print training
    print("training the model ...")
#    print("first element of input x is : "+str(x[50]))
#    print("first element of output y is : "+str(y[50]))
    
#    for train_index, test_index in my_cv:
#        print("TRAIN:", train_index, "TEST:", test_index)
#        X_train, X_test = x[train_index], x[test_index]
#        y_train, y_test = y[train_index], y[test_index]
#    [training.fit(x[train], y[train]).score(x[test], y[test])
#    for train, test in my_cv]
    training.fit(in_list,output_list)
#    grid.fit(x, y)
    
#    pickle.dump(training, open("saved_model_svc.sav", 'wb'))
#    joblib.dump(training,"saved_model_svc.sav")
    print("done training the model, now testing it.")
    out_test = training.predict(testing_input)
#    grid.fit(X, y)

#    print("The best parameters are %s with a score of %0.2f"
#      % (grid.best_params_, grid.best_score_))
#    
#    for item in out_test:
#        print item


#    print len(out_test)
    return out_test

def train_set_linear_regression():
    global testing_input
    global in_list
    global output_list
    global out_test
    
    model = LogisticRegression()
    model.fit(in_list,output_list)
    out_test = model.predict(testing_input)
    return out_test

def train_knn():
    global testing_input
    global in_list
    global output_list
    global out_test
    neigh = KNeighborsClassifier(n_neighbors=4)
    neigh.fit(in_list, output_list)
    out_test = neigh.predict(testing_input)
    return out_test
    
    
    
  
#Function to compare actual/predicted data and to calculate accuracy rate    
def compare():
    global testing_output
#    df = load_format_csv()
#    data = list(df.values)
#    print "Starting date of prediction : " + data[2059][0]
    out_test = train_set()
#    out_test = train_set_linear_regression()
#    out_test = train_knn()
    print len(out_test)
    print len(testing_output)
    correct_predictions = 0
    i = 0
    while(i<len(testing_output)):
        if(out_test[i]==testing_output[i]):
            print "correct prediction ! Predicted = "+ out_test[i]
            correct_predictions += 1
            i +=1
        else:
            print "bad prediction :( Predicted = "+ out_test[i]+ " Actual : "+testing_output[i]
            i +=1
    accuracy = ((correct_predictions/len(testing_output))*100)
    accuracy = round(accuracy,2)
    print str(correct_predictions) + " days correct predictions out of 289 days"
    print "Accuracy : " + str(accuracy)+" %"
    
        
def test_mapping():
    global in_list
    global output_list
    
    global in_list_date
    global output_list_date
    
    global testing_input_date
    global testing_output_date
    
    print "which date you want to predict ?\n"
#    date_predicted = raw_input()
#    print "date predicted : " + date_predicted
#    for item in in_list_date:
#        print item
#        
#    for item in output_list_date:
#        print item
    
#    print in_list_date[2047]
#    print output_list_date[2047]
    
    print in_list_date[150]
    print in_list[150]
    
    
    print output_list_date[150]
    print output_list[150]
    
def feature_select():
    global in_list
    global output_list
    global testing_input
    global testing_output
    
    
    x = testing_input+in_list
    y = output_list+testing_output
    
    
    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(x,y)
    numpy.set_printoptions(precision=3)
    print "using k best chi square : " # + str(fit.scores_) 
    for item in fit.scores_:
        print item
    
    
#    model = ExtraTreesClassifier()
#    model.fit(in_list, output_list)
#    print "using Feature importance : " + str(model.feature_importances_)
    
    model = LogisticRegression()
    rfe = RFE(model, 12)
    fit = rfe.fit(x, y)
    print "using RFE :"
    print("Num Features: %d") % fit.n_features_
    print("Selected Features:") #% fit.support_
    for item in fit.support_:
        print item
    print("Feature Ranking: %s") % fit.ranking_
   
    
    
    
    
    
    
#    print testing_input_date[100]
#    print testing_input[100]
#    
#    print testing_output_date[100]
#    print testing_output[100]
    
    
    
    
        
        
    
    
    
    
#    df2 = pd.DataFrame(data)
#    del df2[0]
#    data = list(df2.values)
#    print data
#    print len(data[0])
#    while (i<2054):
#        input_list.append([data[i][1],[data[i][2],[data[i][3],[data[i][4],[data[i][5])
#        i+=1
#    
#    for item in data:
#        
#        
#        if(item[0] != "2015.12.31"): #happens when i = 2053
#            i += 1
#        else:
#            print "done !"
#            print str(i)
        
        
#training : number of inputs/outputs : 2008 => fin 2015 ( 7ans) 
#testing : number of inputs/outputs ( début 2016 , fin 2016) ( 1an)
#training : 7/8 = 87.5%
#testing : 1/8 = 12.5%
#    print data
    
generate_input_output()
#train_set()
compare()




#train_set()

#feature_select()
#test_mapping()

#how to test if a model takes into account the right inputs and wants to predict the right outputs ?
#create a function which takes as input the date we wants to predict, and output the dates used to predict.




    
