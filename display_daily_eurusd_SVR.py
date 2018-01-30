# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:44:51 2017

@author: smail
"""
from __future__ import division
import numpy
import pandas as pd
#import plotly.plotly as py
#import plotly.figure_factory as FF
from sklearn import svm
from datetime import datetime
import matplotlib.pyplot as plt 
import matplotlib.finance as fin
from numba import cuda
from sklearn.metrics import mean_squared_error


data = []
output_list_training_close = []
input_list_temp = []
in_list = []
testing_output_close = []
testing_input = []
output_prediction_close = []



#This function loads a daily EUR/USD csv file from 2008 to 2017 and returns a dataframe containing the data
def load_format_csv():
    global df
    df = pd.read_csv('eur_usd_daily_v2.csv', sep =',', header = None)
    return df

#This function plots the financial data from 2008 to 2017
def load_plot():   
    df = load_format_csv()
    fig,ax = plt.subplots()
    fin.candlestick2_ohlc(ax,df[1],df[2],df[3],df[4],width=0.6)
    plt.show()
    
#This function generates input/output for training and input/output for testing
def generate_input_output():
    global output_list_training_close
    global input_list_temp
    global in_list
    global testing_input
    global testing_output_close
    df = load_format_csv()
    
#    del df[0]
#    print df.values
    data = df.values.tolist()#ALL values in list
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
#    print df2.values[2053] ###training stops at date 2015.12.31 ( 31 décembre 2015)
    #31 12 2015
#    for item in data:
#        if(item[0] == "2015.12.31"):
#            print "found 31 12 2015"
#            print i
#            i = i+1
#        else:
#            i = i+1
        
#    print df2.values[2054]
    boundary = int(0.8*len(data))#to do a 80/20 split
    del df2['date']
    
    df_input = pd.DataFrame(data[0:boundary],columns=['date','open','high','low','close','volume'])
    del df_input['date']
    del df_input['volume']
    input_list_temp = df_input.values.tolist()
    
    df_input_test = pd.DataFrame(data[-(len(data)-boundary):],columns=['date','open','high','low','close','volume'])
    del df_input_test['date']
    del df_input_test['volume']
    print len(df_input_test.values)
    input_test_temp = df_input_test.values.tolist()
    
    k=0
    while(k<(len(data)-boundary)-6):
        testing_input.append(input_test_temp[k]+input_test_temp[k+1]+input_test_temp[k+2]+input_test_temp[k+3]+input_test_temp[k+4]+input_test_temp[k+5])
        k += 1
    print len(testing_input)
    
    
#    df_test = pd.pivot_table(df_input)
    
#    print input_list_temp
#    print len(input_list_temp)
#    test = []
    j = 0
    while(j<boundary-6):#342 normally
        in_list.append(input_list_temp[j]+input_list_temp[j+1]+input_list_temp[j+2]+input_list_temp[j+3]+input_list_temp[j+4]+input_list_temp[j+5])
        j += 1
#    print len(in_list)

    
     
    
#    print test
#    print len(df2.values)
#    print in_list
    i = 0
    while (i<boundary):
#        if (df2.loc[i][3] - df2.loc[i][0]) < 0:
         output_list_training_close.append(df2.loc[i][3])
         i += 1 
#        else:
#            output_list.append('1')
#            i += 1
    print i
    while(i<len(data)):
#        if (df2.loc[i][3] - df2.loc[i][0]) < 0:
           testing_output_close.append(df2.loc[i][3])
#           print(df2.loc[i][3])
           i += 1 
#        else:
#            testing_output.append('1')
#            i += 1
    
    output_list_training_close = output_list_training_close[6:]
    testing_output_close = testing_output_close[6:]
#    testing_output = output_list[-295:]
#    print output_list
#    print testing_output_close
#    print len(output_list)
#    print input_list_temp
    
#    for item in input_list_temp:
#        print item
#    print len(input_list_temp)
    print "done generating input/output lists"

#This function trains the input/output training list and returns an output prediction list          
def train_set():
    global in_list
    global output_list_training_close
    global out_test
    training = svm.SVR(kernel='rbf', degree=3, gamma=0.2, coef0=0.0, tol=0.0001, C=50, epsilon=0, 
                       shrinking=True, cache_size=600, verbose=False, max_iter=-1)
#    .SVR(kernel='rbf', degree=3, gamma=0.2, coef0=0.0, tol=0.0001, C=100, epsilon=0.0, 
#                       shrinking=True, cache_size=200, verbose=False, max_iter=-1)
#    print training
#    bla = cuda.grid(2)
#    array[x1,y1] = in_list
         
#    x2 = cuda.grid(1)
#    array2[x2] = output_list_training_close
    
#    x3 = cuda.grid(1)
    training.fit(in_list,output_list_training_close)
    out_test = training.predict(testing_input)
#    for item in out_test:
#        print item
    return out_test

def compare():
    global testing_output
    df = load_format_csv()
    data = list(df.values)
#    print "Starting date of prediction : " + data[2059][0]
    out_test = train_set()
    print len(out_test)
    print len(testing_output_close)
    correct_predictions = 0
    i = 0
    j=2060
    while(i<len(testing_output)):
        if(round(out_test[i],5)==testing_output_close[i]):
            print "correct prediction ! Predicted = "+ str(round(out_test[i],5))
            correct_predictions += 1
            i +=1
        else:
            print "bad prediction :( Predicted = "+ str(round(out_test[i],5))+ " Actual : "+ str(testing_output_close[i])
            i +=1
    accuracy = ((correct_predictions/290)*100)
    accuracy = round(accuracy,2)
#    print str(correct_predictions) + " days correct predictions out of 290 days"
#    print "accuracy : " + str(accuracy)+" %"
    mse = mean_squared_error(testing_output_close,out_test)
    print "MSE = " + str(mse)
    plt.plot(testing_output_close,'r', label = 'actual')
    plt.plot(out_test,'b',label = 'predicted')
    plt.show()








generate_input_output()
#train_set()
compare()
#    
