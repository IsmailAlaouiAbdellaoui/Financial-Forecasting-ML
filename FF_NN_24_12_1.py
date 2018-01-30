# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:13:19 2017

@author: smail
"""

import pyrenn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error



x_train = []
x_test = []

y_train = []
y_test = []

data = []

def load_data():
    df = pd.read_csv('EURJPY1440.csv',sep = ',', header= None)
    del df[1]
    del df[6]
    return df

def generate_input_output():
    global x_train
    global x_test
    global y_test
    global y_train
    df = load_data()
    data = df.values.tolist()
    for item in data:
        item[1] = round(item[1],3)
        item[2] = round(item[2],3)
        item[3] = round(item[3],3)
        item[4] = round(item[4],3)
    
    df2 = pd.DataFrame(data,columns=['date','open','high','low','close'])
#    print df2
#    return
    boundary = int(0.8*len(data))#to do a 80/20 split
    df_input = pd.DataFrame(data[0:boundary],columns=['date','open','high','low','close'])
    del df_input['date']
    
    input_list_temp = df_input.values.tolist()
    j = 0
    while(j<boundary-6):#342 normally
        x_train.append(input_list_temp[j]+input_list_temp[j+1]+input_list_temp[j+2]+input_list_temp[j+3]+input_list_temp[j+4]+input_list_temp[j+5])
        j += 1
        
    df_input_test = pd.DataFrame(data[-(len(data)-boundary):],columns=['date','open','high','low','close'])
    del df_input_test['date']
#    del df_input_test['volume']
#    print len(df_input_test.values)
#    return
    input_test_temp = df_input_test.values.tolist()
    
    k=0
    while(k<(len(data)-boundary)-6):
        x_test.append(input_test_temp[k]+input_test_temp[k+1]+input_test_temp[k+2]+input_test_temp[k+3]+input_test_temp[k+4]+input_test_temp[k+5])
        k += 1
#     print len(testing_input)
        
    i = 0
    while (i<boundary):
#        if (df2.loc[i][3] - df2.loc[i][0]) < 0:
         y_train.append(df2.loc[i][4])
         i += 1 
#        else:
#            output_list.append('1')
#            i += 1
#    print i
    while(i<(len(data))):
#        if (df2.loc[i][3] - df2.loc[i][0]) < 0:
           y_test.append(df2.loc[i][4])
#           print(df2.loc[i][3])
           i += 1
          
    y_train = y_train[6:]
    y_test = y_test[6:]
    print "done generating input/output lists"
    
    
def train_set():
    global x_train
    global y_train
    global x_test
    global y_test
    print(len(x_train))
    print(len(y_train))
    structure = [24,12,1]
    for item in y_train:
        item = round(item,3)
        
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    x_test = np.asarray(x_test)
    
    
    print type(x_train.ndim)
    rnn = pyrenn.CreateNN(structure)#dIntern=[1]
    rnn = pyrenn.train_LM(np.transpose(x_train),np.transpose(y_train),rnn,verbose=True,k_max=200,E_stop=1e-7)

    out_test = pyrenn.NNOut(np.transpose(x_test),rnn)
    plt.plot(y_test,'r', label = 'actual')
    plt.plot(out_test,'b',label = 'predicted')
    mse = mean_squared_error(y_test,out_test)
    print "MSE = " + str(mse)
    plt.show()
    
    
    
    
bla2 = pd.DataFrame()
generate_input_output()
train_set()

























