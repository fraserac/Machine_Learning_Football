# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 13:38:52 2020

Deep learning stock prices? 
"""
import sys
import scipy 
import numpy as np
import matplotlib as mtplt
import pandas as pd
import sklearn as sk 
import seaborn as sns
import random 
import datetime
import statistics
from scipy import stats



print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(np.__version__))
print('matplotlib: {}'.format(mtplt.__version__))
print('pandas: {}'.format(pd.__version__))
print('sklearn: {}'.format(sk.__version__))




from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import torch
from torch import nn
from torch import FloatTensor as FT


print("torch: ", torch.__version__)
 
url1 = 'https://raw.githubusercontent.com/zandree/statsBrazilianLeagueChampionship/master/training_2010.csv'
url2 = 'https://raw.githubusercontent.com/zandree/statsBrazilianLeagueChampionship/master/training_2011.csv'
url3 = 'https://raw.githubusercontent.com/zandree/statsBrazilianLeagueChampionship/master/training_2012.csv'
data1 = read_csv(url1)
data2 = read_csv(url2)
data3 = read_csv(url3)
extract = [5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 35, 36, 37, 38, 47, 48, 49]
dataset1 = data1.iloc[:,extract]
training = dataset1.iloc[:-20]
test = dataset1.iloc[-20:]


#normalise columns 0-1:

for e in range(len(training.columns) - 3): #iterate for each column
    num = max(training.iloc[:, e].max(), test.iloc[:, e].max()) #check the maximum value in each column
    if num < 10:
        training.iloc[:, e] /= 10
        test.iloc[:, e] /= 10
    elif num < 100:
        training.iloc[:, e] /= 100
        test.iloc[:, e] /= 100
    elif num < 1000:
        training.iloc[:, e] /= 1000
        test.iloc[:, e] /= 1000
    else:
        print("Error in normalization! Please check!")

training = training.sample(frac=1)
test = test.sample(frac=1)
#all rows, all columns except for the last 3 columns
training_input  = training.iloc[:, :-3]#all rows, the last 3 columns
training_output = training.iloc[:, -3:]#all rows, all columns except for the last 3 columns
test_input  = test.iloc[:, :-3]#all rows, the last 3 columns
test_output = test.iloc[:, -3:]

        
def quickSummary(data, peek, c = 'none'):
    shapeOf = data.shape
    headGlimpse = data.head(peek)
    desc = data.describe()
    if c!= 'none':
        distinct = data.groupby(c).size()
    elif c == 'none':
        distinct = c
    print("shape: ", shapeOf, "\r\n", "glimpse: \r\n", headGlimpse, "\r\n", "desciption: \r\n", desc, "\r\n" ,"distinct groups: \r\n ", distinct)
    return shapeOf, headGlimpse, desc, distinct

def convert_output_win(source):   # SWITCH CASE FOR DRAW/LOSS
    target = source.copy() # make a copy from source
    target['new'] = 2 # create a new column with any value
    for i, rows in target.iterrows():
        if rows['win'] == 1:
            rows['new'] = 1
        if rows['draw'] == 1:
            rows['new'] = 0
        if rows['defeat'] == 1:
            rows['new'] = 0
    return target.iloc[:, -1]  # return all rows, the last column

training_output = convert_output_win(training_output)
test_output = convert_output_win(test_output)

class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.tanh= nn.Tanh()
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()    
        
    def forward(self, x):
        hidden = self.fc1(x)
        tanh = self.tanh(hidden)
        output = self.fc2(tanh)
        output = self.sigmoid(output)
        return output

tr_in = FT(training_input.values)
te_in = FT(test_input.values)
tr_out = FT(training_output.values)
te_out = FT(test_output.values)

in_size = tr_in.size()[1]
hidden_size= 150
model = Net(in_size, hidden_size)
eps = 1e-10
criterion= nn.BCELoss() # binary cross entropy
optimizer = torch.optim.SGD(model.parameters(), lr=0.9, momentum = 0.6)
model.eval()
y_pred = model(te_in)
before_train = criterion(y_pred.squeeze(), te_out)
print('Test loss before training' , before_train.item())


model.train()
epochs = 3000
errors = []
for epoch in range(epochs):
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(tr_in)
    # Compute Loss
    loss = criterion(y_pred.squeeze(), tr_out)
    errors.append(loss.item())    
    
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    # Backward pass
    loss.backward()
    optimizer.step()
    
model.eval()
y_pred = model(te_in)
after_train = criterion(y_pred.squeeze(), te_out)
print('Test loss after Training' , after_train.item())



def plotcharts(errors):
    errors = np.array(errors)   
    plt.figure(figsize=(12, 5))    
    graf02 = plt.subplot(1, 2, 1) # nrows, ncols, index
    graf02.set_title('Errors')
    plt.plot(errors, '-')
    plt.xlabel('Epochs')   
    graf03 = plt.subplot(1, 2, 2)
    graf03.set_title('Tests')
    a = plt.plot(te_out.numpy(), 'yo', label='Real')
    plt.setp(a, markersize=10)
    a = plt.plot(y_pred, 'b+', label='Predicted')
    plt.setp(a, markersize=10)
    plt.legend(loc=7)
    plt.show()
    
    
y_pred = y_pred.detach().numpy()    
y_pred = np.where(y_pred <0.5, 0, y_pred)
y_pred = np.where(y_pred >0.5, 1, y_pred)
        
plotcharts(errors)
#split datasets into important features analytically
# incorporate neural net and validation techniques
#aggregate data a la paper file:///C:/Users/Fraser/Downloads/1-s2.0-S0957417417302890-main.pdf