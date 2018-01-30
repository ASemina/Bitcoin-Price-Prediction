# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 18:18:51 2017

@author: Alexander
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import sklearn as sk

def create(file1, monitorlen, tester = 0, test = False):
    
    full_y = np.genfromtxt(file1, delimiter=',')
    monitor_y = full_y[-monitorlen:]
    #print(full_y, monitor_y)
    if test == True:
        monitor_y = full_y[tester:tester+monitorlen]
    return full_y, monitor_y

def train(data, monitor, length, X, model):
    
    #data = sk.preprocessing.normalize(data, axis=0) #normalize by dimension not instance
    #print(data[0],data[1])
    monitor = sk.preprocessing.normalize(monitor)
    rbf_models = []
    for day,price in enumerate(data):
        svr_rbf = SVR(kernel= model, C=1)
        rbf_models += [svr_rbf.fit(X, sk.preprocessing.normalize(data[day:day+length])).predict(X)]
    svr_rbf = SVR(kernel= model, C=1)
    mon_model = svr_rbf.fit(X, monitor).predict(X)
    return rbf_models, mon_model


def fit(data, monitor, length, X, tester = 0, test = False):
    
    processed = []
    if test == False:
        for day,price in enumerate(data[:-length]):
            processed += [data[day:day+length]]
    else:
        for day,price in enumerate(data[:tester]):
            processed += [data[day:day+length]]
    processed = np.array(processed)
    #print(processed)
    processed = sk.preprocessing.normalize(processed)
    monitor = sk.preprocessing.normalize(np.reshape(monitor,(1,-1)))
    return processed, monitor


def predict(models, monitor, top):
    
    best_sim = [float('inf')] * top
    best_models = [-1] * top
    
    for i,model in enumerate(models):
        simil = np.sqrt(np.sum(np.subtract(monitor, model)**2))
        for j,sim in enumerate(best_sim):
            if simil < sim:
                best_sim = np.insert(best_sim, j, simil)
                best_models = np.insert(best_models, j, i)
                best_sim = np.delete(best_sim, -1)
                best_models = np.delete(best_models, -1)
                break
    return best_models, best_sim
        

def plot(predictions, guesser, X):
    
    lw = 2
    plt.figure(figsize = (15,10))
    rainbow = plt.cm.rainbow(np.linspace(0,1,len(predictions)))
    for i,pred in enumerate(predictions):
        plt.plot(X, pred, color=rainbow[i], lw=lw, label='Simular Trend')
    plt.plot(X, guesser.T, color='cornflowerblue', lw=lw, label='Analyzed Period')
    plt.legend()
    plt.show()


def benefit(data, pred, sim, length, top):
    diff = []
    pos = 0
    for i in pred:
        dif = data[i + length] - data[i+length-1]
        diff += [dif]
        if dif > 0:
            pos += 1
    diff = np.array(diff)
    score = np.sum(diff)
    avg = score/top
    weight_score = diff @ (1 - np.array(sim))
    weight_avg = weight_score/top
    return score, avg, weight_score, weight_avg, pos
    
    
def main(file, length, top, tester = 0, test = False, model = 'rbf', printmode = True):
    
    data, monitor = create(file, length, tester, test)
    X = range(1,length+1)
    #print(monitor)
    #models,monitor_model = train(data, monitor, length, X, model)
    models,monitor_model = fit(data, monitor, length, X)
    top_predictions, top_sim = predict(models, monitor_model, top)
    if printmode == True:
        plot(models[top_predictions], monitor_model, X)
        for i in top_predictions:
            print("Day ", i,"| Monitor Period: ")
            print(data[i:i+length])
            print("Day ", i, "| Prediction Period: ")
            print(data[i+length:i+length+length], "\n")
        print("Monitoring Period: ")
        print(monitor, "\n")
    score, avg, weight_score, weight_avg, pos = benefit(data, top_predictions, top_sim, length, top)
    if printmode == True:
        print("Score: ", score)
        print("Average Change: ", avg)
        print("Weighted Score: ", weight_score)
        print("Weighted Average: ", weight_avg)
        print("Percent Positive: ", pos/top)
    else:
        if weight_score > -10: #pos > 0.5: #(.61)
            return 1
        else:
            return 0