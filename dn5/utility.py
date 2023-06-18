import torch
import numpy as np
import pandas as pd
import random
import torch.nn.functional as F
from torch import nn

def predict_n_deep(X, city, date, model, n):
    
    predictions = torch.zeros(model.layers+n)
    predictions[0:model.layers] = X[date : date + model.layers, city]
    for i in range(n):
        with torch.no_grad():
            predictions[model.layers + i] = model(predictions[i:i+model.layers])
    return predictions.detach()

def compute_errors_deep(X, model, starts, n):
    errors = torch.zeros(n+model.layers)
    for start in starts:
        mine = predict_n_deep(X, 5, start, model, n)
        real = X[start:start+n+model.layers, 5]
        errors += abs(mine - real)
    errors /= len(starts)
    return errors

def predict_n(X, city, date, model, n):
    
    predictions = torch.zeros(model.layers+n)
    predictions[0:model.layers] = X[date : date + model.layers, city]
    for i in range(n):
        with torch.no_grad():
            predictions[model.layers + i] = model(predictions[i:i+model.layers].view(model.layers, 1))
    return predictions.detach()

def compute_errors(X, model, starts, n):
    errors = torch.zeros(n+model.layers)
    for city in range(X.shape[1]):
        for start in starts:
            mine = predict_n(X, city, start, model, n)
            real = X[start:start+n+model.layers, city]
            errors += abs(mine - real)
    errors /= len(starts)
    errors /= X.shape[1]
    return errors


def generate_problem(X, city, j, batch_size, n, m=1, maxwidth = 181):
    # last index available 192. Reserved for data, so i+n = 191
    batch_size = min(batch_size, maxwidth-city-2)
    data1 = X[j:j+n, city:city+batch_size].view(n,batch_size,m)
    ysr = X[j+n+1, city:city+batch_size].view(batch_size, m)
    #a = X[j+n+1:j+n+1+m, city:city+batch_size].split(1)
    #a = tuple(map(lambda z: torch.transpose(z, 1,0), a))
    #ysr = torch.cat(a, 1)
    return (data1, ysr)

def create_dataset(X, batch_size, n, m=1):
    cities_count = int(X.shape[1])
    datapoints = int(X.shape[0])
    arr = []
    for i in range(0, cities_count, batch_size):
        for j in range(datapoints-n-1):
            pair = generate_problem(X, i, j, batch_size, n, m)
            arr.append(pair)
    random.shuffle(arr)
    return arr

def train_recurrent(model, optimiser, criterion, dset, epochs):
    model.train()
    for epoch in range(epochs):
        print("Epoch ", epoch)
        for datax, y_real in dset:
            if torch.numel(datax) == 0:
                continue
            optimiser.zero_grad()
            y_pred = model(datax)

            loss = criterion(y_pred, y_real)
            loss.backward()
            optimiser.step()

            if random.randint(0,100) == 0:
                print(f' Loss: {loss:.4f}, est rea [{float(y_pred[0,0]):.3f} {float(y_real[0,0]):.3f}]')
    
    return model

def train_deep(model, optimiser, criterion, dset, epochs):
    model.train()
    for epoch in range(epochs):
        print("Epoch ", epoch)
        for datax, y_real in dset:
            if torch.numel(datax) == 0:
                continue
            datax = torch.transpose(datax.view(datax.shape[0],datax.shape[1]),0,1)
            optimiser.zero_grad()
            y_pred = model(datax)
            loss = criterion(y_pred, y_real)
            loss.backward()
            optimiser.step()

            if random.randint(0,100) == 0:
                print(f' Loss: {loss:.4f}, est rea [{float(y_pred[0,0]):.3f} {float(y_real[0,0]):.3f}]')
    return model