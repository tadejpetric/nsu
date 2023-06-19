import torch
import numpy as np
import pandas as pd
import random
import torch.nn.functional as F
from torch import nn
import utility

from torch_geometric.data import Data

def data_dict(data):
    return {v:k for k,v in enumerate(data.columns.values)}

def txt_to_coo(filename, data):
    file = pd.read_csv(filename, delim_whitespace=True, header=None)
    ddict = data_dict(data)
    
    file = file.applymap(lambda x: ddict[x]).to_numpy()

    return torch.Tensor(file).long()


def make_symmetric(tensor):
    # makes the graph undirected
    rev_tensor = torch.stack((tensor[1], tensor[0]))
    return torch.cat((tensor, rev_tensor), axis=1).transpose(1,0)

def find_neighbours(node, edge_index):
    # finds all neighbours of a node
    neighbours = edge_index[edge_index[:, 0] == node]
    return neighbours[:, 1]


# uporabi samo sosede

def sample_k(neighbours, k):
    # sample k neighbours with replacement
    sample = torch.empty(k)
    for i in range(k):
        sample[i] = neighbours[random.randint(0,len(neighbours)-1)]
    return sample


def generate_problem_neigh(X, neighbours, city, j, k, n, maxwidth = 193):
    # last index available 192. Reserved for data, so i+n = 191
    indices = torch.empty(k+1, dtype=torch.long)
    indices[0] = city
    indices[1:] = sample_k(neighbours, k)
    data1 = X[j:j+n, indices].view((1+k)*n)


    ysr = X[j+n+1, city]
    # data1 is (k+1) x (n) tensor 
    return (data1, ysr)

def create_dataset_neigh_train(X, neighbours, k, n, test_indices, maxwidth=193):
    # n history, k neighours
    cities_count = int(X.shape[1])
    datapoints = int(X.shape[0])
    arr = []
    for i in range(cities_count):
        if i in test_indices:
            continue
        for j in range(datapoints-n-1):
            city_neigh = find_neighbours(i, neighbours)
            pair = generate_problem_neigh(X, city_neigh, i, j, k, n)
            arr.append(pair)
    random.shuffle(arr)
    return arr


def predict_n_neigh(X, neighbours, city, j, model, k, n, predictions, maxwidth = 193):
    #compute predictions many future events
    pred = torch.empty(predictions+n)
    pred[:n] = X[j:j+n, city]

    for i in range(predictions):
        city_neigh = find_neighbours(i, neighbours)
        data = generate_problem_neigh(X, city_neigh, city, j+i, k, n)[0]
        for history in range(n):
            data[history*(k+1)] = pred[i+history]
        with torch.no_grad():
            pred[n+i] = model(data)

    return pred.detach()

def compute_errors_neigh(model, X, neighbours, k, n , predictions, test_indices):
    errors = torch.zeros(n+predictions)
    for city in test_indices:
        for date in range(X.shape[0]-n-predictions-1):
            estimate = predict_n_neigh(X, neighbours, city, date, model, k, n, predictions)
            real = X[date:date+n+predictions, city]
            errors += abs(estimate-real)
    errors /= X.shape[0]-n-1-predictions
    errors /= len(test_indices)
    return errors
# ---

# uporabi vse
def sample_k_montedfs(node, edge_index, k):
    # sample k neighbours with replacement
    sample = torch.empty(k)
    dfsqueue = list(map(lambda x: int(x), find_neighbours(node, edge_index)))
    for i in range(k):
        sample[i] = random.choice(dfsqueue)
        dfsqueue += list(map(lambda x: int(x), find_neighbours(sample[i], edge_index)))
    return sample

def generate_problem_neigh_montedfs(X, neighbours, city, j, k, n, maxwidth = 193):
    # last index available 192. Reserved for data, so i+n = 191
    indices = torch.empty(k+1, dtype=torch.long)
    indices[0] = city
    indices[1:] = sample_k_montedfs(city, neighbours, k)
    data1 = X[j:j+n, indices].view((1+k)*n)

    ysr = X[j+n+1, city]
    # data1 is (k+1) x (n) tensor 
    return (data1, ysr)

def create_dataset_neigh_train_montedfs(X, neighbours, k, n, test_indices, maxwidth=193):
    # n history, k neighours
    cities_count = int(X.shape[1])
    datapoints = int(X.shape[0])
    arr = []
    for i in range(cities_count):
        if i in test_indices:
            continue
        for j in range(datapoints-n-1):
            pair = generate_problem_neigh_montedfs(X, neighbours, i, j, k, n)
            arr.append(pair)
    random.shuffle(arr)
    return arr

def predict_n_neigh_montedfs(X, neighbours, city, j, model, k, n, predictions, maxwidth = 193):
    #compute predictions many future events
    pred = torch.empty(predictions+n)
    pred[:n] = X[j:j+n, city]

    for i in range(predictions):
        data = generate_problem_neigh_montedfs(X, neighbours, city, j+i, k, n)[0]
        for history in range(n):
            data[history*(k+1)] = pred[i+history]
        with torch.no_grad():
            pred[n+i] = model(data)

    return pred.detach()

def compute_errors_neigh_montedfs(model, X, neighbours, k, n , predictions, test_indices):
    errors = torch.zeros(n+predictions)
    for city in test_indices:
        for date in range(X.shape[0]-n-predictions-1):
            estimate = predict_n_neigh_montedfs(X, neighbours, city, date, model, k, n, predictions)
            real = X[date:date+n+predictions, city]
            errors += abs(estimate-real)
    errors /= X.shape[0]-n-1-predictions
    errors /= len(test_indices)
    return errors
# ---



def test_indices(data):
    obcine = ["ljubljana", "maribor", "kranj", "koper", "celje", "novo_mesto", "velenje", "nova_gorica", "krško", "ptuj", "murska_sobota", "slovenj_gradec"]
    dl = data_dict(data)
    return [dl[x] for x in obcine]