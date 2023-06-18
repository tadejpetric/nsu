import torch
import numpy as np
import pandas as pd
import random
import torch.nn.functional as F
from torch import nn

from torch_geometric.data import Data

def data_dict(data):
    return {k:v for k,v in enumerate(data.columns.values)}

def name_to_index()