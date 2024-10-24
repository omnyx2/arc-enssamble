import json
import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset

class ARCDataset(Dataset):
    def __init__(self, file_path, mode='train', dim={}, keys={}):
        self.mode = mode
        self.dim = {}
        self.keys = {}
        self.data = self.extract_file(file_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        task = self.data[idx]
        return task[1], task[2]
    
    def extract_file(self, file_path):
        data = []
        if isinstance(file_path, list):
            for path in file_path:
                d = self.load_data(path)
                data = [*data, *d]
        else:
            data = self.load_data(file_path)
        return data
    
    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return self.parse_data(data)
    
    def parse_data(self, data):
        ndata = []
        for key in data.keys():
            task = data[key][self.mode]
            dim_inp = []
            dim_out = []
            for i in range(len(task)):
#                 index = list(self.dim).index(key) if key in self.dim else len(self.dim)
                input_tensor, i_shape = self.expand_data(task[i]['input'])
                output_tensor, o_shape = self.expand_data(task[i]['output']) if self.mode == 'train' else [[], [0,0]]
                input_tensor = torch.tensor(input_tensor)
                output_tensor = torch.tensor(output_tensor)
                ndata.append([key, input_tensor, output_tensor])
                if self.mode == 'train':
                    dim_inp.append(tuple(i_shape))
                    dim_out.append(tuple(o_shape))
            if self.mode == 'train':
                counter_inp = Counter(dim_inp)
                counter_out = Counter(dim_out)
                self.dim[key] = {
                    "inp_dim": list(counter_inp.most_common(1)[0][0]),
                    "out_dim": list(counter_out.most_common(1)[0][0]),
                }
        return ndata

    def expand_data(self, data):
        # convert from matrix to 1D array
        data = np.array(data)
        shape = data.shape
        data = data.flatten()
        ndata = []
        while len(ndata) < 100:
            ndata = [*ndata, *data]
        data = np.array(ndata[:100]) / 10
        return data, shape
    
    def set_dataset(self):
        keys = [[d[0]] for d in self.data]
        ndata = []
        for i in range(len(self.data)):
            d = self.data[i]
            if self.mode == 'train':
                if d[0] not in self.keys:
                    self.keys[d[0]] = len(self.keys)
                encoded = self.keys[d[0]]
            else:
                encoded = self.keys[d[0]] if d[0] in self.keys else -1
            ndata.append([
                keys[i][0],
                torch.tensor([torch.tensor(encoded), *d[1]]),
#                 torch.tensor(self.data[i][1]),
                d[2]
            ])
        self.data = ndata