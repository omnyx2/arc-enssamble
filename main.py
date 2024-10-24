import os, gc
import sys, pdb
import copy, time
import json, random

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from pathlib import Path

import matplotlib
import os

import torch

from matplotlib import colors
import matplotlib.pyplot as plt
from colorama import Style, Fore
# %matplotlib inline ipnyb

from dataloader.my_loader import MyDataLoader
from dataloader.OAutoEncoder_loader import ARCDataset

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from visualization.my_vis import ARCPlottor
import torch.optim as optim

from models.O_Auto_Encoder import *

# 기본 셋팅
cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=10)

# 주피터 환경에서 데이터 호출하기
for dirname, _, filenames in os.walk('/home/hyunseok/enssamble/settings'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
local_path = "/home/hyunseok/enssamble/data/kaggle/"
with open("/home/hyunseok/enssamble/settings/kaggle_data_file_name.json",'r') as file:
    path_dict = json.load(file)
    data = MyDataLoader("arcprize", path_dict, local_path) 
    data.cur_data_mode("train")

plotter = ARCPlottor()
# plotter.check_type(data.cur_problem["890034e9"],"890034e9")



# 경로 셋팅
BASE_FOLDER = '/home/hyunseok/enssamble/data/kaggle/'
file_list = [
    f'{BASE_FOLDER}/arc-agi_training_challenges.json',
    f'{BASE_FOLDER}/arc-agi_evaluation_challenges.json',
    f'{BASE_FOLDER}/arc-agi_test_challenges.json'
]
# 학습 데이터셋 로드 - 이상한게 모델이랑 다이렉트로 연결됨,
train_dataset = ARCDataset(file_list)
train_dataset.set_dataset()
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# 디멘션 설정 - 데이터셋별로 디멘션 지정을 다르게 해줘야함 ... 
Dimension = train_dataset.dim
Keys = train_dataset.keys

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 디멘션 설정  - 훈련 종류 별로 디멘션 지정을 다르게 해줘야함 ... 
IN_DIM = 100 + 1
print(IN_DIM)
OUT_DIM = 100
LATENT_DIM = 1024
model = LSTM(IN_DIM, OUT_DIM, LATENT_DIM).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
training = Training(model, train_loader, criterion, optimizer, device)
training.train()

# 모델 불러오기
model.load_state_dict(torch.load('model.pth'))

sanity = Prediction(model, train_dataset, Dimension, cmap, norm, device)
sanity.sanity(1)
result = sanity.predict(show_step=10)

print(result)
# test_dataset = ARCDataset(f'{BASE_FOLDER}/arc-agi_test_challenges.json', 'test', dim=Dimension, keys=Keys)
# test_dataset.set_dataset()
# pred = Prediction(model, test_dataset, Dimension, cmap, norm, device)
# pred.pred_all(show_step=20)
