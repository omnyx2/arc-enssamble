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
from matplotlib import colors
import matplotlib.pyplot as plt
from colorama import Style, Fore

class ARCPlottor:
    def __init__(self):
        self.cmap = colors.ListedColormap(
            ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
             '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
        self.norm = colors.Normalize(vmin=0, vmax=9)
        self.color_list = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]

    def check_type(self, input_data, arc_id):
        # all has include input and output
        # which has include only input 
        # or just single array for arc
        # case of arc prize data set
        if "train" in input_data:
            train_input_arcs = [ a["input"] for a in input_data["train"]]
            train_output_arcs = [ a["output"] for a in input_data["train"]]
            test_input_arcs = [ a["input"] for a in input_data["test"]]

            self.check_axis(train_input_arcs + test_input_arcs, arc_id)
            self.check_axis(train_output_arcs, arc_id)
        
        # case of single or multiple arc data 
        else:
            self.check_axis(input_data, arc_id)
    
    def check_axis(self, data, arc_id):
        if len(data) == 1 :
            self.plot_arc(data[0], arc_id)
        else:
            self.plot_arcs(data, arc_id)
    
    def plot_arc(self, x, arc_id):
        plt.imshow(np.array(x), cmap=self.cmap, norm=self.norm)
        axis[0].set_title(arc_id) 
        plt.show()
        
    def plot_arcs(self, arcs, arc_id):
        print(len(arcs))
        figure, axis = plt.subplots(1,len(arcs))
        for i in range(len(arcs)):
            axis[i].imshow(np.array(arcs[i]), cmap=self.cmap, norm=self.norm)
            axis[i].set_title(arc_id) 
        plt.show()

 