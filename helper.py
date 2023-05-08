'''Based on helper.py from Assignment 1 by Thomas Moerland.
Added some own functions.'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pathlib import Path

def tf_control_memorygrowth():
    '''From https://www.tensorflow.org/guide/gpu 
    (sometimes needed to prevent OOM on dslab servers)'''

    tf.keras.backend.clear_session()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
          logical_gpus = tf.config.list_logical_devices('GPU')
          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
          # Memory growth must be set before GPUs have been initialized
          print(e)

def make_results_dir(name, parent='results'):
    Path(parent + '/' + name).mkdir(parents=True, exist_ok=True)
    return parent + '/' + name

class LearningCurvePlot:

    def __init__(self,title=None, xlabel='Episode', ylabel='Reward'):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)      
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self,y,var=None,label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(y,label=label)
        else:
            self.ax.plot(y)
        if var is not None:
            self.ax.fill_between(range(len(y)),y+var,y-var, alpha=0.1)
    
    def set_ylim(self,lower,upper):
        self.ax.set_ylim([lower,upper])

    def add_hline(self,height,label):
        self.ax.axhline(height,ls='--',c='k',label=label)

    def save(self,name='test.png', results_dir=''):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(results_dir + '/'*min(len(results_dir),1) + name, dpi=300)

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)