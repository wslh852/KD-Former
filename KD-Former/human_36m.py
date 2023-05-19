import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import torch
import copy
import dynamic_cal
import torch.nn.functional as F
import numpy as np
def readCSVasFloat(filename):
  lines = open(filename).readlines()
  # returnArray = [map(float, line.strip().split(',')) for line in lines] # python2.7
  returnArray = [list(map(float, line.strip().split(','))) for line in lines]
  returnArray = np.array(returnArray)
  return returnArray
def readCSVasFloatF(filename):
  lines = open(filename).readlines()
  # returnArray = [map(float, line.strip().split(',')) for line in lines] # python2.7
  returnArray = [list(map(float, line.strip().split(' '))) for line in lines]
  returnArray = np.array(returnArray)
  return returnArray
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y  # Filter requirements.
def filter_6hz(action_sequence,order=6,fs=50,cutoff=6):
    filter_out=np.zeros((action_sequence.shape[0],action_sequence.shape[1]))
    for i in range(action_sequence.shape[1]):
       filter_out[:,i]=butter_lowpass_filter(action_sequence[:,i], cutoff, fs, order)
    return filter_out
def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise
class human_36m_dataset(object):
    def __init__(self, data_dir):
        self._actions = ["walking"]#, "eating", "smoking", "discussion",  "directions",
                     # "greeting", "phoning", "posing", "purchases", "sitting",
                    #  "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether"]
        self._data_dir = data_dir
        self._action2index = self.get_dic_action2index()

    def get_dic_action2index(self):
        return dict(zip(self._actions, range(0, len(self._actions)) ))

    def get_train_subject_ids(self):
        return [1]#, 6, 7, 8, 9, 11]

    def get_test_subject_ids(self):
        return [5]

    def load_data(self, subjects):
        completeData = []
        data_set = {}
        for subj in subjects:
            for action in self._actions:
                for subact in [1, 2]:  # subactions
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self._data_dir, subj, action, subact)
                    filename2 = '{0}/S{1}/{2}_{3}_pos.txt'.format(self._data_dir, subj, action, subact)
                    action_sequence = readCSVasFloat(filename)
                    # action_sequence=filter_6hz(action_sequence)
                    action_sequence2 = readCSVasFloatF(filename2)
                    Force=dynamic_cal.Force(action_sequence2,1)
                    action_sequence=np.concatenate((action_sequence,Force),axis=1)
                    
                    # take 1/2 of that, for a final rate of 25fps
                    # https://github.com/una-dinosauria/human-motion-prediction/issues/8
                    even_list = range(0, action_sequence.shape[0], 2)
                    data_set[(subj, action, subact, 'even')] = action_sequence[even_list]
                    even_list = range(1, action_sequence.shape[0], 2)
                    data_set[(subj, action, subact, 'odd')] = action_sequence[even_list]
                    if len(completeData) == 0:
                       completeData = copy.deepcopy(action_sequence)
                    else:
                       completeData = np.append(completeData, action_sequence, axis=0)
        return data_set, completeData
    
    def normalize_data(self,data,completeData ):
        data_max= np.max(completeData, axis=0)
        data_min = np.min(completeData, axis=0)
       
        data_out = {}
        for key in data.keys():
            nor_force = np.divide((data[key][:,99:] - data_min[99:]), (data_max[99:]-data_min[99:]+0.00001))
            data_ik= data[key][:,:99]
          
            data_out[key]=np.concatenate((data_ik,nor_force),axis=1)
            
        return data_out,data_max,data_min
    def get_test_actions(self):
        return zip(self._actions, range(0, len(self._actions)))

if __name__ == '__main__':
    data_dir = '/home/hust/data/Human_3.6M/h3.6m/dataset'
    db = human_36m_dataset(data_dir)

    db.load_data(db.get_test_subject_ids() )