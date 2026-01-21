import pandas as pd
import numpy as np
import torch
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import StandardScaler
import random

class MyDataset(Dataset):

    def __init__(self, path, data_name, mask=0):
        self.data = np.loadtxt(path, delimiter=',', dtype=np.float32)
        self.x_dim = self.data.shape[-1] - 6
        self.x_dim_start = int(mask*self.x_dim)
        self.x_dim -= self.x_dim_start
        self.sample_num = self.data.shape[0]

    def __getitem__(self, index):

        return self.data[index, self.x_dim_start:]

    def __len__(self):

        return len(self.data)

    def get_sampler(self, treat_weight=1):

        t = self.data[:, -3].astype(np.int16)
        count = Counter(t)
        class_count = np.array([count[0], count[1]*treat_weight])
        weight = 1. / class_count
        samples_weight = torch.tensor([weight[item] for item in t])
        sampler = WeightedRandomSampler(
            samples_weight,
            len(samples_weight),
            replacement=True)

        return sampler

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s



def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def acic2016_processor():

    # test_delta = 3.5
    test_delta = 2 
    hwb_delta = [2.5, 3, 3.5, 4, 4.5]
    data = pd.read_csv('Datasets/ACIC/dataset.csv')
    
    data = data.to_numpy()
    data_length = np.size(data, 0)
    

    samples = np.random.normal(loc=0, scale=1, size=(data_length, 5))
    
    data_left = data[:, :np.size(data, 1) - 5]
    data_right = data[:, np.size(data, 1) - 5:]
    data = np.concatenate((data_left, samples, data_right), axis=1)

    
    exist = [0 for i in range(data_length)]
    test_data = []
    max_test = 0.1 * data_length
    num = 0
    
    for i in range(data_length):
        prob = 1
        if exist[i] == 1:
            continue
        for j in range(np.size(data, 1) - 10, np.size(data, 1) - 10 + 5):
            y_obs = data[i][-4]
            prob *= np.power(np.abs(test_delta), -0.01 * np.abs(y_obs - sign(test_delta) * data[i][j]))
        if random.random() <= prob:
            test_data.append(np.append(data[i], 0))
            num += 1
            exist[i] = 1
            if num >= max_test:
                break
    
    train_data = []
    for d in range(0, len(hwb_delta)):
        max_d = 0.9 * 0.2 * data_length
        num = 0
        for i in range(data_length):
            prob = 1
            if exist[i] == 1:
                continue
            for j in range(np.size(data, 1) - 10, np.size(data, 1) - 10 + 5):
                prob *= np.power(np.abs(hwb_delta[d]), -0.01 * np.abs(y_obs - sign(test_delta) * data[i][j]))

            if random.random() <= prob:
                train_data.append(np.append(data[i], d + 1))
                num += 1
                exist[i] = 1
                if num >= max_d:
                    break
    
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    
    
    eval_num = int(0.3 * np.size(train_data, 0))
    eval_index = np.random.choice(train_data.shape[0], eval_num, replace=False)
    eval_data = train_data[eval_index]
    train_data = np.delete(train_data, eval_index, axis=0)

    np.savetxt("Datasets/ACIC/train_" + str(test_delta) + ".csv", train_data, delimiter=',')
    np.savetxt("Datasets/ACIC/traineval_" + str(test_delta) + ".csv", train_data, delimiter=',')
    np.savetxt("Datasets/ACIC/test_" + str(test_delta) + ".csv", test_data, delimiter=',')
    np.savetxt("Datasets/ACIC/eval_" + str(test_delta) + ".csv", eval_data, delimiter=',')

    print('New ACIC2016 Data')
    return None


def ihdp_processor():
    test_delta = 2
    hwb_delta = [2.5, 3, 3.5, 4, 4.5]
    data = pd.read_csv('Datasets/IHDP/dataset.csv')
    
    data = data.to_numpy()
    data_length = np.size(data, 0)
    
    samples = np.random.normal(loc=0, scale=1, size=(data_length, 5))
    
    data_left = data[:, :np.size(data, 1) - 5]
    data_right = data[:, np.size(data, 1) - 5:]
    data = np.concatenate((data_left, samples, data_right), axis=1)
    exist = [0 for i in range(data_length)]
    test_data = []
    max_test = 0.1 * data_length
    num = 0
    
    for i in range(data_length):
        prob = 1
        if exist[i] == 1:
            continue
        for j in range(np.size(data, 1) - 5, np.size(data, 1) - 5 + 5):
            y_obs = data[i][-4]
            prob *= np.power(np.abs(test_delta), -0.01 * np.abs(y_obs - sign(test_delta) * data[i][j]))
            
        if random.random() <= prob:
            test_data.append(np.append(data[i], 0))
            num += 1
            exist[i] = 1
            if num >= max_test:
                break
    
    train_data = []
    for d in range(0, len(hwb_delta)):
        max_d = 0.9 * 0.2 * np.size(data, 0)
        num = 0
        for i in range(np.size(data, 0)):
            prob = 1
            if exist[i] == 1:
                continue
            for j in range(np.size(data, 1) - 5, np.size(data, 1) - 5 + 5):
                prob *= np.power(np.abs(hwb_delta[d]), -0.01 * np.abs(y_obs - sign(test_delta) * data[i][j]))
                
            if random.random() <= prob:
                train_data.append(np.append(data[i], d + 1))
                num += 1
                exist[i] = 1
                if num >= max_d:
                    break
    
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    
    
    eval_num = int(0.3 * np.size(train_data, 0))
    eval_index = np.random.choice(train_data.shape[0], eval_num, replace=False)
    eval_data = train_data[eval_index]
    train_data = np.delete(train_data, eval_index, axis=0)

    np.savetxt("Datasets/IHDP/train.csv", train_data, delimiter=',')
    np.savetxt("Datasets/IHDP/traineval.csv", train_data, delimiter=',')
    np.savetxt("Datasets/IHDP/test.csv", test_data, delimiter=',')
    np.savetxt("Datasets/IHDP/eval.csv", eval_data, delimiter=',')

    print(np.size(train_data, 0))
    print(np.size(eval_data, 0))
    print(np.size(test_data, 0))
    


if __name__ == "__main__":

    acic2016_processor()
    ihdp_processor()
