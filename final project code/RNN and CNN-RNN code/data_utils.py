import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EEG_Dataset(Dataset):
    def __init__(self, din_x=None, din_y=None):
        self.X = din_x
        self.y = din_y-769
    
    def __len__(self):
        return (self.X.shape[0])

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx, :, :]).float()
        y = torch.tensor(self.y[idx]).long()
        sample = {"X": X, "y": y}

        return sample

def getData():
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
    person_train_valid = np.load("person_train_valid.npy")
    X_train_valid = np.load("X_train_valid.npy")
    y_train_valid = np.load("y_train_valid.npy")
    person_test = np.load("person_test.npy")

    print ('Training/Valid data shape: {}'.format(X_train_valid.shape))
    print ('Test data shape: {}'.format(X_test.shape))
    print ('Training/Valid target shape: {}'.format(y_train_valid.shape))
    print ('Test target shape: {}'.format(y_test.shape))
    print ('Person train/valid shape: {}'.format(person_train_valid.shape))
    print ('Person test shape: {}'.format(person_test.shape))

    return {
        "X_train_valid": X_train_valid,
        "y_train_valid": y_train_valid,
        "X_test": X_test,
        "y_test": y_test,
        "person_train_valid": person_train_valid,
        "person_test": person_test
            }

def getDataSubject(dataset_dict):

    X_test = dataset_dict["X_test"]
    y_test = dataset_dict["y_test"]
    person_train_valid = dataset_dict["person_train_valid"]
    X_train_valid = dataset_dict["X_train_valid"]
    y_train_valid = dataset_dict["y_train_valid"]
    person_test = dataset_dict["person_test"]

    num_person = len(set(person_train_valid.squeeze(1).tolist()))

    #for person in range(num_person):
    person = 0
    person_train_valid = person_train_valid.squeeze(1)
    person_test = person_test.squeeze(1)
    X_train_valid_person = X_train_valid[person_train_valid==person]
    y_train_valid_person = y_train_valid[person_train_valid==person]
    X_test_person = X_test[person_test==person]
    y_test_person = y_test[person_test==person]

    return {
        "X_train_valid": X_train_valid_person,
        "y_train_valid": y_train_valid_person,
        "X_test": X_test_person,
        "y_test": y_test_person
            }

def getDataLoader(din_x, din_y):
    check_dataset = EEG_Dataset(din_x, din_y)
    dataloader = DataLoader(check_dataset, batch_size=128, shuffle=True)
    
    return dataloader

def data_prep(X,y,sub_sample,average,noise):
    
    total_X = None
    total_y = None
    
    # Trimming the data
    X = X[:,:,0:500]
    
    # Maxpooling the data
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    
    total_X = X_max
    total_y = y
    # Averaging + noise
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    
    # Subsampling
    
    for i in range(sub_sample):
        
        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
            
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
        
        
    return total_X,total_y

if __name__=="__main__":
    res = getData()
