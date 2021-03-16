# Imports
import numpy as np
import pdb
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters

from sklearn.model_selection import KFold

from data_utils import *
from model import *
from parser import *

# arguments parser
args = checkParser()
num_epochs = args.num_epochs
model_type = args.model_type
use_prep = args.use_prep
dropout = args.dropout
use_subject = args.use_subject
window = args.window

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preperation
## get dataset
dataset = getData()

if use_subject:
    dataset = getDataSubject(dataset)

## extract data from data dict
X_test = dataset["X_test"]
y_test = dataset["y_test"]
X_train_valid = dataset["X_train_valid"]
y_train_valid = dataset["y_train_valid"]

## data subsampling
if use_prep:
    X_train_valid_prep, y_train_valid_prep = data_prep(X_train_valid, y_train_valid, 2, 2, True)
    X_test_prep, y_test_prep = data_prep(X_test, y_test, 2, 2, True)
else:
    X_train_valid_prep = X_train_valid
    y_train_valid_prep = y_train_valid
    X_test_prep = X_test
    y_test_prep = y_test

if window:
    X_train_valid_prep = X_train_valid_prep[:, :, :window]
    X_test_prep = X_test_prep[:, :, :window]

print("X_train_valid.shape: {}".format(X_train_valid_prep.shape))
print("X_test.shape: {}".format(X_test_prep.shape))
print("y_train_valid.shape: {}".format(y_train_valid_prep.shape))
print("y_test.shape: {}".format(y_test_prep.shape))

## generate data loader
trainloader = getDataLoader(X_train_valid_prep, y_train_valid_prep)
testloader = getDataLoader(X_test_prep, y_test_prep)
checktrainloader = getDataLoader(X_train_valid, y_train_valid)
checktestloader = getDataLoader(X_test, y_test)

# dataset and network configuration
input_size = X_train_valid.shape[1]
num_classes = 4
learning_rate = 5e-4
hidden_size = 64
num_layers = 2

# Initialize network
if model_type in ["lstm", "LSTM"]:
    model = RNN_LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, dropout=dropout).to(device)
elif model_type in ["gru", "GRU"]:
    model = RNN_GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, dropout=dropout).to(device)
elif model_type in ["rnn", "RNN"]:
    model = RNN_RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, dropout=dropout).to(device)
elif model_type in ["cnn_lstm", "CNN_LSTM"]:
    model = RNN_CNN_LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, dropout=dropout).to(device)
elif model_type in ["cnn_gru", "CNN_GRU"]:
    model = RNN_CNN_GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, dropout=dropout).to(device)
else:
    model = RNN_RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, dropout=dropout).to(device)
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)


# K fold cross validation. Fold number == 5
kf = KFold(n_splits=5)
for train_index, val_index in kf.split(X_train_valid_prep):
    X_train, X_val = X_train_valid_prep[train_index], X_train_valid_prep[val_index]
    y_train, y_val = y_train_valid_prep[train_index], y_train_valid_prep[val_index]

    trainloader = getDataLoader(X_train, y_train)
    valloader = getDataLoader(X_val, y_val)

    run_model(num_epochs, trainloader, valloader, testloader, model, criterion, optimizer, device)

    break

# K fold cross validation. Fold number == 5
#kf = KFold(n_splits=5)
#fold = 0
#for train_index, test_index in kf.split(X_train_valid):
#    print("####################### Fold {} ##########################".format(fold))
#    fold += 1
#
#    # get k fold dataset
#    X_train_fold, X_test_fold = X_train_valid[train_index], X_train_valid[test_index]
#    y_train_fold, y_test_fold = y_train_valid[train_index], y_train_valid[test_index]
#
#    # get dataset loader
#    trainloader = getDataLoader(X_train_fold, y_train_fold)
#    testloader = getDataLoader(X_test_fold, y_test_fold)
#
#    # train model on train dataset
#    run_model(num_epochs, trainloader, model, criterion, optimizer, device)
#
#    # check model on test dataset
#    check_accuracy(trainloader, model, device)
#    check_accuracy(testloader, model, device)
#
#    print("\n")


# test model on test dataset
print("\n")
check_accuracy(trainloader, model, device)
check_accuracy(testloader, model, device)
