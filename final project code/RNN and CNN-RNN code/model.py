import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions

# Recurrent neural network with RNN (many-to-one)
class RNN_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(RNN_RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        out, _ = self.rnn(x)
        out_last = out[-1, :, :]
        res = self.fc(out_last)
        return res

# Recurrent neural network with LSTM (many-to-one)
class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(RNN_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        #self.fc = nn.Linear(hidden_size, num_classes)
        self.fc=nn.Sequential(nn.Linear(hidden_size,54),nn.BatchNorm1d(num_features=54,eps=1e-05,momentum=0.2,affine=True),nn.ReLU(inplace=True),nn.Dropout(p=0.5),nn.Linear(54,44),nn.BatchNorm1d(num_features=44,eps=1e-05,momentum=0.2,affine=True),nn.ReLU(inplace=True),nn.Linear(44,4))


    def forward(self, x):
        x = x.permute(2, 0, 1)
        #x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out_last = out[-1, :, :]
        res = self.fc(out_last)
        return res

# Recurrent neural network with GRU (many-to-one)
class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(RNN_GRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        out, _ = self.gru(x)
        out_last = out[-1, :, :]
        res = self.fc(out_last)
        return res

# Recurrent neural network with LSTM (many-to-one)
class RNN_CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, cnn_output_size=25):
        super(RNN_CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_output_size, kernel_size=10, stride=2),
            nn.BatchNorm1d(cnn_output_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(2, 0, 1)
        lstm_out, _ = self.lstm(cnn_out)
        out_last = lstm_out[-1, :, :]
        res = self.fc(out_last)
        return res

# Recurrent neural network with LSTM (many-to-one)
class RNN_CNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, cnn_output_size=25):
        super(RNN_CNN_GRU, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_output_size, kernel_size=10, stride=2),
            nn.BatchNorm1d(cnn_output_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=cnn_output_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(2, 0, 1)
        gru_out, _ = self.gru(cnn_out)
        out_last = gru_out[-1, :, :]
        res = self.fc(out_last)
        return res

def run_model(num_epochs, trainloader, valloader, testloader, model, criterion, optimizer, device):
    model.train()
    for epoch in range(num_epochs):
        print("#######################epoch: {}##########################".format(epoch))

        train_correct = 0
        train_targets = 0
        val_correct = 0
        val_targets = 0

        for idx, batch in enumerate(trainloader):
            data = batch["X"].to(device=device)
            targets = batch["y"].to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            pred_train = torch.argmax(scores, dim=1)
            train_correct += float(torch.sum(pred_train==targets))
            train_targets += targets.shape[0]
        print("correct prediction: {}, total targets: {}, accuracy: {}".format(train_correct, train_targets, train_correct/train_targets))

        check_accuracy(valloader, model, device)
        check_accuracy(testloader, model, device)

        #for idx, batch in enumerate(valloader):
        #    data = batch["X"].to(device=device)
        #    targets = batch["y"].to(device=device)

        #    with torch.set_grad_enabled(False):
        #        scores = model(data)
        #    pred_val = torch.argmax(scores, dim=1)
        #    val_correct += float(torch.sum(pred_val==targets))
        #    val_targets += targets.shape[0]
        #print("correct prediction: {}, total targets: {}, accuracy: {}".format(val_correct, val_targets, val_correct/val_targets))
            


# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model, device):
    num_correct = 0
    num_targets = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            data = batch["X"].to(device=device)
            targets = batch["y"].to(device=device)

            scores = model(data)
            predictions = torch.argmax(scores, dim=1)
            num_correct += float(torch.sum(predictions == targets))
            num_targets += targets.shape[0]

        print("correct prediction: {}, total targets: {}, accuracy: {}".format(num_correct, num_targets, num_correct/num_targets))
    # Set model back to train
    model.train()
