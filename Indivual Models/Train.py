import librosa
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
from torch.utils.data import TensorDataset
import numpy as np
from torch.utils.data import DataLoader
import torch
import scipy.fftpack as fft
import torch.nn as nn
import copy
import pickle
from sklearn.model_selection import train_test_split
from numba import cuda
import torch 
import torch.nn as nn 
import torch.optim as optim

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


def load_pkl(data_file):
    
    file = open(data_file, 'rb') 
    dataset = pickle.load(file)

    df_dataset = pd.DataFrame(dataset, columns=["cqcc", "label"])
    df_dataset.to_csv('dataset.csv')

    train_set, test_set = train_test_split(df_dataset, test_size=0.3, random_state=42,stratify=df_dataset['label'] )

    train_set.to_csv('training_data.csv', index=False)

    test_set ,val_set = train_test_split(test_set, test_size=0.5, random_state=42,stratify=test_set['label'] )
    test_set.to_csv('test_data.csv', index=False)
    val_set.to_csv('val_data.csv', index=False)
    return train_set,test_set,val_set

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 512

class ResNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super().__init__()

        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # grouped 3x3 conv
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=8
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

 
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=2),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        saved = self.shortcut(x)
        ####print("INPUT:" + str(x.shape))
        output = self.conv1(x)
        ####print("Conv1:" + str(output.shape))
        output = self.bn1(output)
        ####print("BN:" + str(output.shape))
        output = self.relu(output)
        ####print("RELU:" + str(output.shape))

        output = self.conv2(output)
        ####print("CONV2:" + str(output.shape)) 
        output = self.bn2(output)
        ####print("BN2:" + str(output.shape))
        output = self.relu(output)
        ####print("RELU2:" + str(output.shape))
        output = self.conv3(output)
        ####print("CONV3:" + str(output.shape)) 
        output = self.bn3(output)
        ###print("BN3:" + str(output.shape)) 
        ###print("SAVED:" + str(saved.shape))
        output += saved

        output =self.relu(output)
        return output
class CRNN(nn.Module):
    def __init__(self, n_mels: int = N_MELS, hidden_size: int = 128, num_layers: int = 1):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        reduced_mels = n_mels // 8
        rnn_input_size = 64 * reduced_mels

        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lengths = torch.sum(torch.any(x != 0, dim=1), dim=1)  # (B,)
        x = self.cnn(x)  # (B, C, mel', time')
        batch_size, channels, mel_bins, time_steps = x.size()

        x = x.permute(0, 3, 1, 2).contiguous()   # (B, time', C, mel')
        x = x.view(batch_size, time_steps, channels * mel_bins)

        reduced_lengths = torch.clamp(lengths // 8, min=1)

        packed = pack_padded_sequence(
            x,
            reduced_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.rnn(packed)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        final_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)

        logits = self.classifier(final_hidden).squeeze(1)
        return logits


class CQC_mode(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1)
        
        self.bn = nn.BatchNorm2d(64)
        
        self.ReLU = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=3,stride=2)
        self.ResNext_1 = ResNextBlock(64,128)
        self.ResNext_2 = ResNextBlock(128,128)
        self.ResNext_3 = ResNextBlock(128,256)
        self.ResNext_4 = ResNextBlock(256,256)
        self.ResNext_5 = ResNextBlock(256,512)
        self.ResNext_6 = ResNextBlock(512,512)

        self.AAPool = nn.AdaptiveAvgPool2d((1,7))
        self.flat = nn.Flatten(start_dim=1)
        self.linear =  nn.Linear(512*1*7, 100)   
        self.output = nn.Linear(100,1)

        self.softmax =nn.Softmax(1)     
    def forward(self,x):

        
        x = self.conv_1(x)
        #x = x.unsqueeze(0)
        #print("Cov: " + str(x.shape))
        x = self.bn(x)
        #print("Bn: " + str(x.shape))
        x = self.ReLU(x)
        #print("Relu: " + str(x.shape))
        x = self.pooling(x)
        #print("Pool: " + str(x.shape))
        x = self.ResNext_1(x)
        #print("RES_1: "+ str(x.shape))
        x = self.ResNext_2(x)
        #print("RES_2: "+ str(x.shape))
        x = self.ResNext_3(x)
        #print("RES_3: "+ str(x.shape))
        x = self.ResNext_4(x)
        #print("RES_4: "+ str(x.shape))
        x = self.ResNext_5(x)
        #print("RES_5: "+ str(x.shape))
        x = self.ResNext_6(x)
        #print("RES_6: "+ str(x.shape))
        x =self.AAPool(x)
        #print("AA: "+ str(x.shape))
        x = self.flat(x)
        x= self.linear(x)
        #print("Linear: "+ str(x.shape))
        x= self.output(x)
        #print("OUT: "+ str(x.shape))
        #print(x)
        return x
def predict(model,test_loader):
    
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for inputs, _ in test_loader:  # if you don't have labels, use _
            inputs = inputs.to("cuda")

            # forward pass
            outputs = model(inputs).view(-1)  # for binary classification
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).long()

            # collect
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
            
    df = pd.DataFrame({
        "prediction": all_preds,
        "probability": all_probs
    })
    df.to_csv("test_predictions.csv", index=False)
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    
    model.to(device)

    scaler = torch.amp.GradScaler()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0

    for epoch in range(epochs):

        # ===== TRAIN =====
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device):
                outputs = model(inputs).view(-1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # ===== VALIDATION =====
        model.eval()
        all_preds = []
        total_loss = 0
        correct = 0
        total = 0

        criterion = nn.BCEWithLogitsLoss()  # or your existing loss

        with torch.no_grad(), torch.amp.autocast(device_type=device):
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float()

                outputs = model(inputs).view(-1)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).long()
                all_preds.extend(predicted.cpu().numpy())

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct / total
        avg_loss = total_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.2f}%")

        # Save last
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, "last.pth")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }, "best.pth")
            print("✅ Saved new BEST model!")
            # Save predictions to CSV (outside the if)
    
            pd.DataFrame({"prediction": all_preds}).to_csv("preds.csv", index=False)


if __name__ == "__main__":
    train,test,val = load_pkl("10s_log_mel_shuffled.pkl")
    
    device = cuda.get_current_device()

    X_tensor = torch.tensor(np.stack(train["cqcc"])   ).unsqueeze(1)
    y_tensor = torch.tensor(train["label"].values, dtype=torch.int)  # change if needed
    

    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
    )
    X_tensor = torch.tensor(np.stack(val["cqcc"])   ).unsqueeze(1)
    
    
    y_tensor = torch.tensor(val["label"].values, dtype=torch.int)  # change if needed
    

    dataset = TensorDataset(X_tensor, y_tensor)
    val_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
    )
    
    
    
    X_tensor = torch.tensor(np.stack(test["cqcc"])   ).unsqueeze(1)
    
    
    y_tensor = torch.tensor(test["label"].values, dtype=torch.int)  # change if needed
    

    dataset = TensorDataset(X_tensor, y_tensor)
    test_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
    )
    
    model = CRNN()
    train_model(model,train_loader,val_loader,50)

    
    model.eval()
    predict(model,test_loader)