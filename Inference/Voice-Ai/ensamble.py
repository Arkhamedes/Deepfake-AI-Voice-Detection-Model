import librosa
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
from torch.utils.data import TensorDataset
import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch
import scipy.fftpack as fft
import torch.nn as nn
import copy
import pickle
from sklearn.model_selection import train_test_split
from numba import cuda
import torch.optim as optim

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 512
## ===================MODELS===================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x):
        return self.block(x)

class UNetEncoderClassifier(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, dropout=0.5):
        super().__init__()
        f = base_filters
        self.inc   = DoubleConv(in_channels, f)
        self.down1 = DownBlock(f,    f*2)
        self.down2 = DownBlock(f*2,  f*4)
        self.down3 = DownBlock(f*4,  f*8)
        self.down4 = DownBlock(f*8,  f*16)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(f*16, f*4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(f*4, 1),
        )
    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return self.classifier(x)
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
        lengths = torch.sum(torch.any(x != 0, dim=2), dim=1)  # (B,)
        x = self.cnn(x)  # (B, C, mel', time')
        batch_size, channels, mel_bins, time_steps = x.size()

        x = x.permute(0, 3, 1, 2).contiguous()   # (B, time', C, mel')
        x = x.view(batch_size, time_steps, channels * mel_bins)

        reduced_lengths = torch.clamp(lengths // 8, min=1)
        #print("Original Lengths:", lengths.shape)
        #print("Reduced Lengths:", reduced_lengths.shape)
      

        _, (hidden, _) = self.rnn(x)

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
    
# ===================DATA PREP===================
class AudioDatasetV2(Dataset):
    def __init__(self, samples: pd.DataFrame):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        spectrogram, label = self.samples.iloc[idx]
        

        spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        time_steps = spectrogram_tensor.shape[1]

        return spectrogram_tensor, label_tensor, time_steps
def collate_fn(batch):
    spectrograms, labels, lengths = zip(*batch)

    specs_for_padding = [spec.T for spec in spectrograms]  # (time, mel)
    padded = pad_sequence(specs_for_padding, batch_first=True)  # (batch, max_time, mel)

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    labels_tensor = torch.stack(labels)

    # (batch, max_time, mel) -> (batch, 1, mel, max_time)
    padded = padded.permute(0, 2, 1).unsqueeze(1)

    return padded, labels_tensor, lengths_tensor
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

def audio_to_log_mel(
    segments: list,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH
) -> np.ndarray:
    log = []
    for segment in segments:
            
        # Normalize waveform
        max_val = np.max(np.abs(segment)) + 1e-9
        segment= segment / max_val

        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Per-sample normalization
        mean = np.mean(log_mel)
        std = np.std(log_mel) + 1e-9
        log_mel = (log_mel - mean) / std
        log.append(log_mel)

    return np.stack(log).astype(np.float32)
# ===================ENSEMBLE===================
class EnsembleAveraging(nn.Module):
    def __init__(self, models):
        super(EnsembleAveraging, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        predictions = [model(x) for model in self.models]
        # Average the predictions
        prediction_label = []
        for i in range(len(predictions)):
            
            probs = torch.sigmoid(predictions[i])
            predicted = (probs > 0.5).long()
            prediction_label.append(predicted.item())
        #avg_prediction = torch.mean(torch.stack(predictions), dim=0)
        #return avg_prediction
        final_prediction = max(set(prediction_label), key=prediction_label.count)
        return final_prediction



def audio_splitter(file_path, segment_length=10, sample_rate=SAMPLE_RATE):
    y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    total_length = len(y)
    segment_samples = segment_length * sample_rate
    segments = []
    for start in range(0, total_length, segment_samples):
        end = min(start + segment_samples, total_length)
        segment = y[start:end]
        if len(segment) < segment_samples:
            pad_length = segment_samples - len(segment)
            segment = np.pad(segment, (0, pad_length), mode='constant')
        segments.append(segment)
    return segments
# ===================MAIN===================

ResNext_checkpoint = torch.load("best_ResNEXT.pth", weights_only=True)

CRNN_checkpoint = torch.load("best_crnn_model.pth", weights_only=True)

UNetcheckpoint = torch.load("BEST_UNET_10sec.pt", weights_only=True)


ResNext = CQC_mode().to(device)
CRNN_model = CRNN().to(device)
UNet  = UNetEncoderClassifier(in_channels=1, base_filters=32).to(device)
ResNext.load_state_dict(ResNext_checkpoint["model_state_dict"])
CRNN_model.load_state_dict(CRNN_checkpoint)
UNet.load_state_dict(UNetcheckpoint)
Models = [ResNext,UNet,CRNN_model]
    
ensemble = EnsembleAveraging(Models)
files = []
import os
from fastapi import FastAPI, UploadFile, File
import torch
from fastapi.middleware.cors import CORSMiddleware
import tempfile

app = FastAPI()
origins = [
    "http://localhost:3000",  # React dev server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, OPTIONS, etc.
    allow_headers=["*"],
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ensemble.to(device)
ensemble.eval()




@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # === YOUR ORIGINAL PIPELINE ===
        segments = audio_splitter(tmp_path, segment_length=10)
        mel_list = audio_to_log_mel(segments)

        predictions = []

        for mel in mel_list:
            mel_tensor = (
                torch.tensor(mel, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                output = ensemble(mel_tensor)
                
                # ⚠️ adjust depending on your model output
                predictions.append(output)

        final_prediction = 1 if predictions.count(1) > 0.5 * len(predictions) else 0
        confidence = predictions.count(1) / len(predictions)

        return {
            "final_prediction": final_prediction,
            "num_segments": len(predictions),
            "segment_predictions": predictions,
            "confidence": confidence   
        }

    finally:
        os.remove(tmp_path)  # cleanup
        
        
def predict_audio(file,actual_label=None):
    
# Save uploaded file temporarily

    # === PIPELINE ===
    segments = audio_splitter(file, segment_length=10)
    mel_list = audio_to_log_mel(segments)

    predictions = []

    for mel in mel_list:
        mel_tensor = (
            torch.tensor(mel, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            output = ensemble(mel_tensor)

            # convert tensor -> scalar (IMPORTANT)
            predictions.append(output)


    final_prediction = 1 if predictions.count(1) > 0.75 * len(predictions) else 0
    confidence = predictions.count(1) / len(predictions)

    return {
        "file"
        "actual_prediction": actual_label,
        "final_prediction": final_prediction,
        "segment_predictions": predictions,
        "confidence": confidence
    }


        
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt
human = "/home/based/Downloads/archive(2)/KAGGLE/AUDIO/human"
non_human = "/home/based/Downloads/archive(2)/KAGGLE/AUDIO/nonhuman"
if __name__ == "__main__":
    
    ensemble.eval()
    all_predictions = []
    all_labels = []
    file_names = []
    for folder, label in [(human, 0), (non_human, 1)]:
        for file in os.listdir(folder):
            if file.endswith(".mp3"):
                file_path = os.path.join(folder, file)

                result = predict_audio(file_path)

                all_predictions.append(result["final_prediction"])
                all_labels.append(label)
                file_names.append(file)
    # save csv
    df_out = pd.DataFrame({
        "file": file_names,
        "true": all_labels,
        "pred": all_predictions
    })
    df_out.to_csv("ensemble_predictions.csv", index=False)
    cm = confusion_matrix(all_labels, all_predictions)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    classes = ["0", "1"]
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                    ha="center", va="center",
                    color="black")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    f1 = f1_score(all_labels, all_predictions, average="macro")
    print("F1 Score (macro):", f1)

    print(classification_report(all_labels, all_predictions))