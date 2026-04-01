import random
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import pickle
import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader


# =========================================================
# CONFIG
# =========================================================
DATASET_ROOT = "/Users/axgelx/Downloads/Dataset"
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 512
# The above code snippet defines some constants for a machine learning model training process in
# Python. Specifically, it sets the batch size to 8, the number of epochs to 20, the learning rate to
# 0.001, and the random seed to 42. These constants are commonly used in training neural networks and
# other machine learning models to control various aspects of the training process.

BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

MODEL_SAVE_PATH = "best_crnn_model.pth"

VALID_EXTENSIONS = {".mp3"}


# =========================================================
# DEVICE
# =========================================================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")   
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# DATA COLLECTION
# =========================================================
def collect_samples(dataset_root: str) -> List[Tuple[str, int, str]]:
    """
    Returns a list of:
    (file_path, label, duration_group)

    label:
        REAL = 0
        FAKE = 1
    duration_group:
        "5s" or "10s"
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    samples: List[Tuple[str, int, str]] = []
    class_map = {"REAL": 0, "FAKE": 1}
    duration_groups = ["5s", "10s"]

    for duration in duration_groups:
        for cls_name, label in class_map.items():
            folder = root / duration / cls_name
            if not folder.exists():
                print(f"Warning: folder not found, skipping: {folder}")
                continue

            for file_path in folder.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
                    samples.append((str(file_path), label, duration))

    if not samples:
        raise ValueError("No audio files found. Check your folder structure and dataset path.")

    return samples


# =========================================================
# SPLIT DATASET
# =========================================================
def split_dataset(samples: List[Tuple[str, int, str]]):
    """
    Split dataset into:
    70% train / 15% validation / 15% test
    using stratification on (label + duration)
    """
    stratify_labels = [f"{label}_{duration}" for _, label, duration in samples]

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=stratify_labels
    )

    temp_stratify = [f"{label}_{duration}" for _, label, duration in temp_samples]

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_stratify
    )

    return train_samples, val_samples, test_samples


def print_distribution(samples: List[Tuple[str, int, str]], name: str) -> None:
    print(f"\n{name} distribution:")
    counts = {}
    for _, label, duration in samples:
        label_name = "REAL" if label == 0 else "FAKE"
        key = f"{duration}_{label_name}"
        counts[key] = counts.get(key, 0) + 1

    for k in sorted(counts.keys()):
        print(f"{k}: {counts[k]}")


# =========================================================
# AUDIO -> MEL SPECTROGRAM
# =========================================================
def audio_to_log_mel(
    file_path: str,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """
    Load audio and convert to standardized log-Mel spectrogram.
    Output shape: (n_mels, time_frames)
    """
    try:
        y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {file_path}\n{e}")

    if len(y) == 0:
        raise ValueError(f"Empty audio file: {file_path}")

    # Normalize waveform
    max_val = np.max(np.abs(y)) + 1e-9
    y = y / max_val

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Per-sample normalization
    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-9
    log_mel = (log_mel - mean) / std

    return log_mel.astype(np.float32)


# =========================================================
# DATASET
# =========================================================
class AudioDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int, str]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        file_path, label, duration = self.samples[idx]
        spectrogram = audio_to_log_mel(file_path)  # (n_mels, time_frames)

        spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        time_steps = spectrogram_tensor.shape[1]

        return spectrogram_tensor, label_tensor, time_steps, file_path, duration



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


# =========================================================
# CRNN MODEL
# =========================================================
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

    def forward(self, x, lengths):
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


# =========================================================
# TRAIN / EVALUATE
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels, lengths in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(inputs, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    all_paths = []
    all_durations = []

    for inputs, labels, lengths in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(inputs, lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * inputs.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    return {
        "loss": avg_loss,
        "acc": acc,
        "preds": np.array(all_preds),
        "probs": np.array(all_probs),
        "labels": np.array(all_labels),

    }


def print_duration_accuracy(results):
    durations = np.array(results["durations"])
    labels = results["labels"]
    preds = results["preds"]

    for duration in sorted(set(durations)):
        mask = durations == duration
        if np.sum(mask) > 0:
            acc = accuracy_score(labels[mask], preds[mask])
            print(f"{duration} accuracy: {acc:.4f} ({np.sum(mask)} samples)")

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




# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(RANDOM_SEED)

    print(f"Using device: {DEVICE}")



    train_samples,test_samples,val_samples = load_pkl("10s_log_mel_shuffled.pkl")

    print(f"\nTrain size: {len(train_samples)}")
    print(f"Validation size: {len(val_samples)}")
    print(f"Test size: {len(test_samples)}")


    train_dataset = AudioDatasetV2(train_samples)
    val_dataset = AudioDatasetV2(val_samples)
    test_dataset = AudioDatasetV2(test_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = CRNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_results = evaluate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch [{epoch}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_results['loss']:.4f} | Val Acc: {val_results['acc']:.4f}"
        )

        if val_results["acc"] > best_val_acc:
            best_val_acc = val_results["acc"]
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Best model saved to {MODEL_SAVE_PATH}")

    print("\nLoading best model for final testing...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    test_results = evaluate(model, test_loader, criterion, DEVICE)

    print("\n========== TEST RESULTS ==========")
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Acc : {test_results['acc']:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        test_results["labels"],
        test_results["preds"],
        target_names=["REAL", "FAKE"]
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(test_results["labels"], test_results["preds"]))

    #print("\nDuration-wise Accuracy:")
    #print_duration_accuracy(test_results)


if __name__ == "__main__":
    main()