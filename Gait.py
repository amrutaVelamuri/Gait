import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from datetime import datetime
from pathlib import Path
import json
from dataclasses import dataclass
from typing import List, Tuple
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from scipy.fft import fft, fftfreq
from scipy import stats
import gc
import re

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HighAccuracyConfig:
    data_dir: str = "physionet.org/files/gaitpdb/1.0.0/"
    target_length: int = 150
    use_filtering: bool = True
    hidden_dim: int = 128
    num_lstm_layers: int = 2
    bidirectional: bool = True
    dropout_rate: float = 0.4
    use_attention: bool = True
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    patience: int = 12
    gradient_clip: float = 1.0
    use_advanced_features: bool = True
    max_features: int = 80
    use_feature_selection: bool = True
    cv_folds: int = 5
    random_state: int = 42
    use_augmentation: bool = True
    augmentation_factor: int = 3
    noise_std: float = 0.02
    num_workers: int = 0
    pin_memory: bool = False
    use_mixed_precision: bool = False
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    use_class_weights: bool = True
    use_oversampling: bool = True
    save_results: bool = True
    results_dir: str = "high_accuracy_results"

config = HighAccuracyConfig()

def determine_label_from_physionet_filename_corrected(filename):
    filename_lower = filename.lower()
    name_without_ext = filename_lower.replace('.txt', '').replace('.dat', '').replace('.csv', '')

    if re.search(r'co\d+', name_without_ext):
        return 0
    elif re.search(r'pt\d+', name_without_ext):
        return 1

    if 'co' in name_without_ext and 'pt' not in name_without_ext:
        return 0
    elif 'pt' in name_without_ext and 'co' not in name_without_ext:
        return 1

    control_indicators = ['control', 'ctrl', 'healthy', 'normal', 'nc', 'h']
    parkinson_indicators = ['patient', 'parkinson', 'pd', 'diseased', 'abnormal', 'p']

    for indicator in control_indicators:
        if indicator in name_without_ext:
            return 0

    for indicator in parkinson_indicators:
        if indicator in name_without_ext:
            return 1

    return None

def analyze_filenames_and_suggest_labeling(data_dir):
    data_path = Path(data_dir)

    all_files = []
    for pattern in ['*.txt', '*.dat', '*.csv']:
        all_files.extend(list(data_path.rglob(pattern)))

    data_files = []
    for file_path in all_files:
        filename = file_path.name.lower()
        if not any(skip in filename for skip in ['format', 'demographics', 'readme', 'index', 'robots', 'header']):
            data_files.append(file_path.name)

    print(f"\n=== FILENAME ANALYSIS FOR LABELING ===")
    print(f"Total data files found: {len(data_files)}")

    co_files = [f for f in data_files if 'co' in f.lower()]
    pt_files = [f for f in data_files if 'pt' in f.lower()]
    other_files = [f for f in data_files if 'co' not in f.lower() and 'pt' not in f.lower()]

    print(f"\nFiles containing 'Co' (likely Controls): {len(co_files)}")
    print("Examples:", co_files[:5])

    print(f"\nFiles containing 'Pt' (likely Patients): {len(pt_files)}")
    print("Examples:", pt_files[:5])

    print(f"\nOther files (need manual inspection): {len(other_files)}")
    print("Examples:", other_files[:10])

    print(f"\n=== TESTING CORRECTED LABELING ===")
    sample_files = data_files[:20]

    for filename in sample_files:
        label = determine_label_from_physionet_filename_corrected(filename)
        label_str = "Control" if label == 0 else "Parkinson's" if label == 1 else "UNKNOWN"
        print(f"{filename:20} -> {label_str}")

    return {
        'total_files': len(data_files),
        'co_files': len(co_files),
        'pt_files': len(pt_files),
        'other_files': len(other_files),
        'all_filenames': data_files
    }

def extract_comprehensive_features(arr: np.ndarray) -> List[float]:
    if len(arr) == 0:
        return [0.0] * 40

    try:
        arr = pd.to_numeric(arr, errors='coerce')
        arr = np.array(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return [0.0] * 40
    except:
        return [0.0] * 40

    features = []
    mean_val = np.mean(arr)
    std_val = np.std(arr)

    features.extend([
        mean_val, std_val, np.min(arr), np.max(arr), np.median(arr),
        np.var(arr), skew(arr) if len(arr) > 2 else 0,
        kurtosis(arr) if len(arr) > 3 else 0,
        np.ptp(arr),
        np.percentile(arr, 25), np.percentile(arr, 75),
        np.percentile(arr, 10), np.percentile(arr, 90),
        np.mean(np.abs(arr - mean_val)),
        np.sqrt(np.mean((arr - mean_val)**2))
    ])

    if len(arr) > 1:
        diff_arr = np.diff(arr)
        features.extend([
            np.mean(diff_arr),
            np.std(diff_arr),
            np.max(np.abs(diff_arr)),
            np.sum(diff_arr > 0) / len(diff_arr) if len(diff_arr) > 0 else 0,
            np.sqrt(np.mean(diff_arr**2)) if len(diff_arr) > 0 else 0,
            np.mean(np.abs(diff_arr)) if len(diff_arr) > 0 else 0,
            np.corrcoef(arr[:-1], arr[1:])[0,1] if len(arr) > 1 else 0,
            len(find_peaks(arr)[0]) / len(arr),
            len(find_peaks(-arr)[0]) / len(arr),
            np.trapz(np.abs(arr))
        ])
    else:
        features.extend([0.0] * 10)

    if len(arr) >= 8:
        try:
            fft_vals = np.fft.rfft(arr - mean_val)
            power_spectrum = np.abs(fft_vals)**2
            freqs = np.fft.rfftfreq(len(arr))

            if len(power_spectrum) > 0 and np.sum(power_spectrum) > 1e-10:
                total_power = np.sum(power_spectrum)
                dominant_freq_idx = np.argmax(power_spectrum)
                dominant_freq = freqs[dominant_freq_idx] if dominant_freq_idx < len(freqs) else 0
                spectral_centroid = np.sum(freqs * power_spectrum) / total_power

                cumsum_power = np.cumsum(power_spectrum)
                rolloff_idx = np.where(cumsum_power >= 0.85 * total_power)[0]
                spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 and rolloff_idx[0] < len(freqs) else 0

                features.extend([
                    total_power,
                    dominant_freq,
                    spectral_centroid,
                    spectral_rolloff,
                    np.max(power_spectrum),
                    np.mean(power_spectrum),
                    np.std(power_spectrum),
                    np.var(power_spectrum),
                    skew(power_spectrum) if len(power_spectrum) > 2 else 0,
                    kurtosis(power_spectrum) if len(power_spectrum) > 3 else 0,
                    np.median(power_spectrum),
                    entropy(power_spectrum + 1e-10),
                    np.sum(power_spectrum[:len(power_spectrum)//4]) / total_power,
                    np.sum(power_spectrum[3*len(power_spectrum)//4:]) / total_power,
                    len(find_peaks(power_spectrum)[0])
                ])
            else:
                features.extend([0.0] * 15)
        except:
            features.extend([0.0] * 15)
    else:
        features.extend([0.0] * 15)

    return features[:40]

def advanced_signal_processing(arr, target_len=150):
    if len(arr) == 0:
        return np.zeros(target_len, dtype=np.float32)

    try:
        arr = pd.to_numeric(arr, errors='coerce')
        arr = np.array(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return np.zeros(target_len, dtype=np.float32)
    except:
        return np.zeros(target_len, dtype=np.float32)

    if len(arr) > 10:
        Q1 = np.percentile(arr, 25)
        Q3 = np.percentile(arr, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        arr = arr[(arr >= lower_bound) & (arr <= upper_bound)]
        if len(arr) == 0:
            arr = np.array([Q1])

    if config.use_filtering and len(arr) > 20:
        try:
            nyquist = 0.5 * 100
            cutoff = 10
            normal_cutoff = cutoff / nyquist
            if 0 < normal_cutoff < 1:
                b, a = butter(4, normal_cutoff, btype='low', analog=False)
                arr = filtfilt(b, a, arr)
        except:
            pass

    if len(arr) > 5:
        try:
            window_length = min(len(arr) if len(arr) % 2 == 1 else len(arr) - 1, 11)
            if window_length >= 3:
                arr = savgol_filter(arr, window_length, 3)
        except:
            pass

    if len(arr) >= 2:
        old_indices = np.linspace(0, len(arr)-1, len(arr))
        new_indices = np.linspace(0, len(arr)-1, target_len)
        interpolated = np.interp(new_indices, old_indices, arr)
        return interpolated.astype(np.float32)
    else:
        return np.full(target_len, arr[0] if len(arr) > 0 else 0.0, dtype=np.float32)

def load_physionet_gait_data_corrected():
    data_path = Path(config.data_dir)
    logger.info(f"Loading PhysioNet gait data with CORRECTED labeling from {data_path}")

    print("Analyzing filename patterns...")
    filename_analysis = analyze_filenames_and_suggest_labeling(config.data_dir)

    all_data = []
    successful_loads = 0
    label_stats = {'control': 0, 'parkinson': 0, 'unknown': 0}
    labeling_issues = []

    txt_files = []
    for pattern in ['*.txt', '*.dat', '*.csv']:
        txt_files.extend(list(data_path.rglob(pattern)))

    logger.info(f"Found {len(txt_files)} potential data files")

    if len(txt_files) == 0:
        logger.error("No data files found! Please check the data directory path.")
        raise FileNotFoundError(f"No data files found in {data_path}")

    logger.info(f"Processing {min(100, len(txt_files))} data files...")

    for file_path in txt_files[:100]:
        filename = file_path.name.lower()

        if any(skip in filename for skip in ['format', 'demographics', 'readme', 'index', 'robots', 'header']):
            continue

        try:
            data = None
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                for sep in ['\t', ' ', ',', ';']:
                    try:
                        data = pd.read_csv(file_path, sep=sep, header=None, encoding=encoding,
                                         on_bad_lines='skip', low_memory=False)
                        if data.shape[0] >= 50 and data.shape[1] >= 2:
                            break
                    except:
                        continue
                if data is not None and data.shape[0] >= 50:
                    break

            if data is None or data.shape[0] < 50:
                continue

            if data.shape[1] >= 8:
                try:
                    left_force = pd.to_numeric(data.iloc[:, 5], errors='coerce').fillna(0)
                    right_force = pd.to_numeric(data.iloc[:, 10], errors='coerce').fillna(0) if data.shape[1] > 10 else left_force
                    cop_x = pd.to_numeric(data.iloc[:, 1], errors='coerce').fillna(0)
                    cop_y = pd.to_numeric(data.iloc[:, 2], errors='coerce').fillna(0)
                except:
                    left_force = pd.to_numeric(data.iloc[:, -2], errors='coerce').fillna(0)
                    right_force = pd.to_numeric(data.iloc[:, -1], errors='coerce').fillna(0)
                    cop_x = pd.to_numeric(data.iloc[:, 1], errors='coerce').fillna(0)
                    cop_y = pd.to_numeric(data.iloc[:, 2], errors='coerce').fillna(0)
            else:
                left_force = pd.to_numeric(data.iloc[:, -2], errors='coerce').fillna(0)
                right_force = pd.to_numeric(data.iloc[:, -1], errors='coerce').fillna(0)
                cop_x = pd.to_numeric(data.iloc[:, 0], errors='coerce').fillna(0) if data.shape[1] > 2 else np.zeros(len(data))
                cop_y = pd.to_numeric(data.iloc[:, 1], errors='coerce').fillna(0) if data.shape[1] > 3 else np.zeros(len(data))

            total_grf = (left_force + right_force).values
            cop_data = np.column_stack([cop_x.values, cop_y.values])

            if len(total_grf) < 50 or np.all(total_grf == 0):
                continue

            label = determine_label_from_physionet_filename_corrected(filename)

            if label is None:
                labeling_issues.append(filename)
                logger.warning(f"Could not determine label for: {filename}")
                continue

            if label == 0:
                label_stats['control'] += 1
                subject_id = f"CO_{label_stats['control']:02d}"
            elif label == 1:
                label_stats['parkinson'] += 1
                subject_id = f"PT_{label_stats['parkinson']:02d}"

            all_data.append({
                'grf': total_grf,
                'cop': cop_data,
                'label': label,
                'subject': subject_id,
                'filename': filename
            })

            successful_loads += 1

            if successful_loads >= 60:
                break

        except Exception as e:
            logger.debug(f"Error processing {filename}: {e}")
            continue

    logger.info(f"Successfully loaded {successful_loads} real data samples")
    logger.info(f"CORRECTED Label distribution: Control={label_stats['control']}, "
                f"Parkinson's={label_stats['parkinson']}, Unknown={label_stats['unknown']}")

    if labeling_issues:
        logger.warning(f"Files with labeling issues ({len(labeling_issues)}): {labeling_issues[:10]}")

    if successful_loads < 10:
        logger.error(f"Insufficient real data loaded ({successful_loads} samples). Cannot proceed.")
        raise ValueError("Insufficient real data for training. Please verify data directory and file formats.")

    if label_stats['control'] == 0 or label_stats['parkinson'] == 0:
        logger.error("One class has no samples! Check your labeling logic.")
        raise ValueError("Severe class imbalance - one class has zero samples.")

    return all_data

def handle_unknown_labels_automatically(all_data):
    known_data = [item for item in all_data if item['label'] is not None]
    unknown_data = [item for item in all_data if item['label'] is None]

    if not unknown_data or len(known_data) < 10:
        return all_data

    logger.info(f"Attempting to classify {len(unknown_data)} unknown files using gait pattern analysis")

    known_features = []
    known_labels = []

    for item in known_data:
        grf = np.array(item['grf'])
        features = [
            np.mean(grf), np.std(grf), np.max(grf), np.min(grf),
            len(grf), np.var(grf), skew(grf) if len(grf) > 2 else 0
        ]
        known_features.append(features)
        known_labels.append(item['label'])

    known_features = np.array(known_features)
    known_labels = np.array(known_labels)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    try:
        scaler = StandardScaler()
        known_features_scaled = scaler.fit_transform(known_features)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(known_features_scaled, known_labels)

        for item in unknown_data:
            grf = np.array(item['grf'])
            features = [
                np.mean(grf), np.std(grf), np.max(grf), np.min(grf),
                len(grf), np.var(grf), skew(grf) if len(grf) > 2 else 0
            ]
            features_scaled = scaler.transform([features])

            prediction = clf.predict(features_scaled)[0]
            confidence = np.max(clf.predict_proba(features_scaled)[0])

            if confidence > 0.7:
                item['label'] = int(prediction)
                logger.info(f"Classified {item['filename']} as {'Control' if prediction == 0 else 'Parkinson'} "
                          f"(confidence: {confidence:.2f})")
            else:
                class_counts = np.bincount([item['label'] for item in all_data if item['label'] is not None])
                minority_class = np.argmin(class_counts)
                item['label'] = minority_class
                logger.info(f"Low confidence for {item['filename']}, assigned to minority class: {minority_class}")

    except Exception as e:
        logger.warning(f"Automatic classification failed: {e}")
        class_counts = np.bincount([item['label'] for item in known_data])
        minority_class = np.argmin(class_counts)

        for item in unknown_data:
            item['label'] = minority_class

    return all_data

def load_physionet_gait_data():
    all_data = load_physionet_gait_data_corrected()
    all_data = handle_unknown_labels_automatically(all_data)
    all_data = [item for item in all_data if item['label'] is not None]

    if len(all_data) == 0:
        logger.error("No valid labeled data found!")
        raise ValueError("No valid labeled data available for training")

    labels = [item['label'] for item in all_data]
    class_counts = np.bincount(labels)
    logger.info(f"Final dataset: {len(all_data)} samples")
    logger.info(f"FINAL CORRECTED Class distribution: Control={class_counts[0] if len(class_counts) > 0 else 0}, "
                f"Parkinson's={class_counts[1] if len(class_counts) > 1 else 0}")

    return all_data

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        attention_scores = self.attention_weights(lstm_output)
        attention_weights = torch.softmax(attention_scores, dim=1)
        weighted_output = torch.sum(attention_weights * lstm_output, dim=1)
        return weighted_output

class HighAccuracyGaitClassifier(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        hidden_dim = config.hidden_dim

        self.grf_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=config.num_lstm_layers,
            batch_first=True,
            dropout=config.dropout_rate if config.num_lstm_layers > 1 else 0,
            bidirectional=config.bidirectional
        )

        self.cop_lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=config.num_lstm_layers,
            batch_first=True,
            dropout=config.dropout_rate if config.num_lstm_layers > 1 else 0,
            bidirectional=config.bidirectional
        )

        lstm_output_dim = hidden_dim * (2 if config.bidirectional else 1)

        if config.use_attention:
            self.grf_attention = AttentionLayer(lstm_output_dim)
            self.cop_attention = AttentionLayer(lstm_output_dim)

        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate * 0.5)
        )

        fusion_dim = lstm_output_dim * 2 + hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate * 0.7),
            nn.Linear(hidden_dim, 2)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, grf, cop, features):
        grf_lstm_out, _ = self.grf_lstm(grf)
        cop_lstm_out, _ = self.cop_lstm(cop)

        if config.use_attention:
            grf_features = self.grf_attention(grf_lstm_out)
            cop_features = self.cop_attention(cop_lstm_out)
        else:
            grf_features = grf_lstm_out.mean(dim=1)
            cop_features = cop_lstm_out.mean(dim=1)

        processed_features = self.feature_processor(features)
        fused_features = torch.cat([grf_features, cop_features, processed_features], dim=1)

        return self.classifier(fused_features)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def augment_data(grf_seq, cop_seq, features):
    if not config.use_augmentation or np.random.random() > 0.6:
        return grf_seq, cop_seq, features

    aug_type = np.random.choice(['noise', 'scaling', 'time_shift'])
    grf_seq = np.array(grf_seq)
    cop_seq = np.array(cop_seq)
    features = np.array(features)

    if aug_type == 'noise':
        noise_grf = np.random.normal(0, config.noise_std * np.std(grf_seq), grf_seq.shape)
        noise_cop = np.random.normal(0, config.noise_std * np.std(cop_seq, axis=0), cop_seq.shape)
        grf_seq = grf_seq + noise_grf
        cop_seq = cop_seq + noise_cop
    elif aug_type == 'scaling':
        scale = np.random.uniform(0.9, 1.1)
        grf_seq = grf_seq * scale
        cop_seq = cop_seq * scale
    elif aug_type == 'time_shift':
        shift = np.random.randint(-len(grf_seq)//10, len(grf_seq)//10)
        grf_seq = np.roll(grf_seq, shift, axis=0)
        cop_seq = np.roll(cop_seq, shift, axis=0)

    return grf_seq, cop_seq, features

def prepare_advanced_datasets(all_data):
    logger.info("Preparing advanced datasets with real data processing...")

    processed_data = []

    for item in all_data:
        try:
            grf_data = np.array(item['grf'], dtype=np.float64)
            cop_data = np.array(item['cop'], dtype=np.float64)

            if cop_data.ndim == 1:
                cop_data = cop_data.reshape(-1, 1)
            if cop_data.shape[1] == 1:
                cop_data = np.column_stack([cop_data.flatten(), np.zeros(len(cop_data))])

            grf_processed = advanced_signal_processing(grf_data, config.target_length)
            cop_x_processed = advanced_signal_processing(cop_data[:, 0], config.target_length)
            cop_y_processed = advanced_signal_processing(cop_data[:, 1], config.target_length)
            cop_processed = np.column_stack([cop_x_processed, cop_y_processed])

            grf_features = extract_comprehensive_features(grf_data)
            cop_x_features = extract_comprehensive_features(cop_data[:, 0])
            cop_y_features = extract_comprehensive_features(cop_data[:, 1])

            combined_features = grf_features + cop_x_features + cop_y_features

            processed_data.append({
                'grf_seq': grf_processed.reshape(-1, 1),
                'cop_seq': cop_processed,
                'features': combined_features,
                'label': item['label'],
                'subject': item['subject']
            })

        except Exception as e:
            logger.warning(f"Error processing sample: {e}")
            continue

    if not processed_data:
        raise ValueError("No data could be processed!")

    X_grf = np.array([item['grf_seq'] for item in processed_data], dtype=np.float32)
    X_cop = np.array([item['cop_seq'] for item in processed_data], dtype=np.float32)
    X_features = np.array([item['features'] for item in processed_data], dtype=np.float32)
    y = np.array([item['label'] for item in processed_data])
    subjects = np.array([item['subject'] for item in processed_data])

    if X_grf.ndim == 2:
        X_grf = X_grf.reshape(X_grf.shape[0], X_grf.shape[1], 1)

    logger.info(f"Data shapes: GRF: {X_grf.shape}, COP: {X_cop.shape}, Features: {X_features.shape}")

    if config.use_augmentation:
        logger.info("Applying data augmentation for class balance...")
        aug_data_grf, aug_data_cop, aug_data_features, aug_labels, aug_subjects = [], [], [], [], []

        class_counts = np.bincount(y)
        max_class_count = np.max(class_counts)
        logger.info(f"Original class distribution: {class_counts}")

        for class_idx in range(len(class_counts)):
            class_mask = y == class_idx
            class_grf = X_grf[class_mask]
            class_cop = X_cop[class_mask]
            class_features = X_features[class_mask]
            class_subjects = subjects[class_mask]

            current_count = class_counts[class_idx]
            target_count = min(max_class_count * config.augmentation_factor, max_class_count * 2)
            augmentations_needed = max(0, target_count - current_count)

            if augmentations_needed > 0:
                aug_indices = np.random.choice(len(class_grf),
                                             size=min(augmentations_needed, len(class_grf) * 5),
                                             replace=True)

                for aug_counter, idx in enumerate(aug_indices):
                    orig_grf = class_grf[idx].reshape(-1)
                    orig_cop = class_cop[idx]
                    orig_feat = class_features[idx]
                    orig_subject = class_subjects[idx]

                    aug_grf, aug_cop, aug_feat = augment_data(orig_grf, orig_cop, orig_feat)

                    aug_grf = aug_grf.reshape(-1, 1)
                    if aug_cop.ndim == 1:
                        aug_cop = aug_cop.reshape(-1, 2)

                    aug_data_grf.append(aug_grf)
                    aug_data_cop.append(aug_cop)
                    aug_data_features.append(aug_feat)
                    aug_labels.append(class_idx)
                    aug_subjects.append(f"{orig_subject}_aug_{aug_counter}")

        if aug_data_grf:
            aug_data_grf = np.array(aug_data_grf)
            aug_data_cop = np.array(aug_data_cop)
            aug_data_features = np.array(aug_data_features)
            aug_labels = np.array(aug_labels)
            aug_subjects = np.array(aug_subjects)

            if aug_data_grf.ndim == 2:
                aug_data_grf = aug_data_grf.reshape(aug_data_grf.shape[0], aug_data_grf.shape[1], 1)

            X_grf = np.concatenate([X_grf, aug_data_grf], axis=0)
            X_cop = np.concatenate([X_cop, aug_data_cop], axis=0)
            X_features = np.concatenate([X_features, aug_data_features], axis=0)
            y = np.concatenate([y, aug_labels], axis=0)
            subjects = np.concatenate([subjects, aug_subjects], axis=0)

            logger.info(f"Final class distribution: {np.bincount(y)}")

    logger.info(f"Total samples after augmentation: {len(y)}")

    if config.use_feature_selection and X_features.shape[1] > config.max_features:
        logger.info("Applying feature selection...")
        selector = SelectKBest(f_classif, k=config.max_features)
        X_features = selector.fit_transform(X_features, y)
        logger.info(f"Selected {X_features.shape[1]} features")

    feature_scaler = RobustScaler()
    X_features = feature_scaler.fit_transform(X_features)

    grf_scaler = RobustScaler()
    X_grf_scaled = grf_scaler.fit_transform(X_grf.reshape(-1, X_grf.shape[-1])).reshape(X_grf.shape)

    cop_scaler = RobustScaler()
    X_cop_scaled = cop_scaler.fit_transform(X_cop.reshape(-1, X_cop.shape[-1])).reshape(X_cop.shape)

    return {
        'X_grf': X_grf_scaled.astype(np.float32),
        'X_cop': X_cop_scaled.astype(np.float32),
        'X_features': X_features.astype(np.float32),
        'y': y,
        'subjects': subjects,
        'scalers': {
            'feature': feature_scaler,
            'grf': grf_scaler,
            'cop': cop_scaler
        }
    }

def create_data_loaders(X_grf, X_cop, X_features, y, train_indices, val_indices):
    X_grf_train = torch.FloatTensor(X_grf[train_indices])
    X_cop_train = torch.FloatTensor(X_cop[train_indices])
    X_features_train = torch.FloatTensor(X_features[train_indices])
    y_train = torch.LongTensor(y[train_indices])

    train_dataset = TensorDataset(X_grf_train, X_cop_train, X_features_train, y_train)

    if config.use_class_weights:
        class_counts = np.bincount(y[train_indices])
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y[train_indices]]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                      num_samples=len(sample_weights),
                                      replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                sampler=sampler, num_workers=config.num_workers,
                                pin_memory=config.pin_memory, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                shuffle=True, num_workers=config.num_workers,
                                pin_memory=config.pin_memory, drop_last=True)

    X_grf_val = torch.FloatTensor(X_grf[val_indices])
    X_cop_val = torch.FloatTensor(X_cop[val_indices])
    X_features_val = torch.FloatTensor(X_features[val_indices])
    y_val = torch.LongTensor(y[val_indices])

    val_dataset = TensorDataset(X_grf_val, X_cop_val, X_features_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                          shuffle=False, num_workers=config.num_workers,
                          pin_memory=config.pin_memory)

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, device):
    if config.use_focal_loss:
        criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=config.learning_rate,
                                 weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-6
    )

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None

    logger.info("Starting enhanced training...")

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (grf_batch, cop_batch, feat_batch, labels) in enumerate(train_loader):
            grf_batch = grf_batch.to(device)
            cop_batch = cop_batch.to(device)
            feat_batch = feat_batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(grf_batch, cop_batch, feat_batch)
            loss = criterion(outputs, labels)

            loss.backward()

            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for grf_batch, cop_batch, feat_batch, labels in val_loader:
                grf_batch = grf_batch.to(device)
                cop_batch = cop_batch.to(device)
                feat_batch = feat_batch.to(device)
                labels = labels.to(device)

                outputs = model(grf_batch, cop_batch, feat_batch)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != old_lr:
            logger.info(f'Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == config.epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch [{epoch+1}/{config.epochs}] - '
                       f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                       f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                       f'LR: {current_lr:.6f}')

        if patience_counter >= config.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

        if epoch % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model with validation accuracy: {best_val_acc:.2f}%")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }

def evaluate_model(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for grf_batch, cop_batch, feat_batch, labels in data_loader:
            grf_batch = grf_batch.to(device)
            cop_batch = cop_batch.to(device)
            feat_batch = feat_batch.to(device)

            outputs = model(grf_batch, cop_batch, feat_batch)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'predictions': all_predictions,
        'true_labels': all_labels
    }

def run_subject_based_cross_validation(dataset_dict):
    logger.info("Starting subject-based cross-validation evaluation...")

    X_grf = dataset_dict['X_grf']
    X_cop = dataset_dict['X_cop']
    X_features = dataset_dict['X_features']
    y = dataset_dict['y']
    subjects = dataset_dict['subjects']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    unique_subjects = np.unique(subjects)
    subject_labels = []

    for subject in unique_subjects:
        subject_mask = subjects == subject
        subject_y = y[subject_mask]
        if len(subject_y) == 1:
            subject_label = subject_y[0]
        else:
            try:
                mode_result = stats.mode(subject_y, keepdims=True)
                subject_label = mode_result.mode[0]
            except:
                try:
                    subject_label = stats.mode(subject_y)[0][0]
                except:
                    unique_vals, counts = np.unique(subject_y, return_counts=True)
                    subject_label = unique_vals[np.argmax(counts)]
        subject_labels.append(subject_label)

    subject_labels = np.array(subject_labels)

    skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)

    fold_results = []
    all_predictions = []
    all_true_labels = []

    for fold, (train_subjects_idx, val_subjects_idx) in enumerate(skf.split(unique_subjects, subject_labels)):
        logger.info(f"\n=== Fold {fold + 1}/{config.cv_folds} ===")

        train_subjects = unique_subjects[train_subjects_idx]
        val_subjects = unique_subjects[val_subjects_idx]

        train_mask = np.isin(subjects, train_subjects)
        val_mask = np.isin(subjects, val_subjects)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        logger.info(f"Fold {fold + 1}: {len(train_subjects)} train subjects, {len(val_subjects)} val subjects")
        logger.info(f"Samples: {len(train_idx)} train, {len(val_idx)} validation")

        assert len(set(train_subjects) & set(val_subjects)) == 0, "Subject overlap detected!"

        train_loader, val_loader = create_data_loaders(
            X_grf, X_cop, X_features, y, train_idx, val_idx
        )

        model = HighAccuracyGaitClassifier(feature_dim=X_features.shape[1])
        model = model.to(device)

        training_history = train_model(model, train_loader, val_loader, device)
        val_results = evaluate_model(model, val_loader, device)

        fold_results.append({
            'fold': fold + 1,
            'accuracy': val_results['accuracy'],
            'f1_score': val_results['f1_score'],
            'precision': val_results['precision'],
            'recall': val_results['recall'],
            'training_history': training_history,
            'train_subjects': train_subjects.tolist(),
            'val_subjects': val_subjects.tolist()
        })

        all_predictions.extend(val_results['predictions'])
        all_true_labels.extend(val_results['true_labels'])

        logger.info(f"Fold {fold + 1} Results:")
        logger.info(f"  Accuracy: {val_results['accuracy']:.4f}")
        logger.info(f"  F1-Score: {val_results['f1_score']:.4f}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracies = [result['accuracy'] for result in fold_results]
    f1_scores = [result['f1_score'] for result in fold_results]

    overall_results = {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_f1_score': np.mean(f1_scores),
        'std_f1_score': np.std(f1_scores),
        'fold_results': fold_results,
        'overall_predictions': np.array(all_predictions),
        'overall_true_labels': np.array(all_true_labels)
    }

    return overall_results

def create_visualizations(results, dataset_dict):
    results_dir = Path(config.results_dir)
    results_dir.mkdir(exist_ok=True)

    plt.style.use('default')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    fold_nums = [r['fold'] for r in results['fold_results']]
    accuracies = [r['accuracy'] for r in results['fold_results']]
    f1_scores = [r['f1_score'] for r in results['fold_results']]

    ax1.bar(fold_nums, accuracies, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.axhline(y=results['mean_accuracy'], color='red', linestyle='--',
                label=f'Mean: {results["mean_accuracy"]:.3f}±{results["std_accuracy"]:.3f}')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Cross-Validation Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(fold_nums, f1_scores, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    ax2.axhline(y=results['mean_f1_score'], color='red', linestyle='--',
                label=f'Mean: {results["mean_f1_score"]:.3f}±{results["std_f1_score"]:.3f}')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Cross-Validation F1-Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    cm = confusion_matrix(results['overall_true_labels'], results['overall_predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Healthy', 'Parkinson\'s'],
                yticklabels=['Healthy', 'Parkinson\'s'])
    ax3.set_title('Overall Confusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')

    if results['fold_results']:
        last_fold_history = results['fold_results'][-1]['training_history']
        epochs = range(1, len(last_fold_history['val_accuracies']) + 1)

        ax4.plot(epochs, last_fold_history['train_losses'], 'b-', label='Training Loss', alpha=0.7)
        ax4.plot(epochs, last_fold_history['val_losses'], 'r-', label='Validation Loss', alpha=0.7)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(epochs, last_fold_history['val_accuracies'], 'g-', label='Val Accuracy', alpha=0.7)

        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4_twin.set_ylabel('Accuracy (%)')
        ax4.set_title('Training History (Last Fold)')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "="*50)
    print("="*50)
    print(f"Overall Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"Overall F1-Score: {results['mean_f1_score']:.4f} ± {results['std_f1_score']:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(results['overall_true_labels'],
                              results['overall_predictions'],
                              target_names=['Healthy', 'Parkinson\'s']))
    print("="*50)

def save_results(results, dataset_dict):
    if not config.save_results:
        return

    results_dir = Path(config.results_dir)
    results_dir.mkdir(exist_ok=True)

    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'configuration': config.__dict__,
        'mean_accuracy': float(results['mean_accuracy']),
        'std_accuracy': float(results['std_accuracy']),
        'mean_f1_score': float(results['mean_f1_score']),
        'std_f1_score': float(results['std_f1_score']),
        'fold_results': [
            {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
             for k, v in fold.items() if k != 'training_history'}
            for fold in results['fold_results']
        ],
        'data_info': {
            'total_samples': len(dataset_dict['y']),
            'feature_dimensions': dataset_dict['X_features'].shape[1],
            'sequence_length': dataset_dict['X_grf'].shape[1],
            'class_distribution': np.bincount(dataset_dict['y']).tolist()
        }
    }

    with open(results_dir / 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    predictions_df = pd.DataFrame({
        'true_labels': results['overall_true_labels'],
        'predictions': results['overall_predictions'],
        'correct': results['overall_true_labels'] == results['overall_predictions']
    })
    predictions_df.to_csv(results_dir / 'predictions.csv', index=False)

    logger.info(f"Results saved to {results_dir}")

def main():
    try:
        np.random.seed(config.random_state)
        torch.manual_seed(config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.random_state)

        logger.info("Loading PhysioNet gait data with CORRECTED labeling...")
        all_data = load_physionet_gait_data()
        logger.info(f"Loaded {len(all_data)} data samples")

        dataset_dict = prepare_advanced_datasets(all_data)
        logger.info("Dataset preparation completed")
        logger.info(f"Final dataset shape: GRF: {dataset_dict['X_grf'].shape}, "
                   f"COP: {dataset_dict['X_cop'].shape}, "
                   f"Features: {dataset_dict['X_features'].shape}")

        results = run_subject_based_cross_validation(dataset_dict)
        create_visualizations(results, dataset_dict)
        save_results(results, dataset_dict)

        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY ")
        print("="*60)
        print(f"Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        print(f"Mean F1-Score: {results['mean_f1_score']:.4f} ± {results['std_f1_score']:.4f}")

        return results

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()