import re
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from pathlib import Path
from typing import List

from config import config

logger = logging.getLogger(__name__)


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
    for filename in data_files[:20]:
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
            np.corrcoef(arr[:-1], arr[1:])[0, 1] if len(arr) > 1 else 0,
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
                    total_power, dominant_freq, spectral_centroid, spectral_rolloff,
                    np.max(power_spectrum), np.mean(power_spectrum), np.std(power_spectrum),
                    np.var(power_spectrum),
                    skew(power_spectrum) if len(power_spectrum) > 2 else 0,
                    kurtosis(power_spectrum) if len(power_spectrum) > 3 else 0,
                    np.median(power_spectrum), entropy(power_spectrum + 1e-10),
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


def load_physionet_gait_data_corrected():
    data_path = Path(config.data_dir)
    logger.info(f"Loading PhysioNet gait data with CORRECTED labeling from {data_path}")

    print("Analyzing filename patterns...")
    analyze_filenames_and_suggest_labeling(config.data_dir)

    all_data = []
    successful_loads = 0
    label_stats = {'control': 0, 'parkinson': 0, 'unknown': 0}
    labeling_issues = []

    txt_files = []
    for pattern in ['*.txt', '*.dat', '*.csv']:
        txt_files.extend(list(data_path.rglob(pattern)))

    logger.info(f"Found {len(txt_files)} potential data files")

    if len(txt_files) == 0:
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
                f"Parkinson's={label_stats['parkinson']}")

    if labeling_issues:
        logger.warning(f"Files with labeling issues ({len(labeling_issues)}): {labeling_issues[:10]}")

    if successful_loads < 10:
        raise ValueError("Insufficient real data for training. Please verify data directory and file formats.")

    if label_stats['control'] == 0 or label_stats['parkinson'] == 0:
        raise ValueError("Severe class imbalance - one class has zero samples.")

    return all_data


def handle_unknown_labels_automatically(all_data):
    from scipy.stats import skew
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

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
            else:
                class_counts = np.bincount([i['label'] for i in all_data if i['label'] is not None])
                item['label'] = int(np.argmin(class_counts))

    except Exception as e:
        logger.warning(f"Automatic classification failed: {e}")
        class_counts = np.bincount([i['label'] for i in known_data])
        minority_class = int(np.argmin(class_counts))
        for item in unknown_data:
            item['label'] = minority_class

    return all_data


def load_physionet_gait_data():
    all_data = load_physionet_gait_data_corrected()
    all_data = handle_unknown_labels_automatically(all_data)
    all_data = [item for item in all_data if item['label'] is not None]

    if len(all_data) == 0:
        raise ValueError("No valid labeled data available for training")

    labels = [item['label'] for item in all_data]
    class_counts = np.bincount(labels)
    logger.info(f"Final dataset: {len(all_data)} samples")
    logger.info(f"FINAL CORRECTED Class distribution: Control={class_counts[0] if len(class_counts) > 0 else 0}, "
                f"Parkinson's={class_counts[1] if len(class_counts) > 1 else 0}")

    return all_data


def prepare_advanced_datasets(all_data):
    import gc
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
