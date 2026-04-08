import gc
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold

from config import config
from data import load_physionet_gait_data, prepare_advanced_datasets, create_data_loaders
from model import HighAccuracyGaitClassifier, FocalLoss

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

        for grf_batch, cop_batch, feat_batch, labels in train_loader:
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

    return {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'f1_score': f1_score(all_labels, all_predictions, average='weighted'),
        'precision': precision_score(all_labels, all_predictions, average='weighted'),
        'recall': recall_score(all_labels, all_predictions, average='weighted'),
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

    accuracies = [r['accuracy'] for r in fold_results]
    f1_scores = [r['f1_score'] for r in fold_results]

    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_f1_score': np.mean(f1_scores),
        'std_f1_score': np.std(f1_scores),
        'fold_results': fold_results,
        'overall_predictions': np.array(all_predictions),
        'overall_true_labels': np.array(all_true_labels)
    }


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
                xticklabels=["Healthy", "Parkinson's"],
                yticklabels=["Healthy", "Parkinson's"])
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
    print(f"Overall Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"Overall F1-Score: {results['mean_f1_score']:.4f} ± {results['std_f1_score']:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(results['overall_true_labels'],
                                results['overall_predictions'],
                                target_names=["Healthy", "Parkinson's"]))
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
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"Mean F1-Score: {results['mean_f1_score']:.4f} ± {results['std_f1_score']:.4f}")

    return results


if __name__ == "__main__":
    main()
