from dataclasses import dataclass


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
