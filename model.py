import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config


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
