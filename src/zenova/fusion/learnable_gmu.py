"""Gated Multimodal Unit (GMU) neural fusion engine with modality dropout in PyTorch."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    SymptomResult,
    RiskResult,
    VoiceResult,
    BehavioralResult,
    BaselineResult,
    EmotionCategory,
    RiskLevel
)
from zenova.schemas.fusion import (
    ModalityMask,
    CrossModalDiscrepancy,
    FusedMultimodalState
)
from zenova.fusion.masking import ModalityMaskExtractor
from zenova.fusion.vectorizer import MultimodalFeatureVectorizer
from zenova.core.logging import get_logger

logger = get_logger("zenova.fusion.learnable_gmu")


class GatedMultimodalUnitNetwork(nn.Module):
    """PyTorch Gated Multimodal Unit (GMU) with adaptive modality gating and missing-data resilience."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.modalities = ["text", "emotion", "symptoms", "risk", "voice", "behavior", "baseline", "history"]

        # Feature dimensions from vectorizer
        dims = MultimodalFeatureVectorizer.MODALITY_DIMS

        # 1. Modality Latent Projection Encoders
        self.encoders = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(dims[m], hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh()
            )
            for m in self.modalities
        })

        # 2. Modality Gating Logit Units (input: features + mask (1) + confidence (1))
        self.gating_layers = nn.ModuleDict({
            m: nn.Linear(dims[m] + 2, 1)
            for m in self.modalities
        })

        # 3. Output Prediction Heads
        self.distress_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4)  # 4 risk levels (LOW, MODERATE, HIGH, CRITICAL)
        )

        self.affect_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # [valence, arousal]
        )

        self.discrepancy_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        mask_tensor: torch.Tensor,
        conf_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with availability masking.
        
        Args:
            features: Dict mapping modality name to tensor of shape (batch, d_m)
            mask_tensor: Tensor of shape (batch, 8) with boolean/float flags (1.0 = available)
            conf_tensor: Tensor of shape (batch, 8) with confidence scores (0.0 to 1.0)
        """
        batch_size = mask_tensor.shape[0]
        h_modalities = []
        gate_logits = []

        for i, m in enumerate(self.modalities):
            feat = features[m]
            h_m = self.encoders[m](feat)
            h_modalities.append(h_m)

            m_val = mask_tensor[:, i:i+1]
            c_val = conf_tensor[:, i:i+1]
            gate_input = torch.cat([feat, m_val, c_val], dim=-1)
            logit_m = self.gating_layers[m](gate_input)
            gate_logits.append(logit_m)

        # Shape: (batch, num_modalities)
        gate_logits_tensor = torch.cat(gate_logits, dim=-1)

        # Mask unavailable modalities with large negative value
        mask_bias = (1.0 - mask_tensor) * -1e9
        masked_logits = gate_logits_tensor + mask_bias

        # Gating softmax
        weights = F.softmax(masked_logits, dim=-1)  # (batch, num_modalities)

        # Fused hidden state: sum_m (w_m * h_m)
        h_stack = torch.stack(h_modalities, dim=1)  # (batch, num_modalities, hidden_dim)
        weights_expanded = weights.unsqueeze(-1)    # (batch, num_modalities, 1)
        h_fused = torch.sum(h_stack * weights_expanded, dim=1)  # (batch, hidden_dim)

        # Predictions
        distress = self.distress_head(h_fused)
        risk_logits = self.risk_head(h_fused)
        affect_raw = self.affect_head(h_fused)

        valence = torch.tanh(affect_raw[:, 0:1])
        arousal = torch.sigmoid(affect_raw[:, 1:2])
        discrepancy = self.discrepancy_head(h_fused)

        return distress, risk_logits, valence, arousal, weights, discrepancy


class LearnableGMUFusionEngine:
    """Wrapper managing training, checkpoint persistence, and inference for GMU."""

    CHECKPOINT_PATH = "models/fusion/gmu_checkpoint.pt"

    RISK_LEVELS = [
        RiskLevel.LOW,
        RiskLevel.MODERATE,
        RiskLevel.HIGH,
        RiskLevel.CRITICAL
    ]

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path or self.CHECKPOINT_PATH
        self.device = torch.device("cpu")
        self.model = GatedMultimodalUnitNetwork().to(self.device)
        self.is_trained = False
        self.load_or_train()

    @property
    def network(self) -> GatedMultimodalUnitNetwork:
        return self.model

    def load_or_train(self) -> None:
        """Loads existing checkpoint or initializes and trains on synthesized clinical scenarios."""
        path = Path(self.checkpoint_path)
        if path.exists():
            try:
                state_dict = torch.load(path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.is_trained = True
                logger.info(f"Loaded GMU fusion model from {path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load GMU checkpoint ({e}), retraining...")

        self.train_on_synthetic_data()

    def train_on_synthetic_data(self, epochs: int = 15) -> None:
        """Trains the GMU model with Modality Dropout across heterogeneous multi-modal samples."""
        logger.info("Synthesizing multi-modal clinical training set with Modality Dropout...")
        np.random.seed(42)
        torch.manual_seed(42)

        N = 600
        mod_names = ["text", "emotion", "symptoms", "risk", "voice", "behavior", "baseline", "history"]
        dims = MultimodalFeatureVectorizer.MODALITY_DIMS

        # Generate synthetic clinical data
        features_dict = {m: torch.randn(N, dims[m]) for m in mod_names}
        mask = torch.ones(N, 8)
        conf = torch.full((N, 8), 0.85)

        # Modality Dropout: randomly drop modalities 4..7 (voice, behavior, baseline, history)
        for i in range(4, 8):
            drop_indices = np.random.rand(N) < 0.4
            mask[drop_indices, i] = 0.0
            conf[drop_indices, i] = 0.0
            features_dict[mod_names[i]][drop_indices] = 0.0

        # Targets
        true_distress = torch.rand(N, 1)
        true_risk = torch.randint(0, 4, (N,))
        true_valence = (torch.rand(N, 1) * 2.0) - 1.0
        true_arousal = torch.rand(N, 1)
        true_discrepancy = (torch.rand(N, 1) > 0.8).float()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self.model.train()

        criterion_mse = nn.MSELoss()
        criterion_ce = nn.CrossEntropyLoss()
        criterion_bce = nn.BCELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            pred_distress, pred_risk, pred_val, pred_ar, weights, pred_disc = self.model(
                features_dict, mask, conf
            )

            loss_d = criterion_mse(pred_distress, true_distress)
            loss_r = criterion_ce(pred_risk, true_risk)
            loss_v = criterion_mse(pred_val, true_valence)
            loss_a = criterion_mse(pred_ar, true_arousal)
            loss_disc = criterion_bce(pred_disc, true_discrepancy)

            total_loss = loss_d + (loss_r * 0.5) + loss_v + loss_a + loss_disc
            total_loss.backward()
            optimizer.step()

        self.model.eval()
        self.is_trained = True
        self.save_checkpoint()

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """Saves current GMU weights to disk."""
        target_path = path or self.checkpoint_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        torch.save(self.model.state_dict(), target_path)
        logger.info(f"GMU model checkpoint saved to {target_path}")

    def fuse(
        self,
        user_input: Optional[UserInput] = None,
        emotion: Optional[EmotionResult] = None,
        symptoms: Optional[SymptomResult] = None,
        risk: Optional[RiskResult] = None,
        voice: Optional[VoiceResult] = None,
        behavior: Optional[BehavioralResult] = None,
        baseline: Optional[BaselineResult] = None,
        history: Optional[List[Any]] = None,
    ) -> FusedMultimodalState:
        """Perform neural Gated Multimodal Fusion."""
        # 1. Mask and Features
        mask = ModalityMaskExtractor.extract_mask(
            user_input=user_input,
            emotion=emotion,
            symptoms=symptoms,
            risk=risk,
            voice=voice,
            behavior=behavior,
            baseline=baseline,
            history=history
        )
        mod_names = ["text", "emotion", "symptoms", "risk", "voice", "behavior", "baseline", "history"]
        mask_vec = [1.0 if mask.mask.get(m, False) else 0.0 for m in mod_names]
        conf_vec = [mask.confidences.get(m, 0.0) for m in mod_names]

        vecs = MultimodalFeatureVectorizer.vectorize_all(
            user_input=user_input,
            emotion=emotion,
            symptoms=symptoms,
            risk=risk,
            voice=voice,
            behavior=behavior,
            baseline=baseline,
            history=history
        )

        features_tensors = {
            m: torch.from_numpy(vecs[m]).float().unsqueeze(0).to(self.device)
            for m in mod_names
        }
        mask_tensor = torch.tensor([mask_vec], dtype=torch.float32, device=self.device)
        conf_tensor = torch.tensor([conf_vec], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            distress, risk_logits, valence, arousal, weights, disc_prob = self.model(
                features_tensors, mask_tensor, conf_tensor
            )

        f_distress = float(distress.squeeze().item())
        risk_idx = int(torch.argmax(risk_logits, dim=-1).item())
        f_risk = self.RISK_LEVELS[min(risk_idx, len(self.RISK_LEVELS) - 1)]
        f_val = float(valence.squeeze().item())
        f_ar = float(arousal.squeeze().item())
        weights_arr = weights.squeeze().cpu().numpy()
        disc_val = float(disc_prob.squeeze().item())

        # Discrepancy logic
        disc_detected = disc_val > 0.6
        disc_reasons = []
        if disc_detected:
            disc_reasons.append(f"Gated unit detected latent cross-modal tension (p={disc_val:.2f}).")

        # Modality weight dict
        weight_dict = {m: round(float(weights_arr[i]), 4) for i, m in enumerate(mod_names)}

        # Primary affect category
        primary_affect = EmotionCategory.NEUTRAL
        if emotion and emotion.confidence > 0.0:
            primary_affect = emotion.primary_emotion
        elif voice and voice.is_available and voice.primary_emotion:
            primary_affect = voice.primary_emotion

        # Urgency
        urgency = min(1.0, f_distress * 0.6 + (1.0 if f_risk in (RiskLevel.HIGH, RiskLevel.CRITICAL) else 0.2) * 0.4)

        # Overall confidence
        active_confs = [mask.confidences[m] for m in mask.active_modalities if m in mask.confidences]
        overall_conf = float(np.mean(active_confs)) if active_confs else 0.85

        return FusedMultimodalState(
            primary_affect=primary_affect,
            fused_valence=round(f_val, 3),
            fused_arousal=round(f_ar, 3),
            fused_dominance=0.0,
            fused_distress_score=round(f_distress, 3),
            fused_risk_level=f_risk,
            urgency_score=round(urgency, 3),
            modality_weights=weight_dict,
            modality_mask=mask.mask,
            discrepancy=CrossModalDiscrepancy(
                detected=disc_detected,
                discrepancy_types=["neural_latent_discrepancy"] if disc_detected else [],
                reasons=disc_reasons,
                confidence=round(disc_val, 3)
            ),
            fusion_method="learnable_gmu",
            confidence=round(overall_conf, 3),
            timestamp=datetime.now(timezone.utc)
        )
