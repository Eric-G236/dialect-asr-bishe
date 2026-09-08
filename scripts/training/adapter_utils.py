"""Adapter packaging/merge helpers for LoRA and DoRA fine-tuning."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from funasr.models.lora.layers import LoRALayer


class DoRALinear(nn.Linear, LoRALayer):
    """LoRA-style linear layer with DoRA magnitude/direction decomposition."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.lora_magnitude = nn.Parameter(torch.ones(out_features, 1))
            self._magnitude_initialized = False
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def _T(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.T if self.fan_in_fan_out else weight

    def reset_parameters(self) -> None:
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
            nn.init.ones_(self.lora_magnitude)
            self._magnitude_initialized = False

    def _ensure_magnitude_init(self) -> None:
        """Set DoRA magnitude to the row-wise norm of the pretrained base."""
        if self.r <= 0:
            return
        if not getattr(self, "_magnitude_initialized", False):
            self.lora_magnitude.data = torch.linalg.norm(
                self._T(self.weight), dim=1, keepdim=True
            )
            self._magnitude_initialized = True

    def _delta_weight(self) -> torch.Tensor:
        if self.r <= 0:
            return torch.zeros_like(self._T(self.weight))
        return (self.lora_B @ self.lora_A) * self.scaling

    def _combined_weight(self) -> torch.Tensor:
        combined = self._T(self.weight) + self._delta_weight()
        norm = torch.linalg.norm(combined, dim=1, keepdim=True).clamp_min(1e-8)
        return self.lora_magnitude * combined / norm

    def merge_to_base(self) -> None:
        if self.r <= 0:
            return
        combined = self._combined_weight()
        self.weight.data.copy_(combined.T if self.fan_in_fan_out else combined)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r <= 0:
            return F.linear(x, self._T(self.weight), self.bias)
        self._ensure_magnitude_init()
        base = F.linear(x, self._T(self.weight))
        after_a = F.linear(self.lora_dropout(x), self.lora_A)
        after_b = F.linear(after_a, self.lora_B)
        delta = after_b * self.scaling
        norm = torch.linalg.norm(
            self._T(self.weight) + self._delta_weight(),
            dim=1,
            keepdim=True,
        ).clamp_min(1e-8)
        factor = (self.lora_magnitude / norm).squeeze(-1)
        out = (base + delta) * factor
        if self.bias is not None:
            out = out + self.bias
        return out


class DoRAMergedLinear(nn.Linear, LoRALayer):
    """Merged q/k/v LoRA layer with DoRA magnitude/direction decomposition."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: list[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        assert out_features % len(enable_lora) == 0
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros(
                    (out_features // len(enable_lora) * sum(enable_lora), r)
                )
            )
            self.lora_magnitude = nn.Parameter(torch.ones(out_features, 1))
            self._magnitude_initialized = False
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def _T(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.T if self.fan_in_fan_out else weight

    def reset_parameters(self) -> None:
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
            nn.init.ones_(self.lora_magnitude)
            self._magnitude_initialized = False

    def _ensure_magnitude_init(self) -> None:
        if self.r <= 0 or not any(self.enable_lora):
            return
        if not getattr(self, "_magnitude_initialized", False):
            self.lora_magnitude.data = torch.linalg.norm(
                self._T(self.weight), dim=1, keepdim=True
            )
            self._magnitude_initialized = True

    def zero_pad(self, x: torch.Tensor) -> torch.Tensor:
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1,
            self.out_features // len(self.enable_lora) * sum(self.enable_lora),
        )
        return result.view((*x.shape[:-1], self.out_features))

    def _delta_weight(self) -> torch.Tensor:
        if self.r <= 0 or not any(self.enable_lora):
            return torch.zeros_like(self._T(self.weight))
        compact_delta = F.conv1d(
            self.lora_A.data.unsqueeze(0),
            self.lora_B.data.unsqueeze(-1),
            groups=sum(self.enable_lora),
        ).squeeze(0) * self.scaling
        delta = torch.zeros_like(self._T(self.weight))
        delta[self.lora_ind] = compact_delta
        return delta

    def _combined_weight(self) -> torch.Tensor:
        combined = self._T(self.weight) + self._delta_weight()
        norm = torch.linalg.norm(combined, dim=1, keepdim=True).clamp_min(1e-8)
        return self.lora_magnitude * combined / norm

    def merge_to_base(self) -> None:
        if self.r <= 0 or not any(self.enable_lora):
            return
        combined = self._combined_weight()
        self.weight.data.copy_(combined.T if self.fan_in_fan_out else combined)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self._T(self.weight))
        if self.r <= 0 or not any(self.enable_lora):
            return base + (self.bias if self.bias is not None else 0.0)
        self._ensure_magnitude_init()
        after_a = F.linear(self.lora_dropout(x), self.lora_A)
        after_b = F.conv1d(
            after_a.transpose(-2, -1),
            self.lora_B.unsqueeze(-1),
            groups=sum(self.enable_lora),
        ).transpose(-2, -1)
        delta = self.zero_pad(after_b) * self.scaling
        norm = torch.linalg.norm(
            self._T(self.weight) + self._delta_weight(),
            dim=1,
            keepdim=True,
        ).clamp_min(1e-8)
        factor = (self.lora_magnitude / norm).squeeze(-1)
        out = (base + delta) * factor
        if self.bias is not None:
            out = out + self.bias
        return out


def patch_lora_for_dora():
    """Replace FunASR LoRA layer classes with DoRA-compatible classes."""
    from funasr.models.lora import layers as lora_layers

    lora_layers.Linear = DoRALinear
    lora_layers.MergedLinear = DoRAMergedLinear
    return lora_layers


def update_lora_config(config_path: Path, lora_cfg: dict) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    for section in ("encoder_conf", "decoder_conf"):
        if section not in cfg:
            continue
        for key, value in lora_cfg.items():
            cfg[section][key] = value
    OmegaConf.save(cfg, config_path)


def merge_adapter_checkpoint(
    base_model_dir: Path,
    checkpoint_path: Path,
    model_out_dir: Path,
    lora_cfg: dict,
    dora: bool = False,
) -> Path:
    """Build an adapter-aware model, merge adapters into base weights, save clean model."""
    if dora:
        patch_lora_for_dora()

    from funasr import AutoModel

    if model_out_dir.exists():
        shutil.rmtree(model_out_dir)
    shutil.copytree(base_model_dir, model_out_dir)

    with tempfile.TemporaryDirectory(prefix="adapter_merge_") as td:
        temp_model_dir = Path(td) / "model"
        shutil.copytree(base_model_dir, temp_model_dir)
        update_lora_config(temp_model_dir / "config.yaml", lora_cfg)
        shutil.copy2(checkpoint_path, temp_model_dir / "model.pt")

        model = AutoModel(model=str(temp_model_dir), device="cpu", disable_update=True)
        torch_model = model.model
        if dora:
            for module in torch_model.modules():
                if hasattr(module, "merge_to_base"):
                    module.merge_to_base()
        else:
            torch_model.eval()

        clean_state = {
            key: value.clone()
            for key, value in torch_model.state_dict().items()
            if "lora_" not in key
        }
        torch.save({"state_dict": clean_state}, model_out_dir / "model.pt")

    with (model_out_dir / "training_source.txt").open("w", encoding="utf-8") as f:
        f.write(f"base_model={base_model_dir}\n")
        f.write(f"checkpoint={checkpoint_path}\n")
        f.write(f"adapter_lora_cfg={lora_cfg}\n")
        f.write(f"dora={dora}\n")
    print(f"Merged adapter model: {model_out_dir}", flush=True)
    return model_out_dir
