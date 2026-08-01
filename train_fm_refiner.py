import argparse
import glob
import math
import os
import random
from pathlib import d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None


# -----------------------------
# Utility
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mse(pred, target, atom_mask):
    """
    pred, target: [B, L, 4, 3]
    atom_mask:    [B, L, 4]

    실제 atom이 존재하는 위치에 대해서만 MSE 계산.
    """
    mask = atom_mask.unsqueeze(-1).float()  # [B, L, 4, 1]
    loss = ((pred - target) ** 2) * mask
    denom = mask.sum().clamp(min=1.0) * pred.shape[-1]
    return loss.sum() / denom


def masked_rmsd(x, y, atom_mask):
    """
    x, y:      [B, L, 4, 3]
    atom_mask: [B, L, 4]

    backbone atom 전체 RMSD.
    """
    diff2 = ((x - y) ** 2).sum(dim=-1)  # [B, L, 4]
    denom = atom_mask.sum().clamp(min=1.0)
    return torch.sqrt((diff2 * atom_mask).sum() / denom)


def masked_ca_rmsd(x, y, atom_mask, ca_index=1):
    """
    C-alpha 기준 RMSD.
    backbone local index:
    0=N, 1=CA, 2=C, 3=O
    """
    diff2 = ((x[:, :, ca_index] - y[:, :, ca_index]) ** 2).sum(dim=-1)  # [B, L]
    mask = atom_mask[:, :, ca_index].float()
    denom = mask.sum().clamp(min=1.0)
    return torch.sqrt((diff2 * mask).sum() / denom)

def kabsch_align_mobile_to_target(mobile, target, atom_mask, ca_index=1):
    """
    mobile:    [B, L, 4, 3]
    target:    [B, L, 4, 3]
    atom_mask: [B, L, 4]

    CA 기준으로 mobile을 target에 Kabsch alignment.
    return: aligned_mobile [B, L, 4, 3]
    """
    B = mobile.shape[0]
    aligned = []

    for b in range(B):
        ca_mask = atom_mask[b, :, ca_index].bool()

        if ca_mask.sum() < 3:
            aligned.append(mobile[b])
            continue

        p = mobile[b, :, ca_index, :][ca_mask]  # mobile CA
        q = target[b, :, ca_index, :][ca_mask]  # target CA

        p_mean = p.mean(dim=0, keepdim=True)
        q_mean = q.mean(dim=0, keepdim=True)

        p_centered = p - p_mean
        q_centered = q - q_mean

        H = p_centered.transpose(0, 1) @ q_centered

        try:
            U, S, Vh = torch.linalg.svd(H)
        except RuntimeError:
            aligned.append(mobile[b])
            continue

        R = U @ Vh

        if torch.det(R) < 0:
            Vh[-1, :] *= -1
            R = U @ Vh

        mobile_aligned = (
            mobile[b] - p_mean.view(1, 1, 3)
        ) @ R + q_mean.view(1, 1, 3)

        aligned.append(mobile_aligned)

    return torch.stack(aligned, dim=0)

def masked_aligned_rmsd(x, y, atom_mask):
    """
    Kabsch alignment 후 backbone RMSD.
    """
    x_aligned = kabsch_align_mobile_to_target(
        mobile=x,
        target=y,
        atom_mask=atom_mask,
    )
    return masked_rmsd(x_aligned, y, atom_mask)


def masked_aligned_ca_rmsd(x, y, atom_mask):
    """
    Kabsch alignment 후 CA RMSD.
    """
    x_aligned = kabsch_align_mobile_to_target(
        mobile=x,
        target=y,
        atom_mask=atom_mask,
    )
    return masked_ca_rmsd(x_aligned, y, atom_mask)

def sinusoidal_time_embedding(t, dim):
    """
    t: [B] 또는 [B, 1]
    return: [B, dim]

    Flow Matching time t를 Transformer 입력에 넣기 위한 sinusoidal embedding.
    """
    if t.ndim == 2:
        t = t.squeeze(-1)

    half = dim // 2
    device = t.device

    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device).float() / max(half - 1, 1)
    )
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb

def random_rotation_matrix(device):
    """
    Generate a random 3D rotation matrix.
    """
    q = torch.randn(4, device=device)
    q = q / q.norm().clamp(min=1e-8)

    w, x, y, z = q

    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
    ])

    return R


def rotate_coords(x, R):
    """
    x: [..., 3]
    R: [3, 3]

    row-vector convention.
    """
    return x @ R.T

# -----------------------------
# Dataset
# -----------------------------

class ResidualDataset(Dataset):
    """
    build_residual_dataset.py가 만든 .pt 파일들을 읽는 Dataset.

    각 .pt 파일에 들어있는 값:
    struct_ids : [L]
    aatype     : [L]
    x_tilde    : [L, 4, 3]
    residual   : [L, 4, 3]
    atom_mask  : [L, 4]
    res_mask   : [L]
    """

    def __init__(self, data_dir, max_length=None):
        self.paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No .pt files found in: {data_dir}")

        self.max_length = max_length

        if max_length is not None:
            filtered = []
            for p in self.paths:
                try:
                    d = torch.load(p, map_location="cpu")
                    L = int(d["seq_length"]) if "seq_length" in d else int(d["x_tilde"].shape[0])
                    if L <= max_length:
                        filtered.append(p)
                except Exception:
                    pass
            self.paths = filtered

        if len(self.paths) == 0:
            raise RuntimeError("No valid samples after max_length filtering.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        d = torch.load(self.paths[idx], map_location="cpu")

        struct_ids = d["struct_ids"].long()       # [L]
        aatype = d["aatype"].long()               # [L]
        x_tilde = d["x_tilde"].float()            # [L, 4, 3]
        residual = d["residual"].float()          # [L, 4, 3]
        atom_mask = d["atom_mask"].float()        # [L, 4]
        res_mask = d["res_mask"].float()          # [L]

        if struct_ids.min() < 0 or struct_ids.max() >= 8192:
            raise ValueError(
                f"Invalid struct token in {self.paths[idx]}: "
                f"min={int(struct_ids.min())}, "
                f"max={int(struct_ids.max())}, "
                f"valid range=[0, 8191]"
            )

        if aatype.min() < 0 or aatype.max() >= 32:
            raise ValueError(
                f"Invalid amino-acid token in {self.paths[idx]}: "
                f"min={int(aatype.min())}, "
                f"max={int(aatype.max())}, "
                f"valid range=[0, 31]"
            )

        sample = {
            "path": self.paths[idx],
            "struct_ids": struct_ids,
            "aatype": aatype,
            "x_tilde": x_tilde,
            "residual": residual,
            "atom_mask": atom_mask,
            "res_mask": res_mask,
            "length": torch.tensor(struct_ids.shape[0], dtype=torch.long),
        }
        return sample


def collate_residual_samples(samples):
    """
    길이가 다른 단백질들을 batch로 묶기 위해 padding.

    반환 shape:
    struct_ids: [B, Lmax]
    aatype:     [B, Lmax]
    x_tilde:    [B, Lmax, 4, 3]
    residual:   [B, Lmax, 4, 3]
    atom_mask:  [B, Lmax, 4]
    res_mask:   [B, Lmax]
    """
    B = len(samples)
    Lmax = max(int(s["length"]) for s in samples)

    struct_ids = torch.zeros(B, Lmax, dtype=torch.long)
    aatype = torch.zeros(B, Lmax, dtype=torch.long)
    x_tilde = torch.zeros(B, Lmax, 4, 3, dtype=torch.float32)
    residual = torch.zeros(B, Lmax, 4, 3, dtype=torch.float32)
    atom_mask = torch.zeros(B, Lmax, 4, dtype=torch.float32)
    res_mask = torch.zeros(B, Lmax, dtype=torch.float32)
    lengths = torch.zeros(B, dtype=torch.long)
    paths = []

    for i, s in enumerate(samples):
        L = int(s["length"])
        lengths[i] = L
        paths.append(s["path"])

        struct_ids[i, :L] = s["struct_ids"]
        aatype[i, :L] = s["aatype"]
        x_tilde[i, :L] = s["x_tilde"]
        residual[i, :L] = s["residual"]
        atom_mask[i, :L] = s["atom_mask"]
        res_mask[i, :L] = s["res_mask"]

    return {
        "paths": paths,
        "struct_ids": struct_ids,
        "aatype": aatype,
        "x_tilde": x_tilde,
        "residual": residual,
        "atom_mask": atom_mask,
        "res_mask": res_mask,
        "lengths": lengths,
    }

def masked_bond_length_loss(x_refined, x_gt, atom_mask):
    """
    x_refined, x_gt: [B, L, 4, 3]
    atom_mask:       [B, L, 4]

    backbone local index:
    0=N, 1=CA, 2=C, 3=O

    예측 구조의 bond length가 GT bond length와 비슷하도록 제한.
    """
    bond_pairs = [
        (0, 1),  # N-CA
        (1, 2),  # CA-C
        (2, 3),  # C-O
    ]

    losses = []

    for i, j in bond_pairs:
        pred_len = torch.norm(x_refined[:, :, i] - x_refined[:, :, j], dim=-1)
        true_len = torch.norm(x_gt[:, :, i] - x_gt[:, :, j], dim=-1)

        mask = atom_mask[:, :, i] * atom_mask[:, :, j]
        diff2 = ((pred_len - true_len) ** 2) * mask
        denom = mask.sum().clamp(min=1.0)

        losses.append(diff2.sum() / denom)

    return sum(losses) / len(losses)

# -----------------------------
# Model
# -----------------------------

class FMResidualRefiner(nn.Module):
    """
    Flow Matching 기반 residual refinement model.

    입력:
    r_t        : 현재 residual 상태 [B, L, 4, 3]
    t          : FM time [B]
    struct_ids : DPLM-2 structure token [B, L]
    aatype     : amino acid token [B, L]
    x_tilde    : DPLM-2 detokenize coarse 좌표 [B, L, 4, 3]
    res_mask   : residue mask [B, L]

    출력:
    v_pred     : residual space에서의 이동 방향 [B, L, 4, 3]
    """

    def __init__(
        self,
        struct_vocab_size=8192,
        aa_vocab_size=32,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        time_dim=128,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        self.struct_emb = nn.Embedding(struct_vocab_size, hidden_dim)
        self.aa_emb = nn.Embedding(aa_vocab_size, hidden_dim)

        # x_tilde와 r_t를 residue별 feature로 flatten.
        # x_tilde: 4*3 = 12
        # r_t:     4*3 = 12
        coord_in_dim = 24

        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * 3),
        )

        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, r_t, t, struct_ids, aatype, x_tilde, res_mask):
        B, L = struct_ids.shape

        struct_feat = self.struct_emb(struct_ids)  # [B, L, H]
        aa_feat = self.aa_emb(aatype)              # [B, L, H]

        coord_feat = torch.cat(
            [
                x_tilde.reshape(B, L, -1),
                r_t.reshape(B, L, -1),
            ],
            dim=-1,
        )  # [B, L, 24]

        coord_feat = self.coord_mlp(coord_feat)    # [B, L, H]

        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_feat = self.time_mlp(t_emb).unsqueeze(1)  # [B, 1, H]

        h = struct_feat + aa_feat + coord_feat + t_feat

        # Transformer padding mask: True인 위치가 padding
        padding_mask = res_mask <= 0.5  # [B, L]

        h = self.transformer(h, src_key_padding_mask=padding_mask)

        v = self.out(h).reshape(B, L, 4, 3)

        # padding 위치는 0으로
        v = v * res_mask[:, :, None, None].float()
        return v

def center_coordinates(x, atom_mask, ca_index=1):
    """
    x:         [B, L, 4, 3]
    atom_mask: [B, L, 4]
    """
    ca = x[:, :, ca_index]                       # [B, L, 3]
    ca_mask = atom_mask[:, :, ca_index].float() # [B, L]

    center = (
        (ca * ca_mask.unsqueeze(-1)).sum(dim=1)
        / ca_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    )  # [B, 3]

    centered = x - center[:, None, None, :]
    centered = centered * atom_mask.unsqueeze(-1)

    return centered

# -----------------------------
# Flow Matching
# -----------------------------

def fm_training_step(model, batch, device, residual_scale=1.0, 
                     use_rotation_aug=False,zero_loss_weight=0.1, bond_loss_weight=0.01, t_override=None,):
    """
    Deterministic zero-source Flow Matching 학습 step.

    r_1 = true residual
    r_0 = zero residual
    r_t = (1 - t) r_0 + t r_1
    u_t = r_1 - r_0
    
    zero source이므로:
        r_t = t * r_1
        u_t = r_1

    model(r_t, t | condition) -> u_t 예측
    """
    struct_ids = batch["struct_ids"].to(device)
    aatype = batch["aatype"].to(device)
    x_tilde = batch["x_tilde"].to(device)
    residual = batch["residual"].to(device)
    atom_mask = batch["atom_mask"].to(device)
    res_mask = batch["res_mask"].to(device)

    if use_rotation_aug:
        R = random_rotation_matrix(device)
        x_tilde = rotate_coords(x_tilde, R)
        residual = rotate_coords(residual, R)

    # invalid atom 위치가 모델 입력에 들어가지 않도록 처리
    atom_mask_4d = atom_mask.unsqueeze(-1)
    x_tilde = x_tilde * atom_mask_4d
    residual = residual * atom_mask_4d

    # residual scale 정규화
    r_1 = residual / residual_scale
    r_0 = torch.zeros_like(r_1)

    B = r_1.shape[0]

    if t_override is None:
        t = torch.rand(B, device=device)  # [B]
    else:
         t = torch.full((B,), float(t_override), device=device)  # [B]
    t_view = t.view(B, 1, 1, 1)

    r_t = (1.0 - t_view) * r_0 + t_view * r_1
    u_t = r_1 - r_0

    x_tilde_cond = center_coordinates(
        x_tilde,
        atom_mask,
    )
    
    v_pred = model(
        r_t=r_t,
        t=t,
        struct_ids=struct_ids,
        aatype=aatype,
        x_tilde=x_tilde_cond,
        res_mask=res_mask,
    )

    fm_loss = masked_mse(v_pred, u_t, atom_mask)

    r_hat_1 = r_t + (1.0 - t_view) * v_pred
    endpoint_loss = masked_mse(r_hat_1, r_1, atom_mask)

    x_hat_1 = x_tilde + r_hat_1 * residual_scale
    x_gt = x_tilde + residual

    bond_loss = masked_bond_length_loss(
        x_refined=x_hat_1,
        x_gt=x_gt,
        atom_mask=atom_mask,
    )

    # inference 시작점인 t=0, r=0에서 바로 true residual 방향을 예측하도록 강제
    r_zero = torch.zeros_like(r_1)
    t_zero = torch.zeros(B, device=device)

    v_zero = model(
        r_t=r_zero,
        t=t_zero,
        struct_ids=struct_ids,
        aatype=aatype,
        x_tilde=x_tilde_cond,
        res_mask=res_mask,
    )

    zero_loss = masked_mse(v_zero, r_1, atom_mask)

    weighted_zero_loss = (zero_loss_weight * zero_loss)

    loss = fm_loss + endpoint_loss + weighted_zero_loss + bond_loss_weight * bond_loss

    return {
        "loss": loss,
        "fm_loss": fm_loss,
        "endpoint_loss": endpoint_loss,
        "zero_loss": zero_loss,
        "weighted_zero_loss": weighted_zero_loss,
        "bond_loss": bond_loss,
        "weighted_bond_loss": bond_loss_weight * bond_loss
        }


@torch.no_grad()
def sample_residual_euler(
    model,
    batch,
    device,
    num_steps=20,
    residual_scale=1.0,
    max_residual_angstrom=None,
    refine_strength=1.0,
):
    """
    Deterministic residual refinement.

    r(0) = 0
    dr/dt = v_theta(r_t, t | condition)
    Euler integration으로 t=0 -> 1 이동.
    """
    model.eval()

    struct_ids = batch["struct_ids"].to(device)
    aatype = batch["aatype"].to(device)
    x_tilde = batch["x_tilde"].to(device)
    atom_mask = batch["atom_mask"].to(device)
    res_mask = batch["res_mask"].to(device)

    r = torch.zeros_like(x_tilde)
    dt = 1.0 / float(num_steps)

    B = x_tilde.shape[0]

    x_tilde_cond = center_coordinates(
            x_tilde,
            atom_mask,
        )
    
    for step in range(num_steps):
        t_value = torch.full((B,), step / float(num_steps), device=device)
        
        v = model(
            r_t=r,
            t=t_value,
            struct_ids=struct_ids,
            aatype=aatype,
            x_tilde=x_tilde_cond,
            res_mask=res_mask,
        )
        r = r + dt * v
        #r = r.clamp(min=-5.0 / residual_scale, max=5.0 / residual_scale)
        r = r * atom_mask.unsqueeze(-1)

    r_pred = r * residual_scale
    r_pred = refine_strength * r_pred

    if max_residual_angstrom is not None:
        r_pred = r_pred.clamp(
            min=-max_residual_angstrom,
            max=max_residual_angstrom,
        )

    x_refined = x_tilde + r_pred

    return x_refined, r_pred


@torch.no_grad()
def evaluate(model, loader, device, num_steps=20, residual_scale=1.0, 
             max_residual_angstrom=None, zero_loss_weight=0.1, bond_loss_weight=0.0, refine_strength=1.0):
    model.eval()

    total_loss = 0.0
    total_fm_loss = 0.0
    total_endpoint_loss = 0.0
    total_zero_loss = 0.0
    total_weighted_zero_loss = 0.0

    total_before_rmsd = 0.0
    total_after_rmsd = 0.0
    total_before_ca = 0.0
    total_after_ca = 0.0

    total_before_aligned_rmsd = 0.0
    total_after_aligned_rmsd = 0.0
    total_before_aligned_ca = 0.0
    total_after_aligned_ca = 0.0

    n = 0

    for batch in loader:
        loss_dict = fm_training_step(
            model=model,
            batch=batch,
            device=device,
            residual_scale=residual_scale,
            use_rotation_aug=False,
            zero_loss_weight=zero_loss_weight,
            bond_loss_weight=bond_loss_weight,
            t_override=0.5, #고정된 시점으로 검증
        )

        loss = loss_dict["loss"]

        x_gt = batch["x_tilde"].to(device) + batch["residual"].to(device)
        x_tilde = batch["x_tilde"].to(device)
        atom_mask = batch["atom_mask"].to(device)

        x_refined, _ = sample_residual_euler(
            model=model,
            batch=batch,
            device=device,
            num_steps=num_steps,
            residual_scale=residual_scale,
            max_residual_angstrom=max_residual_angstrom,
            refine_strength=refine_strength,
        )

        before_rmsd = masked_rmsd(x_tilde, x_gt, atom_mask)
        after_rmsd = masked_rmsd(x_refined, x_gt, atom_mask)

        before_ca = masked_ca_rmsd(x_tilde, x_gt, atom_mask)
        after_ca = masked_ca_rmsd(x_refined, x_gt, atom_mask)

        before_aligned_rmsd = masked_aligned_rmsd(x_tilde, x_gt, atom_mask)
        after_aligned_rmsd = masked_aligned_rmsd(x_refined, x_gt, atom_mask)

        before_aligned_ca = masked_aligned_ca_rmsd(x_tilde, x_gt, atom_mask)
        after_aligned_ca = masked_aligned_ca_rmsd(x_refined, x_gt, atom_mask)

        total_loss += float(loss.item())
        total_fm_loss += float(loss_dict["fm_loss"].item())
        total_endpoint_loss += float(loss_dict["endpoint_loss"].item())
        total_zero_loss += float(loss_dict["zero_loss"].item())
        total_weighted_zero_loss += float(loss_dict["weighted_zero_loss"].item())

        total_before_rmsd += float(before_rmsd.item())
        total_after_rmsd += float(after_rmsd.item())
        total_before_ca += float(before_ca.item())
        total_after_ca += float(after_ca.item())

        total_before_aligned_rmsd += float(before_aligned_rmsd.item())
        total_after_aligned_rmsd += float(after_aligned_rmsd.item())
        total_before_aligned_ca += float(before_aligned_ca.item())
        total_after_aligned_ca += float(after_aligned_ca.item())

        n += 1

    n = max(n, 1)
    return {
        "loss": total_loss / n,
        "fm_loss": total_fm_loss / n,
        "endpoint_loss": total_endpoint_loss / n,
        "zero_loss": total_zero_loss / n,
        "weighted_zero_loss": total_weighted_zero_loss / n,

        "before_rmsd": total_before_rmsd / n,
        "after_rmsd": total_after_rmsd / n,
        "before_ca_rmsd": total_before_ca / n,
        "after_ca_rmsd": total_after_ca / n,

        "before_aligned_rmsd": total_before_aligned_rmsd / n,
        "after_aligned_rmsd": total_after_aligned_rmsd / n,
        "before_aligned_ca_rmsd": total_before_aligned_ca / n,
        "after_aligned_ca_rmsd": total_after_aligned_ca / n,
    }


def estimate_residual_scale(
    dataset,
    num_samples=2000,
    seed=42,
):
    num_samples = min(len(dataset), num_samples)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:num_samples].tolist()

    values = []

    for idx in indices:
        d = dataset[idx]

        residual = d["residual"].float()
        atom_mask = d["atom_mask"].bool().unsqueeze(-1)
        valid_mask = atom_mask.expand_as(residual)

        valid = residual[valid_mask]

        if valid.numel() > 0:
            values.append(valid.reshape(-1))

    if not values:
        return 1.0

    values = torch.cat(values)
    std = values.std(unbiased=True).item()

    if not math.isfinite(std) or std < 1e-6:
        return 1.0

    return std


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--struct_vocab_size", type=int, default=8192)
    parser.add_argument("--aa_vocab_size", type=int, default=32)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--max_length", type=int, default=None)

    parser.add_argument("--num_sample_steps", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dplm2-fm-refiner")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)

    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    parser.add_argument(
        "--residual_scale",
        type=float,
        default=None,
        help="None이면 train dataset에서 자동 추정",
    )

    parser.add_argument(
        "--max_residual_angstrom",
        type=float,
        default=None,
        help="Optional clamp value for final predicted residual in Angstrom. "
            "If None, no final residual clamp is applied.",
    )

    parser.add_argument(
        "--use_rotation_aug",
        action="store_true",
        help="Apply random rotation augmentation to x_tilde and residual during training.",
    )

    parser.add_argument(
        "--zero_loss_weight",
        type=float,
        default=0.1,
        help=(
            "Weight for the t=0, r=0 endpoint velocity loss. "
            "Set 0 to disable."
            ),
    )

    parser.add_argument(
        "--refine_strength",
        type=float,
        default=1.0,
        help="Scale factor for predicted residual before adding to x_tilde.",
    )

    parser.add_argument(
        "--bond_loss_weight",
        type=float,
        default=0.01,
        help="Weight for backbone bond length preservation loss.",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb가 설치되어 있지 않습니다. 먼저 `pip install wandb`를 실행하세요.")

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warning] cuda unavailable. use cpu.")
        device = "cpu"

    print("[info] loading dataset...")
    dataset = ResidualDataset(args.data_dir, max_length=args.max_length)

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size

    if train_size <= 0:
        raise RuntimeError("Dataset too small. Need at least 2 samples.")

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    if args.residual_scale is None:
        residual_scale = estimate_residual_scale(
            train_dataset,
            num_samples=len(train_dataset),
            seed=args.seed,
        )
    else:
        residual_scale = float(args.residual_scale)

    print(f"[info] total samples: {len(dataset)}")
    print(f"[info] train samples: {len(train_dataset)}")
    print(f"[info] val samples: {len(val_dataset)}")
    print(f"[info] residual_scale: {residual_scale:.6f}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_residual_samples,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_residual_samples,
        pin_memory=(device == "cuda"),
    )

    print("[info] building model...")
    model = FMResidualRefiner(
        struct_vocab_size=args.struct_vocab_size,
        aa_vocab_size=args.aa_vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    def lr_lambda(epoch_idx):
        """
        epoch_idx는 0부터 시작.
        warmup 이후 cosine decay.
        """
        if epoch_idx < args.warmup_epochs:
            return float(epoch_idx + 1) / float(max(1, args.warmup_epochs))

        progress = float(epoch_idx - args.warmup_epochs) / float(
            max(1, args.num_epochs - args.warmup_epochs)
        )

        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine


    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_lambda,)

    best_val_after_ca = float("inf")
    best_path = os.path.join(args.save_dir, "best.pt")
    last_path = os.path.join(args.save_dir, "last.pt")

    os.makedirs(args.save_dir, exist_ok=True)

    config = vars(args)
    config["residual_scale"] = residual_scale

    if args.use_wandb:
        wandb.config.update({"residual_scale": residual_scale}, allow_val_change=True)

    print("[info] start training")

    for epoch in range(1, args.num_epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_fm_loss_sum = 0.0
        train_endpoint_loss_sum = 0.0
        train_zero_loss_sum = 0.0
        train_weighted_zero_loss_sum = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.num_epochs}")

        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            loss_dict = fm_training_step(
                model=model,
                batch=batch,
                device=device,
                residual_scale=residual_scale,
                use_rotation_aug=args.use_rotation_aug,
                zero_loss_weight=args.zero_loss_weight,
                bond_loss_weight=args.bond_loss_weight,
                t_override=None,
            )

            loss = loss_dict["loss"]
            loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            train_loss_sum += float(loss_dict["loss"].item())
            train_fm_loss_sum += float(loss_dict["fm_loss"].item())
            train_endpoint_loss_sum += float(loss_dict["endpoint_loss"].item())
            train_zero_loss_sum += float(loss_dict["zero_loss"].item())
            train_weighted_zero_loss_sum += float(loss_dict["weighted_zero_loss"].item())

            train_steps += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if args.use_wandb:
                global_step = (epoch - 1) * len(train_loader) + train_steps
                wandb.log(
                    {
                        "train/step_loss": float(loss.item()),
                        "train/fm_loss": float(loss_dict["fm_loss"].item()),
                        "train/endpoint_loss": float(loss_dict["endpoint_loss"].item()),                     
                        "train/zero_loss": float(loss_dict["zero_loss"].item()),
                        "train/weighted_zero_loss": float(loss_dict["weighted_zero_loss"].item()), 
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        denom = max(train_steps, 1)

        train_loss = train_loss_sum / denom
        train_fm_loss = train_fm_loss_sum / denom
        train_endpoint_loss = (train_endpoint_loss_sum / denom)
        train_zero_loss = (train_zero_loss_sum / denom)
        train_weighted_zero_loss = (train_weighted_zero_loss_sum / denom)

        metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            num_steps=args.num_sample_steps,
            residual_scale=residual_scale,
            max_residual_angstrom=args.max_residual_angstrom,
            zero_loss_weight=args.zero_loss_weight,
            bond_loss_weight=args.bond_loss_weight,
            refine_strength=args.refine_strength,
        )

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={metrics['loss']:.6f} "
            f"RMSD {metrics['before_rmsd']:.4f}->{metrics['after_rmsd']:.4f} "
            f"CA {metrics['before_ca_rmsd']:.4f}->{metrics['after_ca_rmsd']:.4f}"
            f"aligned_RMSD {metrics['before_aligned_rmsd']:.4f}->{metrics['after_aligned_rmsd']:.4f} "
            f"aligned_CA {metrics['before_aligned_ca_rmsd']:.4f}->{metrics['after_aligned_ca_rmsd']:.4f}"
        )

        if args.use_wandb:
            wandb.log(
                {
                    "train/epoch_loss": train_loss,
                    "train/epoch_fm_loss": train_fm_loss,
                    "train/epoch_endpoint_loss": train_endpoint_loss,
                    "train/epoch_zero_loss": train_zero_loss,
                    "train/epoch_weighted_zero_loss": train_weighted_zero_loss,
                    "val/loss": metrics["loss"],
                    "val/before_rmsd": metrics["before_rmsd"],
                    "val/after_rmsd": metrics["after_rmsd"],
                    "val/before_ca_rmsd": metrics["before_ca_rmsd"],
                    "val/after_ca_rmsd": metrics["after_ca_rmsd"],
                    "val/rmsd_improvement": metrics["before_rmsd"] - metrics["after_rmsd"],
                    "val/ca_rmsd_improvement": metrics["before_ca_rmsd"] - metrics["after_ca_rmsd"],
                    "val/before_aligned_rmsd": metrics["before_aligned_rmsd"],
                    "val/after_aligned_rmsd": metrics["after_aligned_rmsd"],
                    "val/before_aligned_ca_rmsd": metrics["before_aligned_ca_rmsd"],
                    "val/after_aligned_ca_rmsd": metrics["after_aligned_ca_rmsd"],
                    "val/aligned_rmsd_improvement": metrics["before_aligned_rmsd"] - metrics["after_aligned_rmsd"],
                    "val/aligned_ca_rmsd_improvement": metrics["before_aligned_ca_rmsd"] - metrics["after_aligned_ca_rmsd"],
                    "epoch": epoch,
                },
                step=epoch * len(train_loader),
            )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
        }

        torch.save(ckpt, last_path)

        # CA-RMSD after가 가장 낮은 모델 저장
        if metrics["after_ca_rmsd"] < best_val_after_ca:
            best_val_after_ca = metrics["after_ca_rmsd"]
            torch.save(ckpt, best_path)
            print(f"[info] saved best checkpoint: {best_path}")

            if args.use_wandb:
                wandb.run.summary["best_val_after_ca_rmsd"] = best_val_after_ca
                wandb.run.summary["best_epoch"] = epoch

        scheduler.step()

    print("[done]")
    print(f"best checkpoint: {best_path}")
    print(f"last checkpoint: {last_path}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()