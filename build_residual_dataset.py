import argparse
import os
import pickle
import warnings
from glob import glob
from pathlib import Path

import torch
from tqdm.auto import tqdm

from byprot.models.utils import get_struct_tokenizer
from byprot.utils.protein import residue_constants as rc

warnings.filterwarnings("ignore")


def get_backbone_atom_indices():
    """
    atom37 기준 backbone atom index를 가져옵니다.

    AlphaFold atom37에서 보통:
    N  = 0
    CA = 1
    C  = 2
    O  = 4

    CB가 3이므로, 단순히 :4를 쓰면 O 대신 CB가 들어갈 수 있습니다.
    따라서 반드시 atom_order를 사용합니다.
    """
    try:
        return [
            rc.atom_order["N"],
            rc.atom_order["CA"],
            rc.atom_order["C"],
            rc.atom_order["O"],
        ]
    except Exception:
        return [0, 1, 2, 4]


def safe_stem(path):
    return Path(path).stem.replace("/", "_").replace(" ", "_")


def load_pkl(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return dict(obj)


def to_tensor(x, device, dtype=None):
    t = torch.as_tensor(x, device=device)
    if dtype is not None:
        t = t.to(dtype)
    return t


def align_mobile_to_target_by_ca(mobile, target, atom_mask, ca_local_index=1):
    """
    mobile: [L, 4, 3]  보정할 좌표, 보통 x_tilde
    target: [L, 4, 3]  기준 좌표, 보통 x_gt
    atom_mask: [L, 4]

    CA 기준으로 mobile을 target에 Kabsch alignment합니다.
    """
    ca_mask = atom_mask[:, ca_local_index].bool()

    if ca_mask.sum() < 3:
        return mobile

    p = mobile[:, ca_local_index, :][ca_mask]
    q = target[:, ca_local_index, :][ca_mask]

    p_mean = p.mean(dim=0, keepdim=True)
    q_mean = q.mean(dim=0, keepdim=True)

    p_centered = p - p_mean
    q_centered = q - q_mean

    h = p_centered.transpose(0, 1) @ q_centered

    try:
        u, s, vh = torch.linalg.svd(h)
    except RuntimeError:
        return mobile

    # row-vector convention: p @ r ≈ q
    r = u @ vh

    if torch.det(r) < 0:
        vh[-1, :] *= -1
        r = u @ vh

    mobile_aligned = (mobile - p_mean.view(1, 1, 3)) @ r + q_mean.view(1, 1, 3)

    return mobile_aligned


def make_batch_from_pkl(obj, device):
    """
    cameo2022/preprocessed/*.pkl 파일을 tokenizer 입력 batch로 바꿉니다.

    pkl key 예시:
    atom_positions : [L, 37, 3]
    atom_mask      : [L, 37]
    aatype         : [L]
    bb_mask        : [L]
    residue_index  : [L]
    """
    if "atom_positions" not in obj:
        raise KeyError("pkl에 atom_positions가 없습니다.")

    atom_positions = to_tensor(obj["atom_positions"], device, torch.float32)
    atom_mask = to_tensor(obj["atom_mask"], device, torch.float32)

    # NaN/Inf 방지
    atom_positions = torch.nan_to_num(atom_positions, nan=0.0, posinf=0.0, neginf=0.0)
    atom_mask = torch.nan_to_num(atom_mask, nan=0.0, posinf=0.0, neginf=0.0)

    length = atom_positions.shape[0]

    if "bb_mask" in obj:
        res_mask = to_tensor(obj["bb_mask"], device, torch.float32)
    else:
        bb_idx = get_backbone_atom_indices()
        res_mask = atom_mask[:, bb_idx].prod(dim=-1).float()

    if "aatype" in obj:
        aatype = to_tensor(obj["aatype"], device, torch.long)
    else:
        aatype = torch.zeros(length, dtype=torch.long, device=device)

    if "residue_index" in obj:
        residue_index = to_tensor(obj["residue_index"], device, torch.long)
    else:
        residue_index = torch.arange(length, dtype=torch.long, device=device)

    batch = {
        # tokenizer.tokenize()가 기대하는 입력 이름
        "all_atom_positions": atom_positions.unsqueeze(0),  # [1, L, 37, 3]
        "all_atom_mask": atom_mask.unsqueeze(0),            # [1, L, 37]
        "res_mask": res_mask.unsqueeze(0),                  # [1, L]
        "seq_length": torch.tensor([length], device=device, dtype=torch.long),
        "aatype": aatype.unsqueeze(0),                      # [1, L]
        "residue_index": residue_index.unsqueeze(0),        # [1, L]
    }

    return batch


def extract_backbone_and_residual(batch, detok_out, align=True):
    """
    x_gt, x_tilde, residual을 backbone atom 기준으로 계산합니다.

    반환 shape:
    x_gt     : [L, 4, 3]
    x_tilde  : [L, 4, 3]
    residual : [L, 4, 3]
    atom_mask: [L, 4]
    """
    bb_idx = get_backbone_atom_indices()

    x_gt_atom37 = batch["all_atom_positions"]       # [1, L, 37, 3]
    gt_atom37_mask = batch["all_atom_mask"]         # [1, L, 37]

    x_tilde_atom37 = detok_out["atom37_positions"]  # [1, L, 37, 3]

    if "atom37_mask" in detok_out:
        pred_atom37_mask = detok_out["atom37_mask"]
    else:
        pred_atom37_mask = torch.ones_like(gt_atom37_mask)

    x_gt = x_gt_atom37[:, :, bb_idx, :]             # [1, L, 4, 3]
    x_tilde = x_tilde_atom37[:, :, bb_idx, :]       # [1, L, 4, 3]

    gt_mask = gt_atom37_mask[:, :, bb_idx]          # [1, L, 4]
    pred_mask = pred_atom37_mask[:, :, bb_idx]      # [1, L, 4]

    atom_mask = gt_mask.float() * pred_mask.float()

    if "res_mask" in batch:
        atom_mask = atom_mask * batch["res_mask"].float().unsqueeze(-1)

    # batch dim 제거
    x_gt = x_gt[0]
    x_tilde = x_tilde[0]
    atom_mask = atom_mask[0]

    if align:
        x_tilde = align_mobile_to_target_by_ca(
            mobile=x_tilde,
            target=x_gt,
            atom_mask=atom_mask,
            ca_local_index=1,
        )

    residual = x_gt - x_tilde

    return x_gt, x_tilde, residual, atom_mask


def build_one_sample(
    pkl_path,
    struct_tokenizer,
    device,
    max_length=None,
    align=True,
    debug=False,
):
    obj = load_pkl(pkl_path)
    batch = make_batch_from_pkl(obj, device)

    seq_len = int(batch["seq_length"][0].item())

    if max_length is not None and seq_len > max_length:
        return None, f"skip_length_{seq_len}"

    with torch.no_grad():
        struct_ids = struct_tokenizer.tokenize(
            batch["all_atom_positions"],
            batch["res_mask"],
            batch["seq_length"],
        )

        detok_out = struct_tokenizer.detokenize(
            struct_ids,
            res_mask=batch["res_mask"],
        )

        x_gt, x_tilde, residual, atom_mask = extract_backbone_and_residual(
            batch=batch,
            detok_out=detok_out,
            align=align,
        )

    if debug:
        print("pkl_path:", pkl_path)
        print("seq_len:", seq_len)
        print("batch all_atom_positions:", batch["all_atom_positions"].shape)
        print("batch res_mask:", batch["res_mask"].shape)
        print("struct_ids:", struct_ids.shape, struct_ids.dtype)
        print("detok_out keys:", list(detok_out.keys()))
        print("x_gt:", x_gt.shape)
        print("x_tilde:", x_tilde.shape)
        print("residual:", residual.shape)
        print("atom_mask:", atom_mask.shape)
        print("res_mask:", batch["res_mask"][0].shape)

        before = masked_rmsd(x_tilde, x_gt, atom_mask)
        print("initial backbone RMSD after alignment:", float(before))

    sample = {
        "source_path": pkl_path,
        "seq_length": seq_len,
        "struct_ids": struct_ids[0].detach().cpu(),           # [L]
        "aatype": batch["aatype"][0].detach().cpu(),          # [L]
        "residue_index": batch["residue_index"][0].detach().cpu(),
        "x_gt": x_gt.detach().cpu(),                          # [L, 4, 3]
        "x_tilde": x_tilde.detach().cpu(),                    # [L, 4, 3]
        "residual": residual.detach().cpu(),                  # [L, 4, 3]
        "atom_mask": atom_mask.detach().cpu(),                # [L, 4]
        "res_mask": batch["res_mask"][0].detach().cpu(),      # [L]
    }

    return sample, "ok"


def masked_rmsd(x, y, mask):
    """
    x, y: [L, A, 3]
    mask: [L, A]
    """
    diff2 = ((x - y) ** 2).sum(dim=-1)
    denom = mask.sum().clamp(min=1.0)
    return torch.sqrt((diff2 * mask).sum() / denom)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_pkl_dir",
        type=str,
        required=True,
        help="preprocessed .pkl 파일들이 있는 폴더",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="residual dataset .pt 파일을 저장할 폴더",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pkl",
        help="검색할 파일 패턴. 기본값: *.pkl",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="이 길이보다 긴 단백질은 skip",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="처리할 최대 pkl 개수. None이면 전체 처리",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda 또는 cpu",
    )
    parser.add_argument(
        "--no_align",
        action="store_true",
        help="사용하면 x_tilde를 x_gt에 Kabsch alignment하지 않음",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="첫 정상 샘플의 shape와 RMSD 출력",
    )

    # 이전 명령어와 호환용. 실제로는 get_struct_tokenizer() 기본값을 사용합니다.
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="호환용 인자. 현재 스크립트에서는 직접 사용하지 않음",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warning] cuda를 사용할 수 없어 cpu로 변경합니다.")
        device = "cpu"

    print("[info] loading structure tokenizer...")
    struct_tokenizer = get_struct_tokenizer()
    struct_tokenizer = struct_tokenizer.to(device)
    struct_tokenizer.eval()

    pkl_paths = sorted(
        glob(os.path.join(args.input_pkl_dir, "**", args.pattern), recursive=True)
    )

    if args.num_samples is not None:
        pkl_paths = pkl_paths[: args.num_samples]

    if len(pkl_paths) == 0:
        raise FileNotFoundError(
            f"pkl 파일을 찾지 못했습니다: {args.input_pkl_dir}/**/{args.pattern}"
        )

    print(f"[info] found {len(pkl_paths)} pkl files")
    print(f"[info] output_dir = {args.output_dir}")
    print(f"[info] max_length = {args.max_length}")
    print(f"[info] align = {not args.no_align}")

    manifest = []
    num_ok = 0
    num_skip = 0
    num_fail = 0

    for idx, pkl_path in enumerate(tqdm(pkl_paths, desc="building residual dataset")):
        try:
            sample, status = build_one_sample(
                pkl_path=pkl_path,
                struct_tokenizer=struct_tokenizer,
                device=device,
                max_length=args.max_length,
                align=not args.no_align,
                debug=(args.debug and num_ok == 0),
            )

            if sample is None:
                num_skip += 1
                manifest.append((pkl_path, status, ""))
                continue

            out_name = f"{idx:06d}_{safe_stem(pkl_path)}.pt"
            out_path = os.path.join(args.output_dir, out_name)

            torch.save(sample, out_path)

            num_ok += 1
            manifest.append((pkl_path, "ok", out_path))

        except Exception as e:
            num_fail += 1
            manifest.append((pkl_path, f"fail:{repr(e)}", ""))

    manifest_path = os.path.join(args.output_dir, "manifest.tsv")
    with open(manifest_path, "w") as f:
        f.write("source_path\tstatus\toutput_path\n")
        for row in manifest:
            f.write("\t".join(map(str, row)) + "\n")

    print("[done]")
    print(f"  ok   : {num_ok}")
    print(f"  skip : {num_skip}")
    print(f"  fail : {num_fail}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()