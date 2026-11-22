"""예측 SMILES의 점수 계산과 구조 그림. 점수는 Kaggle 채점과 같은 규칙을 RDKit으로 구현한다."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator

from scripts.kaggleutils import normalize_smiles

# Morgan 지문: 반지름 2, 2048비트, 입체화학 미반영(기본값)
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def tanimoto_similarity(true_smiles: str, pred_smiles) -> float:
    """정답과 예측 한 쌍의 Tanimoto similarity를 계산한다.

    Args:
        true_smiles: 정답 SMILES(이미 표준 SMILES). 읽지 못하면 ValueError.
        pred_smiles: 모델의 예측. 무효한 예측이면 0점이다.
    Returns:
        0 이상 1 이하의 점수.
    """
    true_mol = Chem.MolFromSmiles(true_smiles)
    if true_mol is None or true_mol.GetNumAtoms() == 0:
        raise ValueError(f"정답 SMILES를 읽을 수 없습니다: {true_smiles!r}")

    normalized = normalize_smiles(pred_smiles)
    if normalized == "":
        return 0.0
    pred_mol = Chem.MolFromSmiles(normalized)
    return DataStructs.TanimotoSimilarity(
        _MORGAN.GetFingerprint(true_mol), _MORGAN.GetFingerprint(pred_mol)
    )


def score_predictions(true_smiles: list[str], pred_smiles: list) -> dict:
    """예측 전체의 점수를 집계한다.

    Args:
        true_smiles: 정답 SMILES 목록.
        pred_smiles: 같은 순서, 같은 길이의 예측 목록(정규화 전 문자열 또는 None).
    Returns:
        - tanimoto: 평균 Tanimoto (무효한 예측은 0). Kaggle 점수와 같은 정의이다.
        - valid_ratio: 유효한 예측의 비율.
        - exact_match: 정규화한 예측이 정답 문자열과 같은 비율(입체화학까지 일치).
        - per_sample: 표본별 {"pred_normalized", "tanimoto", "is_valid"} 목록.
    """
    if len(true_smiles) != len(pred_smiles):
        raise ValueError(
            f"정답 {len(true_smiles)}개와 예측 {len(pred_smiles)}개의 수가 다릅니다."
        )
    if len(true_smiles) == 0:
        raise ValueError("점수를 계산할 표본이 없습니다.")

    per_sample = []
    num_exact = 0
    for true, pred in zip(true_smiles, pred_smiles):
        normalized = normalize_smiles(pred)
        per_sample.append(
            {
                "pred_normalized": normalized,
                "tanimoto": tanimoto_similarity(true, pred),
                "is_valid": normalized != "",
            }
        )
        num_exact += int(normalized == true)

    count = len(per_sample)
    return {
        "tanimoto": sum(sample["tanimoto"] for sample in per_sample) / count,
        "valid_ratio": sum(sample["is_valid"] for sample in per_sample) / count,
        "exact_match": num_exact / count,
        "per_sample": per_sample,
    }


def plot_structures(image_paths: list[str | Path], smiles_columns: dict[str, list[str]]) -> Figure:
    """입력 이미지와 SMILES의 구조 그림을 표본마다 한 줄에 나란히 그린다.

    유효하지 않은 SMILES는 빈 칸에 "invalid"로 표시한다.

    Args:
        image_paths: 입력 이미지 경로 목록.
        smiles_columns: {열 제목: SMILES 목록}. 예: {"true": [...], "prediction": [...]}
            목록의 순서와 길이는 image_paths와 같다.
    Returns:
        matplotlib Figure.
    """
    num_rows, num_cols = len(image_paths), 1 + len(smiles_columns)
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows), squeeze=False
    )
    for row, image_path in enumerate(image_paths):
        axes[row, 0].imshow(Image.open(image_path))
        axes[row, 0].set_title(Path(image_path).stem, fontsize=9)
        for col, (title, smiles_list) in enumerate(smiles_columns.items(), start=1):
            axes[row, col].set_title(title, fontsize=9)
            normalized = normalize_smiles(smiles_list[row])
            if normalized:
                mol = Chem.MolFromSmiles(normalized)
                axes[row, col].imshow(Draw.MolToImage(mol, size=(256, 256)))
            else:
                axes[row, col].text(0.5, 0.5, "invalid", ha="center", va="center")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    return fig
