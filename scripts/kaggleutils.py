"""Kaggle 제출에 필요한 도구: SMILES 정규화, 제출 파일 생성과 형식 검사."""

from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

# 무효한 예측마다 RDKit 경고가 출력되는 것을 막는다.
RDLogger.DisableLog("rdApp.*")

# Kaggle 채점기가 지원하지 않는 표기
_UNSUPPORTED = ("*", ":", ">", "<")

# Kaggle 채점기는 이보다 긴 예측을 0점으로 처리한다.
MAX_SMILES_LENGTH = 1000


def normalize_smiles(smiles) -> str:
    """예측 SMILES를 제출용 표준 SMILES로 바꾼다. 유효하지 않으면 빈 문자열을 돌려준다.

    RDKit이 다시 읽어도 달라지지 않는 문자열이 될 때까지 변환을 반복한다.

    Args:
        smiles: 모델이 생성한 문자열. 문자열이 아니면 무효로 본다.
    Returns:
        RDKit 표준 SMILES 또는 빈 문자열.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    text = smiles.strip()
    for _ in range(5):
        mol = Chem.MolFromSmiles(text)
        if mol is None or mol.GetNumAtoms() == 0:
            return ""
        normalized = Chem.MolToSmiles(mol)
        if normalized == text:
            if any(mark in normalized for mark in _UNSUPPORTED):
                return ""
            return normalized
        text = normalized
    return ""


def make_submission(image_ids: list[str], predicted_smiles: list, path: str | Path) -> pd.DataFrame:
    """예측을 정규화하여 제출 파일(`image_id,SMILES`)로 저장한다.

    무효한 예측과 1,000자를 넘는 예측은 빈 칸으로 둔다.

    Args:
        image_ids: 이미지 ID 목록. sample_submission.csv와 같은 순서여야 한다.
        predicted_smiles: 같은 순서의 모델 출력 문자열.
        path: 저장할 CSV 경로.
    Returns:
        저장한 내용의 DataFrame.
    """
    if len(image_ids) != len(predicted_smiles):
        raise ValueError(
            f"image_id {len(image_ids)}개와 예측 {len(predicted_smiles)}개의 수가 다릅니다."
        )
    normalized = [normalize_smiles(smiles) for smiles in predicted_smiles]
    normalized = [s if len(s) <= MAX_SMILES_LENGTH else "" for s in normalized]

    submission = pd.DataFrame({"image_id": image_ids, "SMILES": normalized})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False, lineterminator="\n")

    num_blank = normalized.count("")
    print(f"제출 파일 저장: {path} (전체 {len(submission)}행, 빈 칸 {num_blank}행)")
    return submission


def validate_submission(path: str | Path, sample_submission_path: str | Path) -> None:
    """제출 파일의 형식을 검사한다. 어긋나면 내용을 밝히는 ValueError를 낸다.

    Args:
        path: 검사할 제출 파일.
        sample_submission_path: `data/sample_submission.csv`의 경로.
    """
    # keep_default_na=False: 빈 칸을 NaN이 아닌 빈 문자열로 읽는다.
    submission = pd.read_csv(path, dtype=str, keep_default_na=False)
    sample = pd.read_csv(sample_submission_path, dtype=str, keep_default_na=False)

    columns = list(submission.columns)
    if columns != ["image_id", "SMILES"]:
        raise ValueError(f"열이 image_id, SMILES여야 합니다. 현재 열: {columns}")

    if len(submission) != len(sample):
        raise ValueError(f"행 수가 {len(sample)}이어야 합니다. 현재 {len(submission)}행입니다.")

    for row, (actual, expected) in enumerate(zip(submission["image_id"], sample["image_id"])):
        if actual != expected:
            raise ValueError(
                f"{row + 1}번째 행의 image_id가 {expected}여야 합니다. 현재 {actual}입니다."
            )

    for image_id, smiles in zip(submission["image_id"], submission["SMILES"]):
        if smiles and normalize_smiles(smiles) != smiles:
            raise ValueError(f"{image_id}의 SMILES가 정규화되어 있지 않습니다: {smiles!r}")
