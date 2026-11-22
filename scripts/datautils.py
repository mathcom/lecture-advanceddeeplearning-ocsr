"""데이터 읽기, train/val 분할, 이미지 변환, Dataset과 DataLoader."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from scripts.kaggleutils import normalize_smiles
from scripts.network import image_normalization
from scripts.tokenizer import PAD_ID, SmilesTokenizer

# data.data_dir 안의 파일과 폴더 이름
TRAIN_CSV = "smiles_train.csv"
TRAIN_IMAGE_DIR = "images_train"
TEST_IMAGE_DIR = "images_test"
SAMPLE_SUBMISSION_CSV = "sample_submission.csv"
ZINC_CSV = "250k_rndm_zinc_drugs_clean_3.csv"


# ---------------------------------------------------------------------------
# 데이터 읽기
# ---------------------------------------------------------------------------

def read_csv(path: str | Path) -> pd.DataFrame:
    """CSV의 모든 값을 문자열로 읽는다(빈 칸은 NaN이 아닌 빈 문자열)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_train_smiles(config: dict) -> dict[str, str]:
    """제공된 훈련 데이터의 `{image_id: SMILES}`를 돌려준다."""
    frame = read_csv(Path(config["data"]["data_dir"]) / TRAIN_CSV)
    return dict(zip(frame["image_id"], frame["SMILES"]))


def build_tokenizer(config: dict) -> SmilesTokenizer:
    """훈련 SMILES와 ZINC250k SMILES에 나온 토큰의 합집합으로 토크나이저를 만든다.

    ZINC250k는 어휘를 만드는 데에만 쓴다.
    """
    data_dir = Path(config["data"]["data_dir"])
    train_smiles = read_csv(data_dir / TRAIN_CSV)["SMILES"].tolist()
    zinc_smiles = read_csv(data_dir / ZINC_CSV)["smiles"].tolist()
    return SmilesTokenizer.build(train_smiles + zinc_smiles, config["tokenizer"]["max_length"])


# ---------------------------------------------------------------------------
# train/val 분할
# ---------------------------------------------------------------------------

def image_id_key(image_id: str) -> tuple[int, int]:
    """image_id를 번호순으로 정렬하기 위한 키. "CDK_Depict_10_101" -> (10, 101)"""
    first, second = image_id.split("_")[-2:]
    return int(first), int(second)


def split_train_val(image_ids: list[str], val_size: int, seed: int) -> tuple[list[str], list[str]]:
    """image_id를 훈련용과 검증용으로 무작위로 나눈다.

    Kaggle의 test도 같은 모집단에서 단순 무작위로 뽑혔으므로 단순 무작위로 나눈다.

    Args:
        image_ids: 전체 image_id.
        val_size: 검증용 개수.
        seed: 난수 시드. 같은 시드이면 같은 분할이 나온다.
    Returns:
        (train_ids, val_ids)
    """
    shuffled = sorted(image_ids, key=image_id_key)
    random.Random(seed).shuffle(shuffled)
    return shuffled[val_size:], shuffled[:val_size]


def make_split(config: dict) -> dict:
    """설정에 따라 분할을 만든다. `save_split`으로 저장하는 내용과 같은 형태이다."""
    image_ids = list(read_train_smiles(config))
    val_size, seed = config["data"]["val_size"], config["seed"]
    train_ids, val_ids = split_train_val(image_ids, val_size, seed)
    return {"seed": seed, "val_size": val_size, "train_ids": train_ids, "val_ids": val_ids}


def save_split(split: dict, path: str | Path) -> None:
    """분할을 JSON 파일로 저장한다."""
    Path(path).write_text(json.dumps(split, indent=2), encoding="utf-8")


def load_split(path: str | Path) -> dict:
    """`save_split`으로 저장한 분할을 읽는다."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 이미지 변환
# ---------------------------------------------------------------------------

def build_train_transform(config: dict) -> transforms.Compose:
    """훈련용 변환을 만든다. 이미지 증강을 바꾸려면 이 함수를 고친다.

    좌우·상하 반전은 쓰지 않는다. 거울상은 입체화학이 반대인 분자이므로 정답 SMILES와
    어긋난다. 평면 안의 회전은 분자를 바꾸지 않으므로 써도 된다.

    Returns:
        RGB PIL 이미지를 (3, H, W) 텐서로 바꾸는 변환.
    """
    augment = config["image"]["augment"]
    steps = [_resize(config)]

    no_augment = (
        augment["rotate_deg"] == 0
        and augment["translate"] == 0
        and list(augment["scale"]) == [1.0, 1.0]
    )
    if not no_augment:
        steps.append(
            transforms.RandomAffine(
                degrees=augment["rotate_deg"],
                translate=(augment["translate"], augment["translate"]),
                scale=tuple(augment["scale"]),
                interpolation=InterpolationMode.BILINEAR,
                fill=(255, 255, 255),  # 빈 영역은 흰색
            )
        )
    return transforms.Compose(steps + _to_normalized_tensor(config))


def build_eval_transform(config: dict) -> transforms.Compose:
    """검증·추론용 변환(증강 없음)을 만든다.

    Returns:
        RGB PIL 이미지를 (3, H, W) 텐서로 바꾸는 변환.
    """
    return transforms.Compose([_resize(config)] + _to_normalized_tensor(config))


def _resize(config: dict) -> transforms.Resize:
    """bicubic 크기 조정. 이미 같은 크기이면 이미지를 그대로 둔다."""
    size = config["image"]["size"]
    return transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC)


def _to_normalized_tensor(config: dict) -> list:
    """PIL 이미지를 텐서로 바꾸고 정규화하는 변환들."""
    mean, std = image_normalization(config)
    return [transforms.ToTensor(), transforms.Normalize(mean, std)]


def to_display_image(image: torch.Tensor, config: dict) -> np.ndarray:
    """정규화된 이미지 텐서를 matplotlib으로 그릴 수 있게 되돌린다.

    Args:
        image: (3, H, W)
    Returns:
        (H, W, 3), 값의 범위 [0, 1]
    """
    mean, std = image_normalization(config)
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Dataset과 DataLoader
# ---------------------------------------------------------------------------

class OCSRDataset(Dataset):
    """분자 이미지와 SMILES의 Dataset.

    Args:
        items: (image_id, 이미지 경로, SMILES 또는 None)의 목록. test이면 SMILES가 None이다.
        transform: PIL 이미지를 텐서로 바꾸는 변환.
        tokenizer: SMILES를 토큰 번호로 바꿀 토크나이저. test에서는 필요 없다.
    """

    def __init__(
        self,
        items: list[tuple[str, Path, str | None]],
        transform: transforms.Compose,
        tokenizer: SmilesTokenizer | None = None,
    ):
        self.items = items
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        image_id, path, smiles = self.items[index]
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        sample = {
            "image": self.transform(image),  # (3, H, W)
            "image_id": image_id,
            "smiles": smiles if smiles is not None else "",
        }
        if smiles is not None:
            sample["input_ids"] = torch.tensor(self.tokenizer.encode(smiles))  # (T,)
        return sample


def collate_fn(samples: list[dict]) -> dict:
    """표본들을 배치로 묶는다. input_ids는 배치 내 최대 길이에 맞춰 <pad>로 채운다.

    Returns:
        "image" (B, 3, H, W), "image_id"와 "smiles" 문자열 목록,
        정답이 있으면 "input_ids" (B, T)
    """
    batch = {
        "image": torch.stack([sample["image"] for sample in samples]),
        "image_id": [sample["image_id"] for sample in samples],
        "smiles": [sample["smiles"] for sample in samples],
    }
    if "input_ids" in samples[0]:
        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            [sample["input_ids"] for sample in samples], batch_first=True, padding_value=PAD_ID
        )
    return batch


def load_extra_train_items(
    sources: list[dict], tokenizer: SmilesTokenizer
) -> list[tuple[str, Path, str]]:
    """추가 훈련 데이터를 읽는다. 쓸 수 없는 행은 제외하고 사유별 개수를 출력한다.

    Args:
        sources: `{"csv": 경로, "image_dir": 경로}`의 목록.
        tokenizer: 토큰과 길이를 검사할 토크나이저.
    Returns:
        (image_id, 이미지 경로, 정규화된 SMILES)의 목록.
    """
    items = []
    for source in sources:
        excluded = {"정규화 실패": 0, "어휘에 없는 토큰": 0, "max_length 초과": 0, "이미지 없음": 0}
        num_kept = 0
        frame = read_csv(source["csv"])
        for image_id, raw_smiles in zip(frame["image_id"], frame["SMILES"]):
            smiles = normalize_smiles(raw_smiles)
            path = Path(source["image_dir"]) / f"{image_id}.png"
            reason = _exclusion_reason(smiles, path, tokenizer)
            if reason is None:
                items.append((image_id, path, smiles))
                num_kept += 1
            else:
                excluded[reason] += 1
        print(f"추가 훈련 데이터 {source['csv']}: 사용 {num_kept}개, 제외 {excluded}")
    return items


def _exclusion_reason(smiles: str, path: Path, tokenizer: SmilesTokenizer) -> str | None:
    """추가 훈련 데이터 한 행의 제외 사유를 돌려준다. 쓸 수 있으면 None."""
    if smiles == "":
        return "정규화 실패"
    try:
        tokens = tokenizer.tokenize(smiles)
    except ValueError:  # 정규식이 읽지 못하는 문자
        return "어휘에 없는 토큰"
    if any(token not in tokenizer.token_to_id for token in tokens):
        return "어휘에 없는 토큰"
    if len(tokens) + 2 > tokenizer.max_length:  # <bos>와 <eos>를 더한 길이
        return "max_length 초과"
    if not path.exists():
        return "이미지 없음"
    return None


def build_dataloaders(
    config: dict, tokenizer: SmilesTokenizer, split: dict
) -> tuple[DataLoader, DataLoader]:
    """훈련 로더와 검증 로더를 만든다.

    Args:
        config: 설정.
        tokenizer: 토크나이저.
        split: `make_split` 또는 `load_split`의 결과.
    Returns:
        (train_loader, val_loader). 검증 로더는 image_id 번호순이다.
    """
    smiles_of = read_train_smiles(config)
    image_dir = Path(config["data"]["data_dir"]) / TRAIN_IMAGE_DIR

    # max_train_samples가 None이면 [:None]이 되어 전부 사용한다.
    train_ids = split["train_ids"][: config["data"]["max_train_samples"]]
    val_ids = sorted(split["val_ids"], key=image_id_key)
    train_items = [(i, image_dir / f"{i}.png", smiles_of[i]) for i in train_ids]
    val_items = [(i, image_dir / f"{i}.png", smiles_of[i]) for i in val_ids]

    # 추가 데이터는 훈련에만 넣는다.
    train_items += load_extra_train_items(config["data"]["extra_train_sources"], tokenizer)

    batch_size = config["train"]["batch_size"]
    if len(train_items) < batch_size:
        raise ValueError(f"훈련 표본 {len(train_items)}개가 배치 크기 {batch_size}보다 적습니다.")

    num_workers = config["train"]["num_workers"]
    train_loader = DataLoader(
        OCSRDataset(train_items, build_train_transform(config), tokenizer),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        OCSRDataset(val_items, build_eval_transform(config), tokenizer),
        batch_size=config["eval"]["batch_size"],
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def build_test_loader(config: dict) -> DataLoader:
    """test 로더를 만든다. 순서는 sample_submission.csv와 같고 정답은 없다."""
    data_dir = Path(config["data"]["data_dir"])
    image_ids = read_csv(data_dir / SAMPLE_SUBMISSION_CSV)["image_id"]
    items = [(i, data_dir / TEST_IMAGE_DIR / f"{i}.png", None) for i in image_ids]
    return DataLoader(
        OCSRDataset(items, build_eval_transform(config)),
        batch_size=config["eval"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        collate_fn=collate_fn,
    )
