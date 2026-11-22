"""설정 파일(JSON)의 로드·검증과 실행 디렉토리 경로.

config 파일 이름이 곧 실행 이름이다. `configs/config001.json`으로 학습한 결과는
모두 `outputs/config001/` 아래에 저장된다.
"""

import json
from pathlib import Path

# 모든 경로는 main/을 작업 디렉토리로 하는 상대 경로이다.
CONFIGS_DIR = Path("configs")
OUTPUTS_DIR = Path("outputs")

# 설정 파일에 있어야 하는 키. 하나라도 없거나 모르는 키가 있으면 오류를 낸다.
# encoder는 type에 따라 키가 달라서 ENCODER_KEYS에 따로 적는다.
REQUIRED_KEYS = {
    "": {"seed", "data", "image", "tokenizer", "encoder", "decoder", "train", "eval", "log"},
    "data": {"data_dir", "val_size", "max_train_samples", "extra_train_sources"},
    "image": {"size", "augment"},
    "image.augment": {"rotate_deg", "translate", "scale"},
    "tokenizer": {"max_length"},
    "decoder": {"dim", "depth", "num_heads", "mlp_ratio", "dropout"},
    "train": {
        "batch_size", "max_steps", "lr", "encoder_lr_scale", "weight_decay", "warmup_steps",
        "warmup_init_lr", "min_lr_ratio", "grad_clip", "label_smoothing", "amp", "num_workers",
        "log_every", "eval_every", "save_every", "keep_last_k",
    },
    "eval": {"batch_size", "num_logged_samples"},
    "log": {"journal_mode"},
}
ENCODER_KEYS = {
    "timm": {"type", "name", "pretrained", "drop_path_rate"},
    "scratch": {"type", "patch_size", "dim", "depth", "num_heads", "mlp_ratio", "dropout"},
}
EXTRA_SOURCE_KEYS = {"csv", "image_dir"}


def load_config(name: str) -> dict:
    """`configs/{name}.json`을 읽고 검증하여 돌려준다.

    Args:
        name: 확장자 없는 파일 이름. 예: "config001"
    Returns:
        설정 딕셔너리.
    """
    path = CONFIGS_DIR / f"{name}.json"
    if not path.exists():
        available = sorted(p.stem for p in CONFIGS_DIR.glob("*.json"))
        raise FileNotFoundError(f"{path} 파일이 없습니다. 사용 가능한 설정: {available}")
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """키가 빠짐없이, 그리고 남김없이 있는지 검사한다. 어긋나면 ValueError."""
    for section, required in REQUIRED_KEYS.items():
        part = config
        for key in section.split(".") if section else []:
            part = part[key]
        _check_keys(part, required, section)

    encoder = config["encoder"]
    if not isinstance(encoder, dict) or encoder.get("type") not in ENCODER_KEYS:
        raise ValueError(f"encoder.type은 {sorted(ENCODER_KEYS)} 중 하나여야 합니다.")
    _check_keys(encoder, ENCODER_KEYS[encoder["type"]], "encoder")

    for i, source in enumerate(config["data"]["extra_train_sources"]):
        _check_keys(source, EXTRA_SOURCE_KEYS, f"data.extra_train_sources[{i}]")


def _check_keys(part: dict, required: set[str], section: str) -> None:
    """설정의 한 부분(part)이 required와 정확히 같은 키를 가지는지 검사한다."""
    if not isinstance(part, dict):
        raise ValueError(f"설정 {section}은 키와 값의 묶음(JSON 객체)이어야 합니다.")
    prefix = f"{section}." if section else ""
    missing = [prefix + key for key in sorted(required - part.keys())]
    unknown = [prefix + key for key in sorted(part.keys() - required)]
    if missing or unknown:
        raise ValueError(f"설정 키 오류 - 빠진 키: {missing}, 모르는 키: {unknown}")


def run_dir(name: str) -> Path:
    """실행 디렉토리 `outputs/{name}`."""
    return OUTPUTS_DIR / name


def ckpts_dir(name: str) -> Path:
    """체크포인트 폴더."""
    return run_dir(name) / "ckpts"


def logs_dir(name: str) -> Path:
    """로그와 그림 폴더."""
    return run_dir(name) / "logs"


def predictions_dir(name: str) -> Path:
    """제출 파일 폴더."""
    return run_dir(name) / "predictions"


def db_path(name: str) -> Path:
    """SQLite 로그 파일."""
    return logs_dir(name) / "logs.sqlite3"


def vocab_path(name: str) -> Path:
    """토크나이저 어휘 파일."""
    return run_dir(name) / "vocab.json"


def split_path(name: str) -> Path:
    """train/val 분할 파일."""
    return run_dir(name) / "split.json"


def save_snapshot(config: dict, name: str) -> None:
    """학습 시작 시점의 설정 사본을 `outputs/{name}/config.json`에 쓴다."""
    path = run_dir(name) / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def load_snapshot(name: str) -> dict:
    """학습 시작 시점의 설정 사본을 읽는다."""
    path = run_dir(name) / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} 파일이 없습니다. 먼저 {name}을 학습하세요.")
    return json.loads(path.read_text(encoding="utf-8"))


def diff_config(a: dict, b: dict) -> list[str]:
    """두 설정에서 값이 다른 키를 점 표기(예: "train.lr")로 돌려준다.

    한쪽에만 있는 키도 다른 것으로 본다.
    """
    differences = []
    for key in sorted(a.keys() | b.keys()):
        if key not in a or key not in b:
            differences.append(key)
        elif isinstance(a[key], dict) and isinstance(b[key], dict):
            differences += [f"{key}.{inner}" for inner in diff_config(a[key], b[key])]
        elif a[key] != b[key]:
            differences.append(key)
    return differences
