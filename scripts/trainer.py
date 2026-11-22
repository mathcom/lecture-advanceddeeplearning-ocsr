"""학습·검증 루프, 체크포인트 저장·복원, 예측.

노트북은 `fit` 하나를 호출한다. 학습은 epoch이 아니라 step(배치 하나) 기준으로 진행한다.
"""

import math
import os
import random
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from scripts import config as config_utils
from scripts import datautils, evalutils
from scripts.logutils import TrainLogger
from scripts.network import OCSRModel, build_model, compute_loss
from scripts.tokenizer import SmilesTokenizer

# JupyterHub 서버에서는 PyTorch가 할당된 CPU(8개)보다 훨씬 많은 스레드를 만들어 느려진다.
# 이 모듈을 불러올 때 CPU 스레드 수를 제한한다.
MAX_CPU_THREADS = 8
torch.set_num_threads(min(MAX_CPU_THREADS, torch.get_num_threads()))


def fit(config_name: str, resume: bool = False) -> dict:
    """모델을 학습한다. 결과는 모두 `outputs/{config_name}/`에 저장한다.

    Args:
        config_name: `configs/`의 설정 파일 이름(확장자 제외).
        resume: True이면 `last.pth`에서 이어서 학습한다.
    Returns:
        {"best_step", "best_tanimoto", "last_step", "run_dir"}.
        검증을 한 번도 하지 않고 중단했으면 best_*는 None이다.
    """
    config = config_utils.load_config(config_name)
    resuming = check_run_dir(config_name, resume)
    trainer = Trainer(config, config_name, resuming)
    return trainer.run()


def check_run_dir(config_name: str, resume: bool) -> bool:
    """실행 디렉토리의 상태를 보고 재개할지(True) 새로 시작할지(False) 정한다.

    이미 학습한 흔적이 있는데 resume=False이거나, 재개할 체크포인트가 없으면 오류를 낸다.
    기존 파일을 지우거나 덮어쓰는 일은 하지 않는다.
    """
    run_dir = config_utils.run_dir(config_name)
    checkpoints = list(config_utils.ckpts_dir(config_name).glob("*.pth"))
    has_last = (config_utils.ckpts_dir(config_name) / "last.pth").exists()
    has_log = config_utils.db_path(config_name).exists()

    if not checkpoints and not has_log:
        return False
    if not resume:
        raise FileExistsError(
            f"{run_dir}에 이미 학습한 기록이 있습니다. 이어서 학습하려면 RESUME = True로 "
            f"바꾸고, 새로 학습하려면 {run_dir} 폴더를 직접 지우거나 config 파일을 "
            "복사해 다른 이름으로 실행하세요."
        )
    if not has_last:
        raise FileNotFoundError(
            f"재개할 체크포인트가 없습니다: {run_dir}/ckpts/last.pth. 새로 학습하려면 "
            f"{run_dir} 폴더를 직접 지우거나 config 파일을 복사해 다른 이름으로 실행하세요."
        )
    return True


class Trainer:
    """학습에 필요한 것(모델, 데이터, 옵티마이저, 로그)을 준비하고 학습 루프를 돌린다.

    Args:
        config: 검증된 설정.
        config_name: 실행 이름.
        resuming: True이면 저장된 상태에서 이어서 학습한다.
    """

    def __init__(self, config: dict, config_name: str, resuming: bool):
        self.config = config
        self.config_name = config_name
        self.resuming = resuming
        self.device = get_device()
        self.amp = config["train"]["amp"] and self.device.type == "cuda"
        self.ckpts_dir = config_utils.ckpts_dir(config_name)
        self.step = 0
        self.best_step: int | None = None
        self.best_tanimoto: float | None = None

        if resuming:
            tokenizer, split = self.load_run_files()
        else:
            tokenizer, split = self.create_run_files()
        self.tokenizer: SmilesTokenizer = tokenizer

        self.train_loader, self.val_loader = datautils.build_dataloaders(config, tokenizer, split)
        # 재개할 때는 곧바로 체크포인트로 덮어쓰므로 사전학습 가중치를 내려받지 않는다.
        model = build_model(config, tokenizer, load_pretrained=not resuming)
        self.model = model.to(self.device)
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)

        if resuming:
            self.load_last()

    # ----- 시작 절차 -----

    def create_run_files(self) -> tuple[SmilesTokenizer, dict]:
        """새 실행: 시드를 정하고 설정 사본, 어휘, 분할을 만들어 저장한다."""
        set_seed(self.config["seed"])
        config_utils.save_snapshot(self.config, self.config_name)
        tokenizer = datautils.build_tokenizer(self.config)
        tokenizer.save(config_utils.vocab_path(self.config_name))
        split = datautils.make_split(self.config)
        datautils.save_split(split, config_utils.split_path(self.config_name))
        return tokenizer, split

    def load_run_files(self) -> tuple[SmilesTokenizer, dict]:
        """재개: 설정이 그대로인지 확인하고 저장된 어휘와 분할을 읽는다."""
        snapshot = config_utils.load_snapshot(self.config_name)
        differences = config_utils.diff_config(snapshot, self.config)
        if differences:
            raise ValueError(
                f"학습을 시작할 때와 설정이 다릅니다: {differences}. 설정을 되돌리거나 "
                "config 파일을 복사해 다른 이름으로 실행하세요."
            )
        tokenizer = SmilesTokenizer.load(config_utils.vocab_path(self.config_name))
        split = datautils.load_split(config_utils.split_path(self.config_name))
        return tokenizer, split

    def load_last(self) -> None:
        """`last.pth`에서 모델과 학습 상태를 복원한다."""
        state = load_checkpoint(self.config_name, "last")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.step = state["step"]
        self.best_step = state["best_step"]
        self.best_tanimoto = state["best_tanimoto"]
        set_seed(self.config["seed"] + self.step)

    # ----- 학습 -----

    def run(self) -> dict:
        """학습을 실행하고 로그에 실행의 시작과 끝을 기록한다."""
        logger = TrainLogger(
            config_utils.db_path(self.config_name), self.config["log"]["journal_mode"]
        )
        self.logger = logger
        logger.start_run(self.config, self.step)
        try:
            self.train_loop()
            logger.end_run("finished", self.step)
        except KeyboardInterrupt:  # 노트북의 중단 버튼
            self.save_last()
            logger.end_run("interrupted", self.step)
            print(f"step {self.step}에서 중단했습니다. RESUME = True로 이어서 학습할 수 있습니다.")
        except Exception:
            logger.end_run("failed", self.step)
            raise
        finally:
            logger.close()

        return {
            "best_step": self.best_step,
            "best_tanimoto": self.best_tanimoto,
            "last_step": self.step,
            "run_dir": str(config_utils.run_dir(self.config_name)),
        }

    def train_loop(self) -> None:
        """max_steps까지 학습한다. 도중에 로그를 남기고, 검증하고, 체크포인트를 저장한다."""
        train_config = self.config["train"]
        max_steps = train_config["max_steps"]
        batches = endless_batches(self.train_loader)
        interval = {"loss": [], "grad_norm": [], "seconds": 0.0, "samples": 0}

        with tqdm(total=max_steps, initial=self.step, desc=self.config_name) as progress:
            while self.step < max_steps:
                started = time.time()
                batch = next(batches)
                lr = self.optimizer.param_groups[-1]["lr"]  # 마지막 그룹은 디코더이다.
                loss, grad_norm = self.train_step(batch)
                self.step += 1

                interval["loss"].append(loss)
                # AMP가 건너뛴 step의 gradient norm은 유한하지 않으므로 평균에서 뺀다.
                if math.isfinite(grad_norm):
                    interval["grad_norm"].append(grad_norm)
                interval["seconds"] += time.time() - started
                interval["samples"] += len(batch["image_id"])
                progress.update(1)

                if self.step % train_config["log_every"] == 0:
                    self.log_train_interval(interval, lr)
                    epoch = self.step / len(self.train_loader)
                    progress.set_postfix(epoch=f"{epoch:.1f}", loss=f"{loss:.3f}")
                    interval = {"loss": [], "grad_norm": [], "seconds": 0.0, "samples": 0}

                # 마지막 step의 검증은 루프가 끝난 뒤에 한다.
                if self.step % train_config["eval_every"] == 0 and self.step < max_steps:
                    self.validate()
                if self.step % train_config["save_every"] == 0:
                    self.save_periodic()
                    self.save_last()

        self.validate()

    def train_step(self, batch: dict) -> tuple[float, float]:
        """배치 하나로 가중치를 한 번 갱신한다.

        Returns:
            (손실, clip 전 gradient norm)
        """
        self.model.train()
        images = batch["image"].to(self.device)         # (B, 3, H, W)
        input_ids = batch["input_ids"].to(self.device)  # (B, T)

        with autocast(self.device, self.amp):
            # 앞 토큰들(마지막 제외)을 보고 한 칸 뒤의 토큰을 맞힌다.
            logits = self.model(images, input_ids[:, :-1])  # (B, T-1, V)
            loss = compute_loss(
                logits, input_ids[:, 1:], self.config["train"]["label_smoothing"]
            )
        if not math.isfinite(loss.item()):
            raise RuntimeError(
                f"step {self.step + 1}에서 손실이 {loss.item()}입니다. 학습이 발산했습니다."
            )

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)  # gradient를 원래 크기로 되돌린 뒤 clip한다.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config["train"]["grad_clip"]
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return loss.item(), grad_norm.item()

    def log_train_interval(self, interval: dict, lr: float) -> None:
        """직전 구간의 훈련 지표를 기록한다."""
        values = {
            "loss": float(np.mean(interval["loss"])),
            "lr": lr,
            "samples_per_sec": interval["samples"] / interval["seconds"],
        }
        if interval["grad_norm"]:
            values["grad_norm"] = float(np.mean(interval["grad_norm"]))
        self.logger.log_scalars(self.step, "train", values)

    def validate(self) -> None:
        """검증하여 기록하고, 최고 점수를 넘으면 best.pth를, 그리고 항상 last.pth를 저장한다."""
        metrics, per_sample = evaluate(
            self.model, self.val_loader, self.tokenizer, self.device, self.amp
        )
        self.logger.log_scalars(self.step, "val", metrics)
        num_logged = self.config["eval"]["num_logged_samples"]
        self.logger.log_samples(self.step, per_sample[:num_logged])  # 검증 로더는 번호순이다.
        tqdm.write(
            f"step {self.step}: val loss {metrics['loss']:.4f}, "
            f"tanimoto {metrics['tanimoto']:.4f}, valid {metrics['valid_ratio']:.3f}, "
            f"exact {metrics['exact_match']:.3f}"
        )

        if self.best_tanimoto is None or metrics["tanimoto"] > self.best_tanimoto:
            self.best_step = self.step
            self.best_tanimoto = metrics["tanimoto"]
            self.save("best.pth", {"metrics": metrics})
        self.save_last()

    # ----- 체크포인트 저장 -----

    def save(self, file_name: str, extra: dict) -> None:
        """모델 가중치, step, 설정에 extra를 더하여 저장한다."""
        state = {"model": self.model.state_dict(), "step": self.step, "config": self.config}
        state.update(extra)
        save_atomic(state, self.ckpts_dir / file_name)

    def save_last(self) -> None:
        """재개에 필요한 전체 상태를 last.pth에 저장한다."""
        extra = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_step": self.best_step,
            "best_tanimoto": self.best_tanimoto,
        }
        self.save("last.pth", extra)

    def save_periodic(self) -> None:
        """주기 체크포인트를 저장하고 최근 keep_last_k개만 남긴다."""
        self.save(f"step{self.step:06d}.pth", {})
        paths = sorted(self.ckpts_dir.glob("step*.pth"), key=lambda path: int(path.stem[4:]))
        num_to_delete = max(0, len(paths) - self.config["train"]["keep_last_k"])
        for path in paths[:num_to_delete]:
            path.unlink()


# ---------------------------------------------------------------------------
# 검증과 예측
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: OCSRModel,
    val_loader: DataLoader,
    tokenizer: SmilesTokenizer,
    device: torch.device,
    amp: bool,
) -> tuple[dict, list[dict]]:
    """검증 세트 전체에서 손실과 생성 결과의 점수를 계산한다.

    Args:
        model: 모델. 평가 모드(`model.eval()`)로 바뀐다.
        val_loader: 정답이 있는 로더.
        tokenizer: 토크나이저.
        device: 장치.
        amp: CUDA에서 혼합 정밀도를 쓸지 여부.
    Returns:
        metrics: loss(토큰당 평균), tanimoto, valid_ratio, exact_match, eval_seconds
        per_sample: 표본별 {"image_id", "true_smiles", "pred_smiles", "tanimoto", "is_valid"}.
            pred_smiles는 정규화 전의 모델 출력이다.
    """
    started = time.time()
    model.eval()

    # (1) teacher forcing 손실
    loss_sum, num_tokens = 0.0, 0
    for batch in val_loader:
        images = batch["image"].to(device)         # (B, 3, H, W)
        input_ids = batch["input_ids"].to(device)  # (B, T)
        targets = input_ids[:, 1:]                 # (B, T-1)
        with autocast(device, amp):
            logits = model(images, input_ids[:, :-1])  # (B, T-1, V)
            loss = compute_loss(logits, targets)
        count = (targets != tokenizer.pad_id).sum().item()
        loss_sum += loss.item() * count
        num_tokens += count

    # (2) greedy 생성 결과의 점수
    image_ids = [item[0] for item in val_loader.dataset.items]
    true_smiles = [item[2] for item in val_loader.dataset.items]
    pred_smiles = predict_smiles(model, val_loader, tokenizer, device, amp)
    scores = evalutils.score_predictions(true_smiles, pred_smiles)

    per_sample = [
        {
            "image_id": image_id,
            "true_smiles": true,
            "pred_smiles": pred,
            "tanimoto": score["tanimoto"],
            "is_valid": score["is_valid"],
        }
        for image_id, true, pred, score in zip(
            image_ids, true_smiles, pred_smiles, scores["per_sample"]
        )
    ]
    metrics = {
        "loss": loss_sum / num_tokens,
        "tanimoto": scores["tanimoto"],
        "valid_ratio": scores["valid_ratio"],
        "exact_match": scores["exact_match"],
        "eval_seconds": time.time() - started,
    }
    return metrics, per_sample


@torch.no_grad()
def predict_smiles(
    model: OCSRModel,
    loader: DataLoader,
    tokenizer: SmilesTokenizer,
    device: torch.device,
    amp: bool,
) -> list[str]:
    """로더의 순서대로 greedy 생성한 문자열(정규화 전)을 돌려준다.

    모델은 평가 모드로 바뀐다. 정답이 없는 test 로더에도 쓸 수 있다.
    """
    model.eval()
    predictions = []
    for batch in loader:
        images = batch["image"].to(device)  # (B, 3, H, W)
        with autocast(device, amp):
            ids = model.generate(
                images, tokenizer.bos_id, tokenizer.eos_id, tokenizer.max_length - 1
            )  # (B, T)
        predictions += [tokenizer.decode(row) for row in ids.tolist()]
    return predictions


# ---------------------------------------------------------------------------
# 옵티마이저와 학습률
# ---------------------------------------------------------------------------

def build_optimizer(model: OCSRModel, config: dict) -> torch.optim.AdamW:
    """AdamW를 만든다.

    인코더와 디코더의 학습률을 다르게 하고, 차원이 2 이상인 가중치에만 weight decay를
    적용한다(bias, LayerNorm, 위치 임베딩, CLS 토큰은 제외). 마지막 그룹은 항상 디코더이다.
    """
    train_config = config["train"]
    groups = []
    for module, lr in (
        (model.encoder, train_config["lr"] * train_config["encoder_lr_scale"]),
        (model.decoder, train_config["lr"]),
    ):
        decay, no_decay = [], []
        for name, parameter in module.named_parameters():
            is_excluded = "pos_embed" in name or "cls_token" in name
            if parameter.ndim >= 2 and not is_excluded:
                decay.append(parameter)
            else:
                no_decay.append(parameter)
        groups.append(
            {"params": decay, "lr": lr, "weight_decay": train_config["weight_decay"]}
        )
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict
) -> torch.optim.lr_scheduler.LambdaLR:
    """선형 warmup 뒤 cosine으로 감소하는 학습률 스케줄러를 만든다.

    학습률은 `warmup_init_lr`에서 시작해 `lr`까지 커진다. 인코더에는 같은 비율을 적용하므로
    인코더의 시작 학습률은 `warmup_init_lr × encoder_lr_scale`이다.
    """
    warmup_steps = config["train"]["warmup_steps"]
    init_ratio = config["train"]["warmup_init_lr"] / config["train"]["lr"]
    max_steps = config["train"]["max_steps"]
    min_ratio = config["train"]["min_lr_ratio"]

    def lr_ratio(step: int) -> float:
        """step번째 갱신에서 최대 학습률에 곱할 비율."""
        if step < warmup_steps:
            return init_ratio + (1 - init_ratio) * step / warmup_steps  # init_ratio -> 1
        progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))  # 0 -> 1
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_ratio)


# ---------------------------------------------------------------------------
# 체크포인트
# ---------------------------------------------------------------------------

def save_atomic(state: dict, path: Path) -> None:
    """임시 파일에 쓴 뒤 바꿔치기한다. 저장 도중 중단되어도 기존 파일이 깨지지 않는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(config_name: str, which: str) -> dict:
    """체크포인트를 CPU 메모리로 읽는다.

    Args:
        config_name: 실행 이름.
        which: "best", "last", "step001000" 같은 이름.
    """
    folder = config_utils.ckpts_dir(config_name)
    path = folder / f"{which}.pth"
    if not path.exists():
        available = sorted(p.stem for p in folder.glob("*.pth"))
        raise FileNotFoundError(f"{path}가 없습니다. 사용 가능한 체크포인트: {available}")
    return torch.load(path, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# 작은 도구들
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """CUDA가 있으면 CUDA, 없으면 CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Python, NumPy, PyTorch의 난수 시드를 정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast(device: torch.device, amp: bool) -> torch.amp.autocast:
    """혼합 정밀도 영역. CUDA가 아니면 아무 효과가 없다."""
    enabled = amp and device.type == "cuda"
    return torch.amp.autocast(device.type, dtype=torch.float16, enabled=enabled)


def endless_batches(loader: DataLoader) -> Iterator[dict]:
    """epoch이 끝나면 다시 섞어 처음부터 배치를 계속 꺼낸다."""
    while True:
        yield from loader
