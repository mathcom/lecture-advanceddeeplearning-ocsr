"""학습 로그(SQLite)의 기록·조회와 요약 표, 학습 곡선.

기록은 학습 프로세스의 `TrainLogger`가 하고, 조회는 모니터링 노트북이 `read_*` 함수로 한다.
"""

import json
import platform
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import rdkit
import timm
import torch
from matplotlib.figure import Figure

from scripts import config as config_utils

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT    NOT NULL,
    ended_at    TEXT,
    status      TEXT    NOT NULL,
    start_step  INTEGER NOT NULL,
    end_step    INTEGER,
    config_json TEXT    NOT NULL,
    env_json    TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS scalars (
    run_id    INTEGER NOT NULL,
    step      INTEGER NOT NULL,
    split     TEXT    NOT NULL,
    name      TEXT    NOT NULL,
    value     REAL    NOT NULL,
    logged_at TEXT    NOT NULL,
    PRIMARY KEY (run_id, step, split, name)
);
CREATE TABLE IF NOT EXISTS samples (
    run_id      INTEGER NOT NULL,
    step        INTEGER NOT NULL,
    image_id    TEXT    NOT NULL,
    true_smiles TEXT    NOT NULL,
    pred_smiles TEXT    NOT NULL,
    tanimoto    REAL    NOT NULL,
    is_valid    INTEGER NOT NULL,
    PRIMARY KEY (run_id, step, image_id)
);
CREATE INDEX IF NOT EXISTS idx_scalars_name_step ON scalars (split, name, step);
"""

RUN_COLUMNS = [
    "run_id", "started_at", "ended_at", "status", "start_step", "end_step", "config_json", "env_json",
]
SCALAR_COLUMNS = ["step", "split", "name", "value", "run_id"]
SAMPLE_COLUMNS = ["run_id", "step", "image_id", "true_smiles", "pred_smiles", "tanimoto", "is_valid"]

# 설정(실행)을 구분하는 색. 색맹이어도 구분되도록 고른 순서이다.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _now() -> str:
    """현재 시각의 ISO 8601 UTC 문자열."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def collect_env() -> dict[str, str]:
    """실행 환경(Python과 패키지 버전, 장치 이름)을 모은다."""
    has_gpu = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "rdkit": rdkit.__version__,
        "device": torch.cuda.get_device_name(0) if has_gpu else "cpu",
    }


# ---------------------------------------------------------------------------
# 쓰기
# ---------------------------------------------------------------------------

class TrainLogger:
    """학습 로그를 SQLite에 기록한다. 기록할 때마다 곧바로 commit한다.

    Args:
        db_path: 로그 파일 경로. 폴더가 없으면 만든다.
        journal_mode: "WAL" 또는 "DELETE". 네트워크 파일시스템에서는 "DELETE"를 쓴다.
    """

    def __init__(self, db_path: str | Path, journal_mode: str):
        if journal_mode not in ("WAL", "DELETE"):
            raise ValueError(f"log.journal_mode는 WAL 또는 DELETE여야 합니다: {journal_mode}")
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(db_path, timeout=30)
        self.connection.execute(f"PRAGMA journal_mode={journal_mode}")
        self.connection.executescript(SCHEMA)
        self.run_id: int | None = None

    def start_run(self, config: dict, start_step: int) -> int:
        """실행 한 행을 추가하고 run_id를 돌려준다."""
        with self.connection:  # with를 벗어날 때 commit된다.
            cursor = self.connection.execute(
                "INSERT INTO runs (started_at, status, start_step, config_json, env_json) "
                "VALUES (?, 'running', ?, ?, ?)",
                (_now(), start_step, json.dumps(config), json.dumps(collect_env())),
            )
        self.run_id = cursor.lastrowid
        return self.run_id

    def log_scalars(self, step: int, split: str, values: dict[str, float]) -> None:
        """스칼라 지표를 기록한다.

        Args:
            step: 학습 step.
            split: "train" 또는 "val".
            values: 지표 이름과 값.
        """
        rows = [
            (self.run_id, step, split, name, float(value), _now()) for name, value in values.items()
        ]
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO scalars VALUES (?, ?, ?, ?, ?, ?)", rows
            )

    def log_samples(self, step: int, rows: list[dict]) -> None:
        """예측 표본을 기록한다.

        Args:
            step: 학습 step.
            rows: {"image_id", "true_smiles", "pred_smiles", "tanimoto", "is_valid"}의 목록.
        """
        values = [
            (
                self.run_id, step, row["image_id"], row["true_smiles"], row["pred_smiles"],
                float(row["tanimoto"]), int(row["is_valid"]),
            )
            for row in rows
        ]
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO samples VALUES (?, ?, ?, ?, ?, ?, ?)", values
            )

    def end_run(self, status: str, end_step: int) -> None:
        """실행을 마친다. status는 "finished", "interrupted", "failed" 중 하나이다."""
        with self.connection:
            self.connection.execute(
                "UPDATE runs SET ended_at = ?, status = ?, end_step = ? WHERE run_id = ?",
                (_now(), status, end_step, self.run_id),
            )

    def close(self) -> None:
        """연결을 닫는다."""
        self.connection.close()


# ---------------------------------------------------------------------------
# 읽기
# ---------------------------------------------------------------------------

def _read(db_path: str | Path, table: str, columns: list[str]) -> pd.DataFrame:
    """읽기 전용으로 DB를 열어 표 하나를 읽는다. 파일이나 표가 없으면 빈 DataFrame을 돌려준다."""
    if not Path(db_path).exists():
        return pd.DataFrame(columns=columns)
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        # 학습이 막 시작되어 파일만 있고 표는 아직 없을 수 있다.
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        if (table,) not in tables:
            return pd.DataFrame(columns=columns)
        # 나중 실행의 행이 뒤에 오도록 run_id 순으로 읽는다.
        query = f"SELECT {', '.join(columns)} FROM {table} ORDER BY run_id, rowid"
        return pd.read_sql_query(query, connection)
    finally:
        connection.close()


def read_runs(db_path: str | Path) -> pd.DataFrame:
    """실행 이력을 run_id 순으로 돌려준다."""
    return _read(db_path, "runs", RUN_COLUMNS)


def read_scalars(db_path: str | Path) -> pd.DataFrame:
    """스칼라 지표를 step 순으로 돌려준다(열: step, split, name, value, run_id).

    재개 때문에 같은 (step, split, name)이 여러 번 기록되었으면 가장 나중 실행의 값만 남긴다.
    """
    frame = _read(db_path, "scalars", SCALAR_COLUMNS)
    frame = frame.drop_duplicates(subset=["step", "split", "name"], keep="last")
    return frame.sort_values(["split", "name", "step"], ignore_index=True)


def read_samples(db_path: str | Path, step: int | None = None) -> pd.DataFrame:
    """한 검증 step에서 기록된 예측 표본을 돌려준다.

    Args:
        db_path: 로그 파일 경로.
        step: 검증 step. None이면 표본이 기록된 가장 최근 step.
    """
    frame = _read(db_path, "samples", SAMPLE_COLUMNS)
    if step is None:
        step = frame["step"].max()
    frame = frame[frame["step"] == step]
    return frame.drop_duplicates(subset=["image_id"], keep="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 모니터링 노트북의 표와 그림
# ---------------------------------------------------------------------------

def summarize(config_names: list[str]) -> pd.DataFrame:
    """config별 현재 step, 최근 손실, 최근 검증 지표, 최고 검증 Tanimoto의 표를 만든다."""
    rows = []
    for name in config_names:
        scalars = read_scalars(config_utils.db_path(name))
        if scalars.empty:
            rows.append({"config": name, "상태": "로그 없음"})
            continue
        runs = read_runs(config_utils.db_path(name))
        row = {"config": name, "상태": runs["status"].iloc[-1], "현재 step": scalars["step"].max()}

        train_loss = get_curve(scalars, "train", "loss")
        if not train_loss.empty:
            row["최근 훈련 손실"] = train_loss["value"].iloc[-1]
        for metric in ("loss", "tanimoto", "valid_ratio", "exact_match"):
            curve = get_curve(scalars, "val", metric)
            if not curve.empty:
                row[f"최근 검증 {metric}"] = curve["value"].iloc[-1]

        tanimoto = get_curve(scalars, "val", "tanimoto")
        if not tanimoto.empty:
            best = tanimoto.loc[tanimoto["value"].idxmax()]  # 처음으로 최고가 된 step
            row["최고 검증 tanimoto"] = best["value"]
            row["최고 step"] = best["step"]
        rows.append(row)
    return pd.DataFrame(rows)


def get_curve(scalars: pd.DataFrame, split: str, name: str) -> pd.DataFrame:
    """`read_scalars`의 결과에서 지표 하나를 step 순으로 뽑는다."""
    return scalars[(scalars["split"] == split) & (scalars["name"] == name)]


def plot_curves(
    config_names: list[str], panels: dict[str, list[tuple[str, str]]], save_path: str | Path
) -> Figure:
    """여러 실행의 지표 곡선을 한 그림에 겹쳐 그리고 파일로 저장한다.

    색은 config를 구분하고, 실선은 train, 점이 있는 파선은 val이다.

    Args:
        config_names: 비교할 config 이름들. 로그가 없는 것은 건너뛴다.
        panels: {칸 제목: [(split, name), ...]}. 칸마다 그 지표들을 그린다.
        save_path: 저장할 PNG 경로. 폴더가 없으면(아직 학습하지 않았으면) 저장하지 않는다.
    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4), squeeze=False)
    for i, config_name in enumerate(config_names):
        scalars = read_scalars(config_utils.db_path(config_name))
        if scalars.empty:
            print(f"{config_name}: 로그 없음")
            continue
        for ax, curves in zip(axes[0], panels.values()):
            for split, name in curves:
                curve = get_curve(scalars, split, name)
                ax.plot(
                    curve["step"], curve["value"],
                    color=COLORS[i % len(COLORS)],
                    linewidth=2,
                    linestyle="-" if split == "train" else "--",
                    marker=None if split == "train" else "o",
                    label=f"{config_name} {split}/{name}",
                )
    for ax, title in zip(axes[0], panels):
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, color="#e5e5e5")
        ax.spines[["top", "right"]].set_visible(False)
        if ax.lines:
            ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    if Path(save_path).parent.exists():
        fig.savefig(save_path, dpi=120)
    return fig
