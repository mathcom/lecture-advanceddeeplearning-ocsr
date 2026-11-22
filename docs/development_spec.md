# 개발 명세

> **문서 상태: 개발 완료(2026-11-22).** 이 문서는 견본 모델(baseline)의 소스코드를 개발하기 위한 작업 지시서였다. 사용 방법은 최상위 폴더의 `README.md`를 본다.
> 개발하면서 담당 교수의 지시로 바꾼 부분은 네 가지이다: `environment.yml`에 `ipykernel` 추가(2.1절), warmup 시작 학습률 `train.warmup_init_lr` 추가(6.2절, 6.3절, 7.8절), CLS 토큰을 weight decay에서 제외(6.3절), CPU 스레드 수 제한(2.1절). 본문은 바뀐 내용으로 고쳐져 있다.
> 배포본에서는 학생이 다룰 소스코드를 한곳에 모으기 위해 `tests/`를 뺐다. 본문의 `tests/`와 10.1절에 관한 내용은 개발할 때의 기록이다. 또한 프로젝트 기준점으로 `configs/config002.json`(ViT-Base 인코더)을 추가했다.
> 개발자는 이 명세를 처음부터 끝까지 읽은 뒤 11절의 순서로 작업한다. 명세와 실제 환경이 어긋나면 임의로 우회하지 말고 13절에 따라 보고한다.

## 1. 목적과 범위

### 1.1 과제

손으로 그린 분자 구조 이미지를 입력받아 SMILES 문자열을 출력하는 OCSR(Optical Chemical Structure Recognition) 모델을 만든다. 성능은 Kaggle에서 정답 SMILES와의 평균 Tanimoto similarity로 평가한다(4절).

### 1.2 개발 대상

학생에게 출발점으로 제시할 **견본 모델**과 그 실행 도구이다.

| 산출물 | 내용 |
|---|---|
| `scripts/` | 모델, 토크나이저, 데이터, 학습, 평가, 로그, 제출 관련 Python 모듈 |
| `configs/` | 모델·학습 파라미터를 담은 JSON 파일 |
| `ocsr_train.ipynb` | config를 읽어 모델을 학습하고, 검증·체크포인트·로그를 남김 |
| `ocsr_train_monitoring.ipynb` | SQLite 로그를 읽어 학습 진행 상황을 그림과 표로 보여 줌 |
| `ocsr_inference.ipynb` | 체크포인트로 test 이미지를 예측하여 Kaggle 제출 파일을 만듦 |
| `tests/` | 표준 라이브러리 `unittest` 기반의 자동 점검 |

### 1.3 견본 모델의 수준

견본은 학생이 개선할 여지를 남기도록 의도적으로 단순하게 만든다.

- **포함**: 사전학습 ViT 인코더, 직접 구현한 GPT 디코더(cross-attention), 원자 단위 SMILES 토크나이저, 제공된 손그림 훈련 이미지만 사용한 지도학습, 약한 기하 증강, greedy 디코딩.
- **포함하지 않음**: VAE/GAN 기반 데이터 증강, ZINC250k 분자를 렌더링한 합성 이미지 학습, beam search, test-time augmentation, test 이미지의 무라벨 활용(pseudo-label 등), 앙상블. 이 항목들은 구현하지 않는다.

### 1.4 학생이 확장할 지점

학생이 견본의 다른 부분을 고치지 않고도 자신의 방법을 끼워 넣을 수 있도록 아래 세 지점을 분리해 둔다. 그 밖의 확장 장치는 만들지 않는다.

1. **추가 훈련 데이터**: config의 `data.extra_train_sources`에 (CSV, 이미지 폴더) 쌍을 적으면 훈련 데이터에 합쳐진다(7.3절). 학생이 생성한 이미지를 넣는 통로이다.
2. **이미지 증강**: `datautils.build_train_transform(config)` 한 함수가 훈련용 변환을 만든다.
3. **디코딩**: `OCSRModel.generate`가 학습 코드와 분리되어 있다.

## 2. 개발 환경과 제약

### 2.1 환경

- 실행 환경은 JupyterHub 서버이다. JupyterLab은 허브가 제공하므로 `environment.yml`에 `jupyterlab`이 없다. 노트북 커널은 `environment.yml`의 `ipykernel`로 만든다.
- PyTorch가 할당된 CPU보다 많은 스레드를 만들어 느려지므로, `trainer.py`를 불러올 때 `torch.set_num_threads`로 8개까지로 제한한다.
- 패키지는 `environment.yml`의 고정 버전을 쓴다: Python 3.12, torch 2.15.0, torchvision 0.29.2, rdkit 2026.9.2, numpy 2.5.5, pandas 3.1.1, pillow 13.0.0, matplotlib 3.11.2, scikit-learn 1.9.1, tqdm 4.70.1, timm 1.0.31, transformers 5.23.0, ipykernel 7.4.0.
- **패키지를 추가하지 않는다.** SQLite는 표준 라이브러리 `sqlite3`, 테스트는 `unittest`, 설정은 `json`을 쓴다.
- 라이브러리 API는 기억에 의존하지 말고 설치된 버전에서 확인한 뒤 사용한다. 아래는 이 명세를 작성하면서 같은 버전에서 확인한 사항이다.
  - `timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=False, img_size=256, num_classes=0)`의 `forward_features`는 256×256 입력에 대해 `(B, 257, 384)`를 돌려준다(CLS 1개 + 패치 256개). 파라미터 수는 21,688,704이다.
  - `timm.get_pretrained_cfg("vit_small_patch16_224.augreg_in21k_ft_in1k")`의 `mean`, `std`는 모두 `(0.5, 0.5, 0.5)`이다.
  - `torch.amp.autocast`, `torch.amp.GradScaler`, `torch.nn.functional.scaled_dot_product_attention`이 있다.
  - 텐서, 숫자, 문자열, `None`, 리스트, 딕셔너리로만 이루어진 체크포인트는 `torch.load(path, weights_only=True)`로 읽힌다.

### 2.2 제약

- `data/`의 파일은 읽기만 한다. 수정, 삭제, 추가하지 않는다.
- 견본 코드는 `data/`에 제공된 데이터만 사용한다. 인터넷에서 다른 데이터셋을 내려받지 않는다.
- 사전학습 가중치는 범용 가중치(ImageNet 등 일반 이미지로 학습된 것)만 쓴다. 화학 구조 이미지나 분자 데이터로 학습된 가중치는 쓰지 않는다.
- 모든 경로는 `main/`(저장소 최상위 폴더)을 기준으로 한 상대 경로이다. 노트북은 `main/`을 작업 디렉토리로 하여 실행한다. 절대 경로를 코드에 적지 않는다.
- 학습 프레임워크(Lightning, Hugging Face `Trainer`, Hydra, wandb, TensorBoard 등)를 쓰지 않는다. 학습 루프는 PyTorch로 직접 작성한다.

### 2.3 코드 작성 기준

학부생이 읽고 수정하는 견본이므로 성능보다 가독성을 우선한다.

- 식별자는 영어, 주석과 docstring은 한국어로 쓴다. 노트북의 설명 셀도 한국어로 쓴다.
- 모듈마다 역할을 설명하는 docstring을 두고, 공개 함수·클래스에 인자, 반환값, 텐서 shape을 적는다. `network.py`는 연산마다 shape 주석을 단다(예: `# (B, T, D)`).
- 타입 힌트를 쓴다.
- 필요 이상의 추상화(레지스트리, 플러그인, 콜백 체계)를 만들지 않는다.
- 노트북에는 로직을 두지 않는다. 노트북은 설정 선택, `scripts` 함수 호출, 결과 표시만 한다.
- 노트북은 출력을 비운 상태로 저장한다.

## 3. 제공 데이터

| 경로 | 내용 | 형식 |
|---|---|---|
| `data/smiles_train.csv` | 훈련 분자 2,894개의 정답 | 헤더 `image_id,SMILES`, 줄바꿈 CRLF |
| `data/images_train/{image_id}.png` | 훈련용 손그림 이미지 2,894장 | PNG |
| `data/images_test/{image_id}.png` | 평가용 손그림 이미지 300장(정답 없음) | PNG |
| `data/sample_submission.csv` | 제출 형식 예시 300행. `SMILES`는 모두 `C` | 헤더 `image_id,SMILES`, 줄바꿈 LF |
| `data/250k_rndm_zinc_drugs_clean_3.csv` | 추가 분자 249,425개(이미지 없음) | 헤더 `smiles,logP,qed,SAS`, 줄바꿈 LF |

확인된 사실:

- 모든 SMILES는 RDKit 표준 SMILES이다. 즉 `normalize_smiles`(4.2절)를 적용해도 문자열이 바뀌지 않는다. 방향족 고리는 소문자(`c1ccccc1`)로 적혀 있고 입체화학 표기(`@`, `@@`, `/`, `\`)가 남아 있다.
- `image_id`는 `CDK_Depict_{번호}_{번호}` 형태이며 PNG 파일 이름과 같다.
- 7.2절의 토크나이저 기준으로, 토큰 길이의 최댓값은 훈련 SMILES 58, ZINC250k 72이다. 토큰 종류는 훈련 72개, ZINC250k 62개, 합집합 85개이다.
- test 이미지 중 48장을 표본으로 확인했을 때 모두 256×256, RGB, 흰 배경에 어두운 선이었다. **전체 이미지에 대해서는 확인하지 않았다.** 개발 첫 단계에서 훈련·평가 이미지 전체의 크기와 모드를 집계하고, 256×256 RGB가 아닌 이미지가 있으면 보고한다.
- ZINC250k에는 훈련·평가 분자와 연결 구조가 같은 분자가 없다(제거되어 있음).
- test 정답은 이 저장소에 없다. 견본 코드는 정답을 필요로 하지 않아야 한다.

견본 모델은 ZINC250k를 **토크나이저 어휘 구성에만** 사용한다.

## 4. 평가 방법

검증 지표와 제출 파일은 Kaggle 채점과 정확히 같은 규칙을 따라야 한다. 이 절의 내용은 규범이며 임의로 바꾸지 않는다.

### 4.1 Kaggle 채점 규칙

- 제출 파일은 `image_id`, `SMILES` 두 열을 가지며 `sample_submission.csv`와 같은 300개 `image_id`를 담는다.
- 행마다 정답과 예측을 Morgan fingerprint(반지름 2, 2048비트, 입체화학 미반영)로 바꾸어 Tanimoto similarity를 계산하고, 전체 행의 평균이 점수이다.
- 빈 칸이 있는 파일은 제출할 수 없다. 유효한 SMILES가 아니거나 1,000자를 넘는 예측은 0점이다. 제출 파일에서는 이런 예측을 `submission.default_smiles`로 채운다(7.6절).
- 채점기는 RDKit 없이 동작하며 SMILES를 **적힌 그대로** 읽는다. 따라서 제출하는 SMILES는 반드시 RDKit 표준 SMILES로 정규화되어 있어야 한다. 정규화하지 않은 예측(예: Kekulé 표기 `C1=CC=CC=C1`)은 같은 분자여도 점수가 낮게 나온다.
- 채점기는 wildcard 원자(`*`), 명시적 방향족 결합(`:`), 배위 결합(`->`, `<-`)을 지원하지 않는다. 이 표기가 남은 예측은 무효로 처리한다.

### 4.2 정규화 함수 (규범)

`scripts/kaggleutils.py`의 `normalize_smiles`는 아래와 동작이 같아야 한다. RDKit이 다시 읽어도 달라지지 않는 문자열이 될 때까지 변환을 반복한다.

```python
from rdkit import Chem

_UNSUPPORTED = ("*", ":", ">", "<")


def normalize_smiles(smiles) -> str:
    """예측 SMILES를 제출용 표준 SMILES로 바꾼다. 유효하지 않으면 빈 문자열을 돌려준다."""
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
```

모듈을 불러올 때 `RDLogger.DisableLog("rdApp.*")`로 RDKit 경고 출력을 끈다(무효 예측마다 경고가 출력되는 것을 막기 위함).

### 4.3 로컬 점수 계산 (규범)

`scripts/evalutils.py`는 RDKit으로 같은 점수를 계산한다.

- 지문: `rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)`의 `GetFingerprint`.
- 행 점수: `normalize_smiles(예측)`이 빈 문자열이면 0, 아니면 `DataStructs.TanimotoSimilarity(정답 지문, 정규화된 예측 지문)`.
- 정답은 이미 표준 SMILES이므로 그대로 읽는다.

이 방식이 Kaggle 채점기와 같은 지문을 낸다는 점은 훈련·평가 분자 3,194개와 ZINC250k 249,425개 전체에서 확인되어 있다.

### 4.4 기준값

아래 15쌍은 Kaggle 채점 스크립트로 계산한 값이다. `tests/test_metrics.py`는 `evalutils`가 이 값을 소수 여섯째 자리까지 재현하는지 검사한다.

| # | 정답 | 예측(원문) | 정규화 결과 | Tanimoto |
|---|---|---|---|---|
| 1 | `CCO` | `OCC` | `CCO` | 1.000000 |
| 2 | `c1ccccc1` | `C1=CC=CC=C1` | `c1ccccc1` | 1.000000 |
| 3 | `CCO` | `CCN` | `CCN` | 0.333333 |
| 4 | `C[C@H](N)C(=O)O` | `CC(N)C(=O)O` | `CC(N)C(=O)O` | 1.000000 |
| 5 | `CC(=O)Oc1ccccc1C(=O)O` | `COC(=O)c1ccccc1OC(C)=O` | `COC(=O)c1ccccc1OC(C)=O` | 0.666667 |
| 6 | `Cn1c(=O)c2c(ncn2C)n(C)c1=O` | `Cn1cnc2c1c(=O)[nH]c(=O)n2C` | `Cn1cnc2c1c(=O)[nH]c(=O)n2C` | 0.529412 |
| 7 | `C[N+](C)(C)CC(=O)[O-]` | `CN(C)CC(=O)O` | `CN(C)CC(=O)O` | 0.208333 |
| 8 | `[2H]C([2H])([2H])O` | `CO` | `CO` | 0.500000 |
| 9 | `CC(C)Cc1ccc(C(C)C(=O)O)cc1` | `CC(C)Cc1ccc(CC(=O)O)cc1` | `CC(C)Cc1ccc(CC(=O)O)cc1` | 0.600000 |
| 10 | `CCO` | 앞뒤 공백이 붙은 `" CCO "` | `CCO` | 1.000000 |
| 11 | `CCO` | `C1CC` (닫히지 않은 고리) | 빈 문자열 | 0.000000 |
| 12 | `CCO` | 빈 문자열 `""` | 빈 문자열 | 0.000000 |
| 13 | `CCO` | `None` | 빈 문자열 | 0.000000 |
| 14 | `CCO` | `C*` (지원하지 않는 표기) | 빈 문자열 | 0.000000 |
| 15 | `O=C(O)c1ccccc1` | `O=C(O)C1=CC=CC=C1` | `O=C(O)c1ccccc1` | 1.000000 |

15쌍 전체의 집계값: 평균 Tanimoto **0.522516**, 유효 비율 **0.733333**(11/15), 완전 일치율 **0.266667**(4/15).
4번은 입체화학만 다른 예로, Tanimoto는 1이지만 완전 일치는 아니다.

## 5. 디렉토리 구조

```
main/
├ data/                          # 제공 데이터 (읽기 전용)
├ scripts/
│ ├ __init__.py
│ ├ config.py                    # JSON 로드·검증, 실행 디렉토리 경로
│ ├ tokenizer.py                 # SMILES 토크나이저, 어휘 저장·로드
│ ├ datautils.py                 # Dataset, 전처리·증강, train/val 분할, DataLoader
│ ├ network.py                   # ViT 인코더, GPT 디코더, 생성
│ ├ trainer.py                   # 학습·검증 루프, 체크포인트 저장·복원, 예측
│ ├ evalutils.py                 # Morgan 지문, Tanimoto, 유효 비율, 일치율
│ ├ logutils.py                  # SQLite 기록·조회
│ └ kaggleutils.py               # normalize_smiles, 제출 파일 생성·형식 검사
├ configs/
│ ├ config000.json               # 동작 확인용 소형 설정 (GPU 없이 실행 가능)
│ └ config001.json               # 견본 설정
├ outputs/                       # 실행 시 생성 (git에 올리지 않음)
│ └ config001/
│   ├ config.json                # 학습 시작 시점의 설정 사본
│   ├ vocab.json                 # 토크나이저 어휘
│   ├ split.json                 # train/val 분할
│   ├ ckpts/
│   │ ├ last.pth                 # 재개용 전체 상태
│   │ ├ best.pth                 # 검증 Tanimoto가 가장 높았던 시점의 가중치
│   │ └ step001000.pth, ...      # 주기 저장 (가중치만, 최근 k개 유지)
│   ├ logs/
│   │ ├ logs.sqlite3
│   │ └ plot_loss.png, plot_metrics.png, plot_lr.png   # 모니터링 노트북이 저장
│   └ predictions/
│     └ submission_best.csv      # Kaggle 제출용 (파일 이름에 체크포인트 이름 포함)
├ tests/
│ └ test_*.py
├ ocsr_train.ipynb
├ ocsr_train_monitoring.ipynb
├ ocsr_inference.ipynb
├ environment.yml
├ .gitignore
└ README.md
```

- config 파일 이름(확장자 제외)이 곧 실행 이름이며, `outputs/` 아래 하위 디렉토리 이름이 된다. 하나의 config는 하나의 실행에 대응한다. 다른 조건으로 학습하려면 config 파일을 복사하여 새 이름으로 만든다.
- `.gitignore`에는 `outputs/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.pyc`를 넣는다.

## 6. 설정 파일

### 6.1 원칙

- config 파일에는 **모든 키를 빠짐없이** 적는다. 코드에 숨은 기본값을 두지 않는다. config 파일만 보면 실행 조건 전체를 알 수 있어야 한다.
- `scripts/config.py`는 로드할 때 키를 검증한다. 필요한 키가 없거나 모르는 키가 있으면 어느 키가 문제인지 밝히는 오류를 낸다(오타 방지).
- JSON에는 주석을 쓸 수 없으므로 키의 의미는 6.3절의 표로 관리한다.

### 6.2 견본 설정 `configs/config001.json`

```json
{
  "seed": 42,
  "data": {
    "data_dir": "data",
    "val_size": 300,
    "max_train_samples": null,
    "extra_train_sources": []
  },
  "image": {
    "size": 256,
    "augment": {
      "rotate_deg": 10,
      "translate": 0.05,
      "scale": [0.9, 1.1]
    }
  },
  "tokenizer": {
    "max_length": 128
  },
  "encoder": {
    "type": "timm",
    "name": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "pretrained": true,
    "drop_path_rate": 0.1
  },
  "decoder": {
    "dim": 384,
    "depth": 6,
    "num_heads": 6,
    "mlp_ratio": 4.0,
    "dropout": 0.1
  },
  "train": {
    "batch_size": 32,
    "max_steps": 20000,
    "lr": 0.0003,
    "encoder_lr_scale": 0.1,
    "weight_decay": 0.05,
    "warmup_steps": 500,
    "warmup_init_lr": 0.000001,
    "min_lr_ratio": 0.01,
    "grad_clip": 1.0,
    "label_smoothing": 0.0,
    "amp": true,
    "num_workers": 4,
    "log_every": 50,
    "eval_every": 500,
    "save_every": 1000,
    "keep_last_k": 3
  },
  "eval": {
    "batch_size": 64,
    "num_logged_samples": 32
  },
  "log": {
    "journal_mode": "WAL"
  },
  "submission": {
    "default_smiles": "C"
  }
}
```

**이 수치들은 초깃값이며 성능 검증을 거치지 않았다.** 서버에서 학습해 본 뒤 조정하고, 조정한 값과 근거를 보고한다(12.2절).

참고: 훈련 2,594장(2,894 − 검증 300), 배치 32이면 1 epoch은 81 step이고, 20,000 step은 약 247 epoch이다.

### 6.3 키 설명

| 키 | 의미 |
|---|---|
| `seed` | 난수 시드. train/val 분할, 가중치 초기화, 데이터 순서에 쓰인다 |
| `data.data_dir` | 데이터 폴더. 그 안의 파일 이름(`smiles_train.csv` 등)은 `datautils.py`의 상수로 둔다 |
| `data.val_size` | 훈련 데이터에서 떼어 낼 검증 분자 수 |
| `data.max_train_samples` | `null`이면 전부 사용. 정수이면 `split.json`의 `train_ids` 순서에서 앞의 그 개수만 사용(동작 확인용) |
| `data.extra_train_sources` | 추가 훈련 데이터 목록. 각 원소는 `{"csv": 경로, "image_dir": 경로}` (7.3절) |
| `image.size` | 모델 입력의 한 변 길이(정사각형) |
| `image.augment.rotate_deg` | 훈련 시 무작위 회전 범위(±도) |
| `image.augment.translate` | 훈련 시 무작위 평행이동 범위(이미지 크기 대비 비율) |
| `image.augment.scale` | 훈련 시 무작위 확대·축소 범위 `[최소, 최대]` |
| `tokenizer.max_length` | `<bos>`, `<eos>`를 포함한 최대 토큰 수. 디코더 위치 임베딩의 길이이기도 하다 |
| `encoder.type` | `"timm"` 또는 `"scratch"` |
| `encoder.name`, `encoder.pretrained`, `encoder.drop_path_rate` | `type`이 `"timm"`일 때만 쓰는 키. timm 모델 이름, 사전학습 가중치 사용 여부, stochastic depth 비율 |
| `encoder.patch_size`, `encoder.dim`, `encoder.depth`, `encoder.num_heads`, `encoder.mlp_ratio`, `encoder.dropout` | `type`이 `"scratch"`일 때만 쓰는 키 |
| `decoder.dim`, `decoder.depth`, `decoder.num_heads`, `decoder.mlp_ratio`, `decoder.dropout` | GPT 디코더의 은닉 차원, 블록 수, 헤드 수, MLP 확장 비율, dropout |
| `train.batch_size`, `train.max_steps` | 배치 크기, 총 학습 step 수 |
| `train.lr` | 디코더(및 scratch 인코더)의 최대 학습률 |
| `train.encoder_lr_scale` | 인코더 학습률 = `lr × encoder_lr_scale`. 사전학습 인코더는 낮게 둔다 |
| `train.weight_decay` | AdamW weight decay. 차원이 2 이상인 가중치에만 적용하고 bias, LayerNorm, 위치 임베딩, CLS 토큰에는 적용하지 않는다 |
| `train.warmup_steps`, `train.warmup_init_lr`, `train.min_lr_ratio` | 선형 warmup step 수, warmup의 시작 학습률(디코더 기준. 인코더는 `× encoder_lr_scale`), cosine 감쇠의 최종 학습률 비율 |
| `train.grad_clip` | gradient norm 상한 |
| `train.label_smoothing` | cross-entropy의 label smoothing |
| `train.amp` | CUDA에서 혼합 정밀도 사용 여부. CPU에서는 무시한다 |
| `train.num_workers` | DataLoader 워커 수 |
| `train.log_every`, `train.eval_every`, `train.save_every` | 훈련 지표 기록, 검증, 주기 체크포인트 저장의 간격(step) |
| `train.keep_last_k` | 주기 체크포인트(`stepNNNNNN.pth`)를 최근 몇 개까지 남길지 |
| `eval.batch_size` | 검증·추론 배치 크기 |
| `eval.num_logged_samples` | 검증 때마다 예측 결과를 로그에 남길 검증 표본 수 |
| `log.journal_mode` | SQLite journal 모드. `"WAL"` 또는 `"DELETE"` (7.7절) |
| `submission.default_smiles` | 제출 파일에서 유효하지 않은 예측 대신 적을 SMILES. 유효한 SMILES여야 하며 빈 문자열은 쓸 수 없다. 학습에 쓰이지 않으므로 학습한 뒤에 바꾸어도 된다 |

`encoder`는 `type`에 따라 허용되는 키 집합이 다르다. 검증할 때 `type`에 맞는 키만 정확히 있어야 한다.

### 6.4 동작 확인용 설정 `configs/config000.json`

GPU 없이도 수 분 안에 전체 흐름(학습 → 모니터링 → 추론)이 끝나도록 만든다. config001과 다른 부분만 적으면 다음과 같다(파일에는 모든 키를 적는다).

| 키 | 값 |
|---|---|
| `data.val_size` | 64 |
| `data.max_train_samples` | 256 |
| `encoder` | `{"type": "scratch", "patch_size": 16, "dim": 128, "depth": 2, "num_heads": 4, "mlp_ratio": 4.0, "dropout": 0.0}` |
| `decoder` | `{"dim": 128, "depth": 2, "num_heads": 4, "mlp_ratio": 4.0, "dropout": 0.0}` |
| `train.batch_size` | 16 |
| `train.max_steps` | 200 |
| `train.lr` | 0.001 |
| `train.encoder_lr_scale` | 1.0 |
| `train.warmup_steps` | 20 |
| `train.amp` | false |
| `train.num_workers` | 0 |
| `train.log_every`, `train.eval_every`, `train.save_every` | 10, 50, 100 |
| `eval.num_logged_samples` | 8 |

참고로 이 크기(파라미터 약 110만 개)의 모델은 CPU 2코어에서 배치 16 기준 step당 약 0.13초였다.

## 7. 스크립트 명세

아래 이름과 동작은 모듈 사이, 그리고 노트북과의 약속이다. 내부 구현은 자유이나 공개 인터페이스는 지킨다. 인터페이스를 바꿔야 한다면 이유를 보고한다.

### 7.1 `config.py`

- `load_config(name: str) -> dict`: `configs/{name}.json`을 읽고 6절의 규칙으로 검증하여 돌려준다. `name`은 확장자 없는 파일 이름이다.
- `run_dir(name: str) -> pathlib.Path`: `outputs/{name}`을 돌려준다. 하위 경로(`ckpts`, `logs`, `predictions`)를 얻는 함수도 여기에 둔다.
- `save_snapshot(config, name)`, `load_snapshot(name)`: `outputs/{name}/config.json`을 쓰고 읽는다.
- `diff_config(a, b) -> list[str]`: 두 설정에서 값이 다른 키를 `"train.lr"`과 같은 점 표기로 돌려준다. 재개와 추론에서 설정 불일치를 알리는 데 쓴다. 학습 결과에 영향을 주지 않는 `submission`은 비교하지 않는다.

### 7.2 `tokenizer.py`

원자 단위 토크나이저이다. 아래 정규식으로 SMILES를 토큰으로 나눈다. 대괄호 원자(`[C@@H]`, `[nH]`, `[O-]` 등)는 통째로 한 토큰이고, `Cl`과 `Br`은 한 토큰이다.

```python
import re

SMILES_TOKEN_PATTERN = re.compile(
    r"\[[^\]]+\]|Br|Cl|%\d{2}|[BCNOSPFIbcnosp]|[=#\-+\\/().@:~*$]|\d"
)
```

- 분할 결과를 이어 붙였을 때 원래 문자열과 같지 않으면(정규식이 못 읽은 문자가 있으면) 오류를 낸다. 제공된 훈련 SMILES와 ZINC250k 전체는 이 정규식으로 빠짐없이 분할된다.
- 특수 토큰과 번호는 고정한다: `<pad>` 0, `<bos>` 1, `<eos>` 2, `<unk>` 3.
- 어휘는 `data/smiles_train.csv`의 SMILES와 ZINC250k의 SMILES에 나온 토큰의 합집합으로 만든다. 순서는 특수 토큰 4개 뒤에 나머지를 문자열 정렬 순으로 둔다. 따라서 어휘 크기는 89이다(특수 4 + 85).
  - 검증으로 떼어 낸 분자의 토큰도 어휘에 포함된다. 어휘는 입력 표기의 목록일 뿐이므로 문제가 되지 않는다.
  - ZINC250k를 포함하는 이유는 학생이 ZINC250k 기반 데이터를 추가할 때 어휘를 다시 만들 필요가 없게 하기 위함이다.
- `SmilesTokenizer` 클래스의 공개 인터페이스:
  - `SmilesTokenizer.build(smiles_iterable, max_length) -> SmilesTokenizer`
  - `tokenize(smiles) -> list[str]`
  - `encode(smiles) -> list[int]`: `[<bos>] + 토큰 + [<eos>]`. 어휘에 없는 토큰은 `<unk>`. 길이가 `max_length`를 넘으면 오류.
  - `decode(ids) -> str`: `<bos>`는 건너뛰고 첫 `<eos>`에서 멈추며 `<pad>`, `<unk>`는 버린다.
  - `save(path)`, `SmilesTokenizer.load(path)`: `vocab.json`으로 저장·복원. 파일에는 토큰 목록과 `max_length`를 담는다.
  - `vocab_size`, `pad_id`, `bos_id`, `eos_id`, `unk_id`

### 7.3 `datautils.py`

**분할.** `split_train_val(image_ids, val_size, seed) -> (train_ids, val_ids)`

1. `image_id`를 번호순으로 정렬한다(`CDK_Depict_10_101` → `(10, 101)`; 문자열 정렬이 아니다).
2. `random.Random(seed)`로 한 번 섞는다.
3. 앞의 `val_size`개가 검증, 나머지가 훈련이다.

단순 무작위 분할을 쓰는 이유는 Kaggle의 test도 같은 모집단에서 단순 무작위로 뽑혔기 때문이다. 검증 점수가 리더보드 점수와 같은 조건에서 측정된다. 층화나 그룹 분할은 하지 않는다.

분할 결과는 `outputs/{name}/split.json`에 `{"seed", "val_size", "train_ids", "val_ids"}`로 저장한다. 재개·모니터링·추론은 다시 계산하지 않고 이 파일을 읽는다.

**이미지 전처리.**

- PIL로 읽어 RGB로 바꾼다. 크기가 `image.size`와 다르면 bicubic으로 맞춘다.
- 정규화의 mean, std는 `network.image_normalization(config) -> (mean, std)`에서 얻는다. timm 인코더는 `timm.get_pretrained_cfg(name)`의 `mean`, `std`를(모델을 만들지 않고 얻을 수 있다), scratch 인코더는 0.5를 쓴다.
- `build_train_transform(config)`: 무작위 affine 변환(회전 ±`rotate_deg`, 평행이동 `translate`, 확대·축소 `scale`)을 적용하며, 빈 영역은 **흰색**으로 채운다. `rotate_deg`가 0이고 `translate`가 0이고 `scale`이 `[1.0, 1.0]`이면 증강 없이 동작해야 한다.
- `build_eval_transform(config)`: 증강 없이 크기 조정과 정규화만 한다. 검증과 추론에 쓴다.
- **좌우·상하 반전은 쓰지 않는다.** 거울상은 입체화학이 반대인 분자이므로 정답 SMILES와 어긋난다. 평면 내 회전은 분자를 바꾸지 않으므로 허용된다.

**Dataset과 DataLoader.**

- `OCSRDataset`: `(image_id, 이미지 경로, SMILES 또는 None)` 목록을 받아 `{"image": 텐서 (3, H, W), "input_ids": 텐서, "image_id": 문자열, "smiles": 문자열}`을 돌려준다. SMILES가 `None`이면(test) `input_ids`를 만들지 않는다.
- `collate_fn`: `input_ids`를 배치 내 최대 길이에 맞춰 `<pad>`로 채운다.
- `build_dataloaders(config, tokenizer, split) -> (train_loader, val_loader)`: 훈련 로더는 `shuffle=True`, `drop_last=True`. 검증 로더는 순서를 고정한다(`image_id` 번호순).
- `build_test_loader(config) -> test_loader`: `data/sample_submission.csv`의 `image_id`를 **그 파일의 순서대로** 읽고 `data/images_test`에서 이미지를 찾는다.
- Dataset과 `collate_fn`은 반드시 이 모듈에 정의한다(노트북 셀에 정의하면 DataLoader 워커에서 불러올 수 없는 환경이 있다).

**추가 훈련 데이터.** `data.extra_train_sources`의 각 원소 `{"csv", "image_dir"}`를 읽어 훈련 목록에 더한다.

- CSV 형식은 `smiles_train.csv`와 같다(`image_id,SMILES`). 이미지는 `{image_dir}/{image_id}.png`이다.
- SMILES는 `normalize_smiles`를 거친다. 정규화에 실패했거나, 어휘에 없는 토큰이 있거나, `max_length`를 넘거나, 이미지 파일이 없는 행은 제외하고, 사유별 제외 개수를 출력한다.
- 추가 데이터는 **훈련에만** 들어간다. 검증 세트는 항상 제공된 손그림 데이터에서만 만든다.
- 견본 설정에서는 빈 목록이다. 이 기능은 tests에서 작은 가짜 데이터로 점검한다.

### 7.4 `network.py`

전체 구조는 `이미지 → ViT 인코더 → 패치 표현 → GPT 디코더(cross-attention) → 다음 토큰 분포`이다.

**인코더.** `build_encoder(config) -> nn.Module`. 출력은 `(B, N, D_enc)`의 토큰열이다.

- `type == "timm"`: `timm.create_model(name, pretrained=pretrained, img_size=image.size, num_classes=0, drop_path_rate=...)`로 만들고 `forward_features`의 출력 전체(CLS 포함)를 쓴다. 입력 크기가 사전학습 해상도(224)와 다르므로 가중치를 불러올 때 위치 임베딩이 보간된다. **사전학습 가중치가 실제로 내려받아지고 256 입력으로 적재되는지는 확인하지 못했다.** 서버에서 확인하고, 내려받기에 실패하면 무작위 초기화로 조용히 넘어가지 말고 오류를 내어 보고한다.
- `type == "scratch"`: ViT를 직접 구현한다. Conv2d 패치 임베딩, 학습되는 위치 임베딩, pre-LN Transformer 블록(`self-attention → MLP`), 마지막 LayerNorm. CLS 토큰은 두지 않는다.

**디코더.** `GPTDecoder`를 직접 구현한다. `nn.TransformerDecoder`, `nn.MultiheadAttention`, Hugging Face 모델을 쓰지 않고, attention은 `F.scaled_dot_product_attention`을 사용하는 자체 `MultiHeadAttention` 클래스로 작성한다(강의에서 다룬 구조가 코드에서 그대로 보이도록 하기 위함).

- 입력: 토큰 임베딩 + 학습되는 위치 임베딩(길이 `tokenizer.max_length`).
- 블록(pre-LN) × `depth`: `masked self-attention → cross-attention(인코더 출력 참조) → MLP(GELU)`. 각 하위 층에 residual 연결과 dropout.
- self-attention에는 causal mask를 적용한다. 패딩 위치는 손실에서 제외하므로 별도 패딩 마스크는 필수가 아니다.
- 마지막 LayerNorm 뒤에 어휘 크기의 출력층을 두고, 출력층 가중치를 토큰 임베딩과 공유한다(weight tying).
- 인코더 출력 차원이 `decoder.dim`과 다르면 `nn.Linear`로 맞춘다.
- 디코더와 scratch 인코더의 `nn.Linear`, `nn.Embedding` 가중치와 위치 임베딩은 표준편차 0.02의 truncated normal로, bias는 0으로 초기화한다. 이렇게 하면 학습 시작 시 손실이 `ln(어휘 크기)`(어휘 89개이면 약 4.49) 근처에서 출발한다. 사전학습 인코더의 가중치는 다시 초기화하지 않는다.

**전체 모델.** `OCSRModel(encoder, decoder)`

- `forward(images, input_ids) -> logits`: `images (B, 3, H, W)`, `input_ids (B, T)` → `(B, T, V)`. teacher forcing용이다. 손실은 `input_ids[:, :-1]`을 넣어 `input_ids[:, 1:]`을 맞히도록 계산하고 `<pad>`는 무시한다.
- `generate(images, bos_id, eos_id, max_new_tokens) -> LongTensor (B, ≤max_new_tokens+1)`: 배치 greedy 디코딩. 인코더는 한 번만 실행한다. `<eos>`가 나온 표본은 이후 `<pad>`로 채우고, 모든 표본이 끝나면 일찍 멈춘다. `torch.no_grad`로 실행한다. 생성 길이는 `<bos>`를 포함하여 `tokenizer.max_length`를 넘지 않게 한다(위치 임베딩의 길이).
- `build_model(config, tokenizer, load_pretrained: bool = True) -> OCSRModel`: `load_pretrained=False`이면 config의 `encoder.pretrained`와 무관하게 사전학습 가중치를 내려받지 않는다. 체크포인트로 가중치를 덮어쓰는 경우(재개, 추론)에 쓴다.
- `image_normalization(config) -> (mean, std)`: 7.3절 참조.
- `count_parameters(model) -> dict`: 인코더, 디코더, 전체 파라미터 수.

참고: 견본 설정의 파라미터 수는 인코더 약 2,170만(확인됨), 디코더 약 1,400만(추정)이다.

### 7.5 `evalutils.py`

4.3절의 규칙을 구현한다.

- `tanimoto_similarity(true_smiles: str, pred_smiles) -> float`: 한 쌍의 점수.
- `score_predictions(true_smiles: list[str], pred_smiles: list) -> dict`: 아래를 돌려준다.
  - `tanimoto`: 평균 Tanimoto (무효 예측은 0). Kaggle 점수와 같은 정의이다.
  - `valid_ratio`: `normalize_smiles` 결과가 비어 있지 않은 예측의 비율.
  - `exact_match`: 정규화된 예측 문자열이 정답 문자열과 같은 비율(입체화학까지 일치해야 함).
  - `per_sample`: 표본별 `{"pred_normalized", "tanimoto", "is_valid"}` 목록.
- 길이가 다르면 `ValueError`. 정답을 읽지 못하면 `ValueError`.

### 7.6 `kaggleutils.py`

- `normalize_smiles(smiles) -> str`: 4.2절과 같다.
- `make_submission(image_ids, predicted_smiles, path, default_smiles) -> pandas.DataFrame`: 예측을 모두 `normalize_smiles`에 통과시켜 `image_id,SMILES` CSV로 저장한다. 무효 예측과 1,000자를 넘는 예측은 `default_smiles`를 정규화한 값으로 채운다. `default_smiles`가 유효하지 않으면 `ValueError`. 저장 후 전체 행 수와 기본값으로 채운 행 수를 출력한다. 줄바꿈은 LF로 고정한다.
- `validate_submission(path, sample_submission_path) -> None`: 아래를 검사하고 어긋나면 내용을 밝히는 오류를 낸다.
  - 열이 정확히 `image_id`, `SMILES`이다.
  - 행 수와 `image_id`의 순서가 `sample_submission.csv`와 같다.
  - 모든 `SMILES`가 비어 있지 않고, `normalize_smiles`를 적용해도 바뀌지 않는다.

### 7.7 `logutils.py`

학습 로그를 `outputs/{name}/logs/logs.sqlite3`에 기록하고 조회한다.

**스키마.**

```sql
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
```

- `runs`: 학습을 시작하거나 재개할 때마다 한 행이 추가된다. `status`는 `running`, `finished`, `interrupted`, `failed` 중 하나이다. `env_json`에는 Python·torch·timm·rdkit 버전과 장치 이름(GPU 모델명 또는 `cpu`)을 담는다. 시각은 ISO 8601 UTC 문자열이다.
- `scalars`: 스칼라 지표를 한 행에 하나씩 기록한다(긴 형식). 학생이 지표를 추가해도 스키마를 바꿀 필요가 없다. `split`은 `train` 또는 `val`이다.
- `samples`: 검증 때마다 고정된 검증 표본의 예측을 기록한다. `pred_smiles`에는 정규화 전의 모델 출력 문자열을 그대로 넣는다(무효 예측이 어떤 모양인지 볼 수 있도록).

**기록하는 지표.**

| split | name | 시점 | 내용 |
|---|---|---|---|
| train | `loss` | `log_every`마다 | 직전 구간의 평균 손실 |
| train | `lr` | `log_every`마다 | 디코더 학습률 |
| train | `grad_norm` | `log_every`마다 | clip 전 gradient norm의 구간 평균 |
| train | `samples_per_sec` | `log_every`마다 | 처리 속도 |
| val | `loss` | `eval_every`마다 | teacher forcing 손실 |
| val | `tanimoto` | `eval_every`마다 | 생성 결과의 평균 Tanimoto (모델 선택 기준) |
| val | `valid_ratio` | `eval_every`마다 | 생성 결과의 유효 비율 |
| val | `exact_match` | `eval_every`마다 | 생성 결과의 완전 일치율 |
| val | `eval_seconds` | `eval_every`마다 | 검증에 걸린 시간 |

**쓰기.** `TrainLogger` 클래스

- 학습 프로세스 안에서 연결 하나를 유지한다. 연결할 때 `PRAGMA journal_mode`를 config 값으로 설정하고 `timeout=30`을 준다.
- `start_run(config, start_step) -> run_id`, `log_scalars(step, split, values: dict)`, `log_samples(step, rows: list[dict])`, `end_run(status, end_step)`, `close()`.
- 기록 호출마다 곧바로 commit한다(트랜잭션을 길게 열어 두지 않는다). 같은 키를 다시 쓰면 덮어쓴다(`INSERT OR REPLACE`).

**읽기.** 모니터링 노트북이 학습 도중에 호출한다.

- `read_scalars(db_path) -> pandas.DataFrame` (열: `step`, `split`, `name`, `value`, `run_id`), `read_samples(db_path, step=None) -> DataFrame`, `read_runs(db_path) -> DataFrame`.
- 읽기 전용으로 연다: `sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)`.
- 재개 때문에 같은 `(step, split, name)`이 여러 실행에 기록될 수 있다. 조회 함수는 `run_id`가 가장 큰(가장 나중의) 값만 돌려준다.
- DB 파일이 아직 없으면 오류 대신 빈 DataFrame을 돌려준다.

**journal 모드.** 위 구성(WAL, 호출마다 commit, 읽기 전용 조회)은 로컬 디스크에서 쓰기 400회와 조회 200회를 동시에 실행해 오류가 없음을 확인했다. 다만 SQLite의 WAL 모드는 NFS 같은 네트워크 파일시스템에서 동작이 보장되지 않는다. JupyterHub의 홈 디렉토리가 네트워크 파일시스템인지는 확인하지 못했다. 개발 초기에 `df -T .`로 파일시스템 종류를 확인하고 10.2절의 동시 접근 점검을 서버에서 실행한다. 실패하면 두 config의 `log.journal_mode`를 `"DELETE"`로 바꾸고 보고한다.

### 7.8 `trainer.py`

**진입점.** `fit(config_name: str, resume: bool = False) -> dict`

노트북은 이 함수 하나를 호출한다. 반환값은 `{"best_step", "best_tanimoto", "last_step", "run_dir"}`이다.

**시작 절차.**

1. `load_config(config_name)`.
2. 실행 디렉토리 상태에 따라 분기한다.

   | `outputs/{name}`의 상태 | `resume=False` | `resume=True` |
   |---|---|---|
   | 없음, 또는 체크포인트와 로그가 모두 없음 | 새로 시작 | 새로 시작 |
   | `ckpts/last.pth`가 있음 | **오류**: 이미 학습된 실행임을 알리고, 이어서 학습하려면 `RESUME = True`, 새로 하려면 디렉토리를 직접 지우거나 다른 config 이름을 쓰라고 안내 | `last.pth`에서 재개 |
   | 로그나 다른 체크포인트는 있으나 `last.pth`가 없음 | 위와 같은 오류 | **오류**: 재개할 체크포인트가 없음 |

   코드가 기존 실행 디렉토리를 자동으로 지우는 일은 없어야 한다.
3. 새로 시작할 때: 시드 설정 → 설정 사본 저장 → 어휘 생성·저장 → 분할 생성·저장.
   재개할 때: 설정 사본과 현재 config를 `diff_config`로 비교하여 다르면 다른 키를 나열하는 오류를 낸다. 어휘와 분할은 저장된 파일에서 읽고, 모델은 `load_pretrained=False`로 만든 뒤 `last.pth`를 불러온다.
4. 장치는 CUDA가 있으면 CUDA, 없으면 CPU이다.

**학습 루프.** step 기준이다(epoch은 계산해서 표시만 한다).

- 옵티마이저: AdamW. 파라미터 그룹은 인코더(`lr × encoder_lr_scale`)와 나머지(`lr`)로 나눈다.
- 학습률: `warmup_steps` 동안 `warmup_init_lr`에서 선형 증가, 이후 `max_steps`까지 cosine으로 `lr × min_lr_ratio`까지 감소.
- 손실: `<pad>`를 무시하는 cross-entropy, `label_smoothing` 적용.
- `amp`가 켜져 있고 CUDA이면 `torch.amp.autocast`와 `torch.amp.GradScaler`를 쓴다.
- gradient clipping은 `grad_clip`으로 한다.
- 진행 표시는 `tqdm.auto`를 쓴다.
- `eval_every`마다 검증하고, `save_every`마다 주기 체크포인트를 저장한다. `max_steps`에 도달하면 마지막 검증과 저장을 하고 끝낸다.
- `KeyboardInterrupt`(노트북의 중단 버튼)를 받으면 `last.pth`를 저장하고 `runs.status`를 `interrupted`로 기록한 뒤 끝낸다. 그 밖의 예외는 `failed`로 기록하고 다시 던진다.

**검증.** `evaluate(model, val_loader, tokenizer, device, amp) -> (metrics: dict, per_sample: list[dict])`

- 검증 세트 전체에 대해 (1) teacher forcing 손실과 (2) greedy 생성 결과의 지표(`score_predictions`)를 계산한다.
- 모델 선택 기준은 `val/tanimoto`이다. teacher forcing 손실이나 토큰 정확도로 고르지 않는다(생성 품질과 다르게 움직일 수 있다).
- 로그에 남기는 표본은 검증 세트를 `image_id` 번호순으로 정렬했을 때 앞의 `num_logged_samples`개로 고정한다. 매번 같은 분자를 기록해야 시간에 따른 변화를 볼 수 있다.

**예측.** `predict_smiles(model, loader, tokenizer, device, amp) -> list[str]`: 로더의 순서대로 greedy 생성한 문자열(정규화 전)을 돌려준다. 검증과 추론 노트북이 함께 쓴다.

**체크포인트.** `torch.load(path, weights_only=True)`로 읽을 수 있도록 텐서와 기본 자료형만 담는다.

| 파일 | 저장 시점 | 내용 |
|---|---|---|
| `last.pth` | 검증할 때마다, 주기 저장 때마다, 종료·중단 시 | `model`, `optimizer`, `scheduler`, `scaler`의 state_dict, `step`, `best_step`, `best_tanimoto`, `config` |
| `best.pth` | `val/tanimoto`가 이전 최고를 **초과**했을 때 | `model` state_dict, `step`, `metrics`(그 시점의 검증 지표), `config` |
| `step{step:06d}.pth` | `save_every`마다 | `model` state_dict, `step`, `config` |

- 저장은 임시 파일에 쓴 뒤 `os.replace`로 바꿔치기한다(저장 도중 중단되어도 기존 파일이 깨지지 않도록).
- 주기 체크포인트는 최근 `keep_last_k`개만 남기고 오래된 것부터 지운다. `last.pth`와 `best.pth`는 지우지 않는다.
- `load_checkpoint(config_name, which) -> dict`: `which`는 `"best"`, `"last"`, `"step001000"` 같은 이름이다. 없는 이름이면 사용 가능한 체크포인트 목록을 알리는 오류를 낸다.
- 재개 후의 결과가 중단 없이 학습한 결과와 비트 단위로 같을 필요는 없다. 재개할 때는 시드를 `seed + step`으로 다시 설정한다.

참고(추정): 견본 설정의 파일 크기는 가중치만 약 140 MB, `last.pth`는 옵티마이저 상태를 포함하여 약 410 MB이다.

## 8. 노트북 명세

세 노트북 모두 첫 코드 셀에 사용자가 고칠 변수만 모아 둔다. 그 아래 셀은 고치지 않고 실행할 수 있어야 한다. 각 코드 셀 앞에 무엇을 하는 셀인지 설명하는 마크다운 셀을 둔다.

### 8.1 `ocsr_train.ipynb`

| 순서 | 내용 |
|---|---|
| 1 | 설정 변수: `CONFIG_NAME = "config001"`, `RESUME = False` |
| 2 | import, config 로드, 설정 내용 출력 |
| 3 | 환경 확인: 패키지 버전, 장치(GPU 이름과 메모리) |
| 4 | 데이터 확인: 토크나이저 어휘 크기, train/val 개수, 증강이 적용된 훈련 이미지 몇 장과 그 SMILES·토큰열 표시 |
| 5 | 모델 확인: 구조 요약과 파라미터 수 |
| 6 | 학습: `trainer.fit(CONFIG_NAME, resume=RESUME)` |
| 7 | 결과 요약: 최고 검증 Tanimoto와 그 step, 체크포인트·로그 경로 |

4와 5의 확인 셀은 `fit`이 실행 디렉토리를 만들기 전에 실행되므로, 실행 디렉토리에 아무것도 쓰지 않는다.

### 8.2 `ocsr_train_monitoring.ipynb`

학습 중에도, 학습이 끝난 뒤에도 실행할 수 있어야 한다. 셀을 다시 실행하면 최신 상태가 반영된다.

| 순서 | 내용 |
|---|---|
| 1 | 설정 변수: `CONFIG_NAMES = ["config001"]` (여러 개를 적으면 한 그림에 겹쳐 비교), `SAMPLE_STEP = None` (`None`이면 가장 최근 검증 step) |
| 2 | 실행 이력 표(`runs`): 시작·종료 시각, 상태, step 범위 |
| 3 | 요약 표: config별 현재 step, 최근 훈련 손실, 최근 검증 지표, 최고 검증 Tanimoto와 그 step |
| 4 | 손실 곡선: 훈련 손실과 검증 손실 → `logs/plot_loss.png` |
| 5 | 검증 지표 곡선: `tanimoto`, `valid_ratio`, `exact_match` → `logs/plot_metrics.png` |
| 6 | 학습률 곡선 → `logs/plot_lr.png` |
| 7 | 예측 표본: `SAMPLE_STEP`에서 기록된 표본의 표(정답, 예측, Tanimoto, 유효 여부)와, 몇 개에 대해 입력 이미지·정답 구조·예측 구조를 나란히 그린 그림. 구조 그림은 RDKit `Draw`로 그리고, 무효 예측은 빈 칸에 "invalid"로 표시 |

- DB는 `logutils`의 읽기 함수로만 접근한다.
- 그림 파일은 `CONFIG_NAMES`의 첫 번째 실행의 `logs/`에 저장한다.
- 아직 로그가 없는 config는 오류 없이 "로그 없음"으로 표시한다.

### 8.3 `ocsr_inference.ipynb`

| 순서 | 내용 |
|---|---|
| 1 | 설정 변수: `CONFIG_NAME = "config001"`, `CHECKPOINT = "best"` (`"last"` 또는 `"step001000"` 형태도 가능) |
| 2 | `configs/{CONFIG_NAME}.json`을 읽고 `outputs/{CONFIG_NAME}/config.json`(학습 당시 사본)과 비교한다. 다르면 다른 키를 경고로 보여 주고, **모델은 학습 당시 사본으로** 만든다 |
| 3 | 저장된 어휘로 토크나이저를 복원하고, 모델을 만들어 체크포인트를 불러온다. 체크포인트의 step과 저장 당시 검증 지표를 출력한다 |
| 4 | (점검) 검증 세트를 다시 예측하여 점수를 계산하고, 체크포인트에 기록된 검증 지표와 나란히 보여 준다. 혼합 정밀도 때문에 끝자리가 조금 다를 수 있다 |
| 5 | test 예측: `build_test_loader` → `predict_smiles` |
| 6 | 제출 파일 저장: `make_submission` → `outputs/{CONFIG_NAME}/predictions/submission_{CHECKPOINT}.csv`(무효 예측은 현재 config의 `submission.default_smiles`로 채움), 이어서 `validate_submission` |
| 7 | 결과 확인: 유효 예측 비율(기본값으로 채우기 전 기준), test 이미지 몇 장과 예측 구조 그림 |

- 이 노트북은 test 정답 없이 동작해야 한다.
- 인코더를 만들 때 사전학습 가중치를 다시 내려받지 않는다(곧바로 체크포인트로 덮어쓰므로 `build_model(..., load_pretrained=False)`로 만든다).

## 9. 실행 디렉토리 요약

| 파일 | 만드는 주체 | 읽는 주체 |
|---|---|---|
| `config.json` | `trainer.fit` (새로 시작할 때) | 재개 시 `fit`, 추론 노트북 |
| `vocab.json` | `trainer.fit` | 재개 시 `fit`, 추론 노트북 |
| `split.json` | `trainer.fit` | 재개 시 `fit`, 추론 노트북(검증 점검) |
| `ckpts/*.pth` | `trainer.fit` | 재개 시 `fit`, 추론 노트북 |
| `logs/logs.sqlite3` | `trainer.fit` | 모니터링 노트북 |
| `logs/plot_*.png` | 모니터링 노트북 | — |
| `predictions/submission_*.csv` | 추론 노트북 | Kaggle 제출 |

## 10. 검증

### 10.1 자동 점검 (`tests/`)

`python -m unittest discover -s tests -v`로 실행한다(작업 디렉토리 `main/`). GPU 없이 1분 안팎에 끝나야 하며, `outputs/`에 흔적을 남기지 않는다(임시 디렉토리 사용).

| 파일 | 점검 내용 |
|---|---|
| `test_tokenizer.py` | 훈련 SMILES와 ZINC250k 전체에서 토큰 분할을 이어 붙이면 원문과 같다. `decode(encode(s)) == s`. 어휘 크기 89. 특수 토큰 번호. 저장·복원 후 동일. 정규식이 못 읽는 문자에서 오류 |
| `test_metrics.py` | 4.4절의 15쌍을 행별로, 그리고 집계값 3개를 소수 여섯째 자리까지 재현. `normalize_smiles`가 훈련 SMILES 전체를 바꾸지 않음. `make_submission` → `validate_submission` 왕복. 잘못된 제출 파일(열 이름, 행 수, 순서, 정규화되지 않은 SMILES)에서 오류 |
| `test_config.py` | `configs/`의 모든 파일이 검증을 통과. 키 누락·모르는 키·`encoder.type`에 맞지 않는 키에서 오류. `diff_config` |
| `test_datautils.py` | 분할이 시드에 대해 재현되고 서로 겹치지 않으며 합치면 전체. 변환 결과의 shape과 값 범위. 증강 끈 변환이 평가용 변환과 같은 결과. 추가 훈련 데이터의 제외 규칙(가짜 데이터) |
| `test_network.py` | 소형 scratch 모델로: `forward` 출력 shape, causal 성질(뒤쪽 토큰을 바꿔도 앞쪽 logits가 변하지 않음), `generate`가 `<eos>` 이후를 `<pad>`로 채움, 초기 손실이 `ln(어휘 크기)`의 ±0.5 이내, 체크포인트 저장 후 `weights_only=True`로 복원하면 출력이 동일 |
| `test_logutils.py` | 기록·조회 왕복. 같은 step을 나중 실행이 다시 기록하면 나중 값만 조회됨. DB가 없을 때 빈 DataFrame |

timm 사전학습 가중치 내려받기는 자동 점검에 넣지 않는다(네트워크에 의존하므로).

### 10.2 실행 확인

아래는 서버에서 직접 실행하여 확인하고 결과를 보고한다.

1. **과적합 점검.** 소형 scratch 모델(config000의 모델 크기)을 훈련 표본 16개에 증강 없이, 학습률 0.001, 배치 16으로 600 step 학습한다. 손실이 0.05 아래로 내려가고, 그 16개에 대한 greedy 생성의 완전 일치율이 0.9 이상이어야 한다. 미달하면 손실의 입력·목표 정렬(한 칸 밀기), causal mask, 위치 임베딩, `generate`를 점검한다. 참고로 이 명세를 쓰면서 같은 구조를 RDKit으로 그린 이미지 16장에 시험했을 때 600 step에서 손실 0.001, 완전 일치율 1.0이었다.
2. **전체 흐름(config000).** `ocsr_train.ipynb` → `ocsr_train_monitoring.ipynb` → `ocsr_inference.ipynb`를 `CONFIG_NAME = "config000"`으로 처음부터 끝까지 실행한다. 300행의 제출 파일이 만들어지고 `validate_submission`을 통과해야 한다. 이 설정의 점수는 낮아도 된다.
3. **중단과 재개.** config000 학습을 도중에 중단한 뒤 `RESUME = True`로 다시 실행한다. step이 이어지고, `runs`에 두 행이 생기며, 모니터링 곡선이 끊기지 않아야 한다. `RESUME = False`로 다시 실행하면 7.8절의 오류가 나야 한다.
4. **동시 접근.** 학습이 도는 동안 모니터링 노트북을 여러 번 실행하여 `database is locked` 같은 오류가 없는지 확인한다(7.7절).
5. **사전학습 인코더.** config001로 모델을 만들어 가중치 내려받기, 256 입력 적재, 한 배치의 forward·backward가 되는지 확인한다.
6. **견본 학습(config001).** GPU를 쓸 수 있으면 끝까지 학습하고 12.2절의 표를 채운다.

## 11. 개발 순서

1. 환경 확인: 패키지 버전, GPU 유무, 파일시스템 종류, 인터넷 접근(timm 가중치), 이미지 전체의 크기·모드 집계. 3절과 다른 점이 있으면 먼저 보고한다.
2. `tokenizer.py`, `kaggleutils.py`, `evalutils.py`와 해당 테스트. 4.4절의 기준값을 재현하는지 여기서 확정한다.
3. `config.py`, 두 config 파일, 해당 테스트.
4. `datautils.py`와 테스트.
5. `network.py`와 테스트, 과적합 점검(10.2절 1번).
6. `logutils.py`와 테스트.
7. `trainer.py`.
8. 노트북 3종.
9. 실행 확인(10.2절 2~5번).
10. 견본 학습(10.2절 6번)과 결과 기록.
11. 마무리 작업(12.3절).

## 12. 완료 기준과 마무리

### 12.1 완료 기준

- 10.1절의 자동 점검이 모두 통과한다.
- 10.2절의 1~5번이 확인되었다.
- 노트북 3종이 출력을 비운 상태로 저장되어 있고, 위에서부터 순서대로 실행하면 오류 없이 끝난다.
- `data/`가 변경되지 않았다.
- 1.3절에서 포함하지 않기로 한 기능이 구현되어 있지 않다.

### 12.2 견본 결과 기록

견본 학습을 마치면 아래 표를 **측정값으로만** 채운다. 실행하지 않은 항목은 "미측정"으로 둔다. 추정값이나 예상값을 적지 않는다.

| 항목 | config000 | config001 |
|---|---|---|
| 장치(GPU 모델) | | |
| 학습 시간 | | |
| 최대 GPU 메모리 | | |
| 최고 검증 Tanimoto (step) | | |
| 그때의 유효 비율 / 완전 일치율 | | |
| 체크포인트 크기 (`best.pth` / `last.pth`) | | |
| 초깃값에서 바꾼 설정과 이유 | | |

### 12.3 마무리 작업

개발과 확인이 끝나면 다음을 수행한다.

1. 이 "개발 명세"를 `docs/development_spec.md`로 옮긴다.
2. README에는 맨 위의 소개 문단을 그대로 두고, 그 아래에 학생용 안내를 작성한다: 폴더 구조, 실행 순서(학습 → 모니터링 → 추론 → 제출), config 키 설명, 학생이 확장할 지점(1.4절), 평가 방법 요약(4절), 데이터 출처.
3. 12.2절의 결과 표는 담당 교수에게 보고하고, README에 실을지는 담당 교수가 정한다.

## 13. 확인되지 않은 사항과 보고

이 명세는 실제 서버를 보지 않고 작성되었다. 아래는 확인되지 않은 사항이며, 개발 중에 확인하여 결과를 보고한다.

| 항목 | 확인 방법 | 어긋날 때 |
|---|---|---|
| 이미지가 모두 256×256 RGB인지 | 전체 집계 | 분포를 보고하고 전처리 방식에 대한 지시를 받는다 |
| timm 사전학습 가중치를 내려받을 수 있는지 | 10.2절 5번 | 보고한다. 무작위 초기화로 대체하지 않는다 |
| 256 입력으로 사전학습 가중치가 적재되는지 | 10.2절 5번 | 보고한다 |
| 홈 디렉토리에서 WAL 모드가 동작하는지 | 10.2절 4번 | `journal_mode`를 `"DELETE"`로 바꾸고 보고한다 |
| GPU 종류, 메모리, 공유 여부 | `nvidia-smi` | 견본 설정의 배치 크기를 조정하고 보고한다 |
| 견본 설정의 하이퍼파라미터가 적절한지 | 10.2절 6번 | 조정하고 12.2절에 기록한다 |
| 디코더 파라미터 수, 체크포인트 크기 추정값 | 실측 | 12.2절에 실측값을 적는다 |

명세와 다르게 구현해야 할 이유가 생기면, 구현을 바꾸기 전에 무엇을 왜 바꾸려는지 보고한다. 특히 4절(평가 규칙)과 2.2절(제약)은 보고 없이 바꾸지 않는다.

## 데이터 출처

- 손그림 이미지와 정답 SMILES는 DECIMER hand-drawn molecule images dataset의 일부를 선별하여 재구성한 것이다. 원 데이터는 CC-BY 4.0으로 공개되어 있다.
  Brinkhaus, H. O., Zielesny, A., Steinbeck, C., Rajan, K. DECIMER—hand-drawn molecule images dataset. *Journal of Cheminformatics* 14, 36 (2022). https://doi.org/10.1186/s13321-022-00620-9
- `250k_rndm_zinc_drugs_clean_3.csv`는 ZINC 데이터베이스에서 추출된 약 25만 개 분자 목록으로, `aspuru-guzik-group/chemical_vae` 저장소에 공개된 파일에서 위 손그림 데이터와 중복되는 분자를 제거하고 표기를 통일한 것이다.
  Gómez-Bombarelli, R. et al. Automatic chemical design using a data-driven continuous representation of molecules. *ACS Central Science* 4, 268–276 (2018). https://doi.org/10.1021/acscentsci.7b00572
