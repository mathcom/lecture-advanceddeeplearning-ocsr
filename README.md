# lecture-advanceddeeplearning-ocsr

한림대학교 고급딥러닝 기말프로젝트

본 자료는 한림대학교 강의(과목명: 고급딥러닝)을 위해 개설되었습니다.

본 자료의 저작권은 한림대학교 최종환교수(jonghwanc@hallym.ac.kr)에게 있으며, 문의사항이 있는 경우 issues 또는 이메일을 통해 연락주시기 바랍니다.

---


손으로 그린 분자 구조 이미지를 읽어 SMILES 문자열을 출력하는 OCSR(Optical Chemical Structure Recognition) 모델을 만드는 프로젝트입니다. 이 저장소에는 출발점이 되는 **견본 모델**이 들어 있습니다. 사전학습 ViT 인코더와 직접 구현한 GPT 디코더로 이루어져 있으며, 여러분이 개선할 여지를 남기도록 단순하게 만들었습니다.

견본의 설계와 세부 규칙은 [docs/development_spec.md](docs/development_spec.md)(이하 "명세")에 있습니다.

## 1. 폴더 구조

```
├ data/                          # 제공 데이터 (읽기 전용, 고치지 마세요)
├ configs/                       # 설정 파일
│ ├ config000.json               # 동작 확인용 소형 모델 (GPU 없이 실행 가능)
│ ├ config001.json               # 견본 (ViT-Small 인코더)
│ └ config002.json               # 견본의 인코더를 ViT-Base로 바꾼 것 (프로젝트 기준점)
├ scripts/
│ ├ config.py                    # 설정 읽기와 검증, 실행 디렉토리 경로
│ ├ tokenizer.py                 # SMILES 토크나이저
│ ├ datautils.py                 # 데이터, 이미지 변환, DataLoader
│ ├ network.py                   # ViT 인코더, GPT 디코더, 생성
│ ├ trainer.py                   # 학습, 검증, 체크포인트, 예측
│ ├ evalutils.py                 # 점수 계산, 구조 그림
│ ├ logutils.py                  # 학습 로그 기록과 조회, 학습 곡선
│ └ kaggleutils.py               # SMILES 정규화, 제출 파일
├ docs/development_spec.md       # 견본 모델의 개발 명세
├ ocsr_train.ipynb               # 학습
├ ocsr_train_monitoring.ipynb    # 학습 진행 상황
├ ocsr_inference.ipynb           # 예측과 제출 파일 생성
├ environment.yml                # 패키지 목록
└ outputs/{config 이름}/         # 실행하면 생기는 폴더 (git에 올라가지 않음)
  ├ config.json, vocab.json, split.json
  ├ ckpts/                       # best.pth, last.pth, stepNNNNNN.pth
  ├ logs/                        # logs.sqlite3, plot_*.png
  └ predictions/                 # submission_*.csv
```

## 2. 환경 준비

```bash
conda env create -f environment.yml
conda activate ocsr
python -m ipykernel install --user --name ocsr --display-name "Python (ocsr)"
```

JupyterLab에서 노트북을 열고 커널을 **Python (ocsr)** 로 선택합니다. 기본 `python3` 커널에는 필요한 패키지가 없을 수 있습니다.

환경이 제대로 준비되었는지는 `ocsr_train.ipynb`를 `CONFIG_NAME = "config000"`으로 실행해 확인합니다. 작은 모델이라 GPU가 있으면 1분 안에 끝납니다.

## 3. 실행 순서

노트북은 이 README가 있는 폴더를 작업 디렉토리로 하여 실행합니다. 각 노트북은 첫 코드 셀의 변수만 고치면 됩니다.

1. **학습** `ocsr_train.ipynb`: `CONFIG_NAME`을 정하고 실행합니다. 처음에는 `"config000"`으로 전체 흐름을 확인하세요. config001과 config002는 학습 시간이 깁니다(7절). 중단 버튼으로 멈춘 뒤 `RESUME = True`로 이어서 학습할 수 있습니다.
2. **모니터링** `ocsr_train_monitoring.ipynb`: 학습 도중에도 실행할 수 있습니다. 여러 config를 겹쳐 비교할 수 있습니다.
3. **추론** `ocsr_inference.ipynb`: 체크포인트로 test 이미지를 예측하여 `outputs/{config 이름}/predictions/submission_{체크포인트}.csv`를 만듭니다.
4. **제출**: 이 CSV를 Kaggle에 올립니다.

**하나의 config는 하나의 실행에 대응합니다.** 조건을 바꾸어 학습하려면 config 파일을 복사해 새 이름으로 만드세요. 이미 학습한 config를 `RESUME = False`로 다시 실행하면 오류가 나며, 기존 결과는 자동으로 지워지지 않습니다.

## 4. 설정 키

설정 파일에는 모든 키를 적어야 합니다. 빠진 키나 모르는 키가 있으면 어느 키가 문제인지 알려 주는 오류가 납니다.

| 키 | 의미 |
|---|---|
| `seed` | 난수 시드. train/val 분할, 가중치 초기화, 데이터 순서에 쓰인다 |
| `data.data_dir` | 데이터 폴더. 그 안의 파일 이름(`smiles_train.csv` 등)은 `datautils.py`의 상수로 둔다 |
| `data.val_size` | 훈련 데이터에서 떼어 낼 검증 분자 수 |
| `data.max_train_samples` | `null`이면 전부 사용. 정수이면 `split.json`의 `train_ids` 순서에서 앞의 그 개수만 사용(동작 확인용) |
| `data.extra_train_sources` | 추가 훈련 데이터 목록. 각 원소는 `{"csv": 경로, "image_dir": 경로}` (명세 7.3절) |
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
| `log.journal_mode` | SQLite journal 모드. `"WAL"` 또는 `"DELETE"` (명세 7.7절) |

## 5. 확장할 지점

견본의 다른 부분을 고치지 않고도 아래 세 곳에 자신의 방법을 끼워 넣을 수 있습니다.

1. **추가 훈련 데이터**: config의 `data.extra_train_sources`에 `{"csv": 경로, "image_dir": 경로}`를 적으면 훈련 데이터에 합쳐집니다. CSV는 `smiles_train.csv`와 같은 형식(`image_id,SMILES`)이고 이미지는 `{image_dir}/{image_id}.png`입니다. 직접 생성한 이미지를 넣는 통로이며, 쓸 수 없는 행은 사유별 개수를 출력하고 제외합니다. 추가 데이터는 검증에 들어가지 않습니다.
2. **이미지 증강**: `scripts/datautils.py`의 `build_train_transform`. 좌우·상하 반전은 입체화학이 반대인 분자가 되므로 쓰면 안 됩니다.
3. **디코딩**: `scripts/network.py`의 `OCSRModel.generate`. 견본은 greedy 디코딩입니다.

## 6. 평가 방법 요약

- 점수는 정답 SMILES와의 **평균 Tanimoto similarity**입니다. Morgan fingerprint(반지름 2, 2048비트, 입체화학 미반영)를 씁니다.
- 예측이 비어 있거나, 유효한 SMILES가 아니거나, 1,000자를 넘으면 그 행은 0점입니다. wildcard 원자(`*`), 명시적 방향족 결합(`:`), 배위 결합(`->`, `<-`)은 지원되지 않습니다.
- 채점기는 SMILES를 적힌 그대로 읽으므로 **RDKit 표준 SMILES로 정규화해서** 제출해야 합니다. 추론 노트북의 `make_submission`이 정규화합니다.
- 제출 파일은 `image_id,SMILES` 두 열이고, `sample_submission.csv`와 같은 300개 `image_id`를 담습니다.
- 학습 중의 검증 지표 `tanimoto`는 Kaggle 점수와 같은 정의이며, 가장 높은 시점의 모델이 `best.pth`로 저장됩니다.

자세한 규칙과 기준값은 명세 4절에 있습니다.

## 7. 견본 결과

**이 프로젝트의 기준점(baseline)은 config002입니다.**

아래는 제공된 손그림 훈련 이미지만으로 각 설정을 한 번씩(시드 42) 학습하여 측정한 값입니다(사전학습 인코더는 ImageNet 가중치에서 시작). 검증 세트는 config000이 64장, 나머지가 300장이며, 모두 훈련 데이터에서 무작위로 떼어 낸 것입니다. **Kaggle 점수가 아닙니다.**

| 항목 | config000 | config001 | config002 |
|---|---|---|---|
| 인코더 | 소형 ViT(직접 구현) | ViT-Small(사전학습) | ViT-Base(사전학습) |
| 파라미터 수(인코더 / 디코더) | 0.53M / 0.56M | 21.69M / 14.28M | 85.84M / 14.58M |
| 장치 | RTX 5000 Ada | RTX 5000 Ada | RTX 5000 Ada |
| 학습 시간 | 16.8초 | 32.5분 | 63.3분 |
| 최대 GPU 메모리(할당) | 202 MiB | 2,411 MiB | 4,812 MiB |
| 최고 검증 Tanimoto (step) | 0.0522 (100) | 0.3608 (17,000) | **0.5156** (19,500) |
| 그때의 유효 비율 / 완전 일치율 | 1.000 / 0.000 | 0.710 / 0.133 | 0.763 / 0.273 |
| `best.pth` / `last.pth` 크기 | 4.4 / 13.1 MB | 144.0 / 432.1 MB | 401.8 / 1,205.5 MB |

- config002는 config001에서 `encoder.name` 하나만 바꾼 것입니다. 나머지 조건은 같습니다.
- 검증 표본이 적어 점수가 출렁입니다. 검증 때마다의 최고점을 골랐으므로 최고값은 조금 낙관적일 수 있습니다. 마지막 step(20,000)의 검증 Tanimoto는 config001이 0.3499, config002가 0.5079입니다.
- 두 설정 모두 훈련 손실이 0에 가까워지는 동안 검증 손실은 초반(step 2,000~2,500)에 최저가 된 뒤 올라갔습니다.
- config002의 체크포인트는 모두 합해 약 2.8 GB입니다. 저장 공간을 확인하고 학습하세요.

## 8. 데이터와 가중치 출처

이 저장소의 `data/`는 아래 공개 자료를 수업용으로 가공한 것입니다. 데이터를 다시 배포하거나 다른 곳에 쓸 때는 원 출처의 조건을 따르고 아래 문헌을 인용하세요.

**손그림 이미지와 정답 SMILES** (`images_train/`, `images_test/`, `smiles_train.csv`)

- DECIMER hand-drawn molecule images dataset의 일부를 **선별하여 재구성**한 것입니다. 원 데이터는 [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)으로 공개되어 있습니다([Zenodo](https://doi.org/10.5281/zenodo.7617107)).
- Brinkhaus, H. O., Zielesny, A., Steinbeck, C., Rajan, K. DECIMER—hand-drawn molecule images dataset. *Journal of Cheminformatics* 14, 36 (2022). https://doi.org/10.1186/s13321-022-00620-9

**추가 분자 목록** (`250k_rndm_zinc_drugs_clean_3.csv`)

- [`aspuru-guzik-group/chemical_vae`](https://github.com/aspuru-guzik-group/chemical_vae) 저장소에서 제공하는 ZINC250k 데이터(ZINC 데이터베이스에서 추출된 약 25만 개 분자 목록)를 **본 프로젝트에 맞추어 가공한 것**입니다. 위 손그림 데이터와 중복되는 분자를 제거하고 SMILES 표기를 통일했습니다. 따라서 원본 파일과 내용이 다릅니다.
- Gómez-Bombarelli, R. et al. Automatic chemical design using a data-driven continuous representation of molecules. *ACS Central Science* 4, 268–276 (2018). https://doi.org/10.1021/acscentsci.7b00572
- ZINC: Irwin, J. J., Sterling, T., Mysinger, M. M., Bolstad, E. S., Coleman, R. G. *Journal of Chemical Information and Modeling* (2012). https://doi.org/10.1021/ci3001277 (이용 조건: [UCSF ZINC License](https://wiki.docking.org/index.php/UCSF_ZINC_License))

**사전학습 가중치** (저장소에 포함되지 않으며 처음 실행할 때 내려받습니다)

- config001: [`timm/vit_small_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k), config002: [`timm/vit_base_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k). ImageNet으로 학습한 범용 가중치입니다.
