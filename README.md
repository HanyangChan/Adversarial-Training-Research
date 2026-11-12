# TDAT CIFAR-10 연구 프로젝트

AAAI 2024 TDAT 공식 구현은 `vendor/TDAT`, 이식 가능한 연구 코드는 `tdat/`에 있습니다. HW2의 데이터·실험 기록 요구사항을 반영한 코드와 실험 절차이며 과제 보고서 본문은 포함하지 않습니다. **실제 CIFAR 학습과 논문 Baseline 재현은 수행하지 않았습니다.**

[원본 코드·논문 차이점](docs/PROVENANCE.md), [데이터·분할·라이선스 확인 상태](docs/DATASET.md), [테스트 결과](docs/VALIDATION.md)를 먼저 확인하세요. 공개 코드의 `tan`과 batch momentum 덮어쓰기 동작은 보존했습니다. 원본 파일 자체는 번역하거나 수정하지 않았습니다.

## 설치

Python 3.10–3.12를 권장합니다. 현재 검증 환경은 Linux Python 3.12 CPU입니다. 저장소 루트에서 실행합니다.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell에서는 다음 명령 사용
# .venv\Scripts\Activate.ps1
```

Linux CPU:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Windows/Linux NVIDIA CUDA 12.1: 호환되는 NVIDIA 드라이버가 필요합니다.

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

macOS Apple Silicon MPS:

```bash
python -m pip install -r requirements.txt
```

AutoAttack은 공식 소스의 특정 커밋을 사용하는 선택 의존성입니다.

```bash
python -m pip install -r requirements-autoattack.txt
```

실제 원본 ResNet18을 사용하는 CPU synthetic 검증입니다. 데이터 다운로드와 GPU 학습은 실행하지 않습니다.

```bash
python -m pytest -q -s tests
```

현재 클라우드에서는 `source /workspace/tdat-venv/bin/activate`로 준비된 환경을 사용할 수 있습니다.

## 학습 없이 데이터 준비

```bash
python -m tdat.prepare_data --download
```

공식 압축파일 다운로드·체크섬, 전체 학습/테스트 분할·클래스 수, 전처리와 10% 표본을 검증합니다. 결과는 `runs/data-validation.json`에 저장하며 학습은 실행하지 않습니다. 이미 준비된 데이터는 `--download` 없이 다시 확인합니다.

## 15분 한도 환경·데이터·모델 진단

로컬 3060 Ti에서 CUDA 의존성을 설치하고 데이터를 준비한 뒤 실행합니다.

```bash
python -m tdat.preflight --device cuda --batch-size 128 --steps 3 --max-seconds 900 --output runs/preflight-3060ti
```

실제 CIFAR 학습/테스트에서 첫 batch만 읽고 ResNet18 Baseline과 제안 손실을 각각 3 step 점검합니다. 학습 조건의 batch 크기만 진단용 인자로 덮어쓰며 원본 YAML은 수정하지 않습니다. 같은 초기 가중치·seed에서 비교하고 loss/gradient, 파라미터 갱신, PGD 요청 경로·섭동 범위, 체크포인트 저장·복원을 검사합니다. PGD는 원본의 early-stop 조건을 따르므로 실제 반복 횟수가 20회보다 적을 수 있습니다. 진단 가중치의 정확도는 보고하지 않습니다.

900초는 실행 상한이며 15분을 채워 실행하는 benchmark가 아닙니다. 초과 시 종료 코드 124와 timeout.json을 남깁니다. 출력 디렉터리는 새 경로여야 합니다. result.json에는 설정, 장치, 시간, CUDA peak 메모리와 비교 조건별 손실을 저장합니다. 이 짧은 검사로 전체 학습 시간·epoch 평가 비용을 확정할 수는 없습니다. CUDA OOM이 나면 자동으로 batch를 줄이지 않으므로 실패 기록을 확인하고 별도 진단 출력 경로에서 작은 batch로 다시 검사하세요.

현재 클라우드에는 GPU가 없어 다음 CPU 명령만 검증했습니다.

```bash
python -m tdat.preflight --device cpu --batch-size 4 --steps 3 --max-seconds 900 --output runs/preflight-cpu
```

## 로컬 학습 명령어 — 이 환경에서는 실행하지 않음

전체 데이터 Baseline과 제안 방법:

```bash
python -m tdat.train --config configs/cifar10_baseline.yaml --device cuda --download --output runs/baseline-s0
python -m tdat.train --config configs/cifar10_feature.yaml --device cuda --output runs/feature-s0
```

macOS는 `--device mps`, Linux CPU는 `--device cpu`로 바꿉니다. 전체 CPU 학습은 오래 걸립니다. CUDA/MPS 경로는 구현했으나 해당 하드웨어에서 검증하지 않았습니다. 지정한 장치를 사용할 수 없으면 오류를 내며, 장치 간 수치적 일치도 보장하지 않습니다. 단일 장치 FP32를 사용합니다.

Colab/Kaggle에서는 노트북 설정에서 GPU를 선택하고 아래 명령을 실행합니다. Colab 작업 위치는 `/content`, Kaggle은 `/kaggle/working`입니다.

```bash
git clone https://github.com/HanyangChan/Adversarial-Training-Research.git
cd Adversarial-Training-Research
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt -r requirements-autoattack.txt
python -m pytest -q tests
python -m tdat.train --config configs/cifar10_baseline.yaml --device cuda --download --output runs/baseline-s0
```

위 clone 명령을 사용하려면 현재 코드를 먼저 커밋·푸시해야 합니다. 노트북에서 작업 디렉터리를 유지하려면 `%cd`를 사용하거나 명령 전체를 하나의 `%%bash` 셀에 넣으세요. Kaggle은 다운로드를 위한 Internet 설정을 켜야 합니다. 세션 종료 전 `runs/`를 개인 영구 저장소에 보관하세요.

출력 디렉터리는 새 경로여야 합니다. 학습 재개 기능은 제공하지 않으므로 110 epoch 전체 예산을 확보하세요. 원본 README처럼 epoch 인덱스 1부터 매 epoch 체크포인트를 저장하므로 수 GB의 디스크가 필요합니다.

축소 실험은 별도 `configs/cifar10_reduced.yaml`을 사용합니다. 표본 비율·근거는 데이터 문서에 있습니다. 축소 결과를 전체 Baseline 재현으로 표현하면 안 됩니다. 여러 seed(예: 0/1/2)는 이름이 구분되는 설정 사본에 기록하고 Baseline과 제안 방법에 동일하게 적용하세요.

## 최종 평가

동일한 프로토콜의 특정 체크포인트를 지정합니다. 아래 예시는 마지막 epoch인 인덱스 109입니다.

```bash
python -m tdat.evaluate --config configs/cifar10_baseline.yaml --checkpoint runs/baseline-s0/epoch-109.pt --device cuda --attack clean --output runs/baseline-s0/clean.json
python -m tdat.evaluate --config configs/cifar10_baseline.yaml --checkpoint runs/baseline-s0/epoch-109.pt --device cuda --attack pgd20 --output runs/baseline-s0/pgd20.json
python -m tdat.evaluate --config configs/cifar10_baseline.yaml --checkpoint runs/baseline-s0/epoch-109.pt --device cuda --attack autoattack --output runs/baseline-s0/aa.json
```

정확도는 전체 테스트 10,000장에 대한 0–1 비율입니다. PGD-20은 Linf 8/255, 보폭 2/255, 20회, random start, restart 1회이며 원본처럼 오분류된 표본은 더 갱신하지 않습니다. AutoAttack은 고정된 공식 구현의 standard ensemble과 Linf 8/255를 사용합니다. 의존성·장치 오류를 다른 평가로 대체하지 않습니다. 전체 AutoAttack은 테스트·공격 텐서를 장치에 보관하므로 여유 메모리가 필요합니다. MPS AutoAttack은 미검증입니다.

## 실험 기록과 제안 손실

`metadata.json`에는 전체 config, seed, Git commit/변경 여부, 원본 커밋, Python·Torch·장치, GPU 모델·용량, 동기화한 학습·전체 시간, CUDA peak allocated/reserved 메모리를 기록합니다. 첫 커밋 전에는 `UNCOMMITTED`이며 비 CUDA의 VRAM은 0이 아닌 null입니다.

`history.json`에는 epoch별 시간·표본 수·손실·Clean/PGD-10 정확도를 저장하며 실제 학습 인덱스도 별도로 남깁니다. 체크포인트에는 모델·optimizer·scheduler·설정 메타데이터가 있습니다. 데이터·체크포인트는 Git에서 제외합니다. 실제 실험에는 `python -m pip freeze`도 보관하고 동일한 seed·분할·체크포인트 선택·공격 예산으로 비교하세요.

제안 손실은 `feature_consistency.enabled`로 켜고 끕니다. 초기 실험 가중치는 0.1이며 논문 설정이 아닙니다. 켜더라도 TDAT 목적함수는 유지되고, 끄면 추가 clean forward도 실행되지 않습니다. BN 모드 차이를 포함한 설계 가정은 출처 문서에 있습니다. 성능 향상은 아직 입증되지 않았습니다.
