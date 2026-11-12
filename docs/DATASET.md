# CIFAR-10 데이터 프로토콜

공식 출처: https://www.cs.toronto.edu/~kriz/cifar.html
공식 압축파일: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
참고문헌: Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images (2009).

32×32 RGB 이미지 60,000장, 균형 잡힌 10개 클래스입니다. 공식 train은 50,000장(클래스당 5,000), test는 10,000장(클래스당 1,000)이며 원본에는 별도 validation 분할이 없습니다. 테스트 데이터를 학습에 섞지 않습니다. 원본은 매 epoch 테스트 Clean/PGD-10 정확도를 확인하므로 best 체크포인트를 선택한다면 이 사실을 공개해야 합니다. 마지막 결과와 원본 기준으로 지정한 best 결과를 구분하고, 독립 검증셋을 사용한 것처럼 표현하지 않습니다.

학습 전처리: RandomCrop(32,padding=4), RandomHorizontalFlip, ToTensor, 항등 Normalize(mean=0,std=1). 테스트: ToTensor와 항등 Normalize입니다. 픽셀 범위는 [0,1]이며 일반적인 CIFAR 평균·표준편차 정규화를 적용하지 않습니다.

Torchvision CIFAR10은 공식 압축파일을 내려받고 내장 MD5 `c58f30108f718f92721af3b95e74349a`로 무결성을 검사합니다. 검증을 끄지 않습니다. 학습 없이 다운로드하려면 저장소 루트에서 다음을 실행합니다.

```python
from torchvision.datasets import CIFAR10
CIFAR10('data', train=True, download=True)
CIFAR10('data', train=False, download=True)
```

`tdat.train --download`는 다운로드 후 학습까지 시작하므로 단순 다운로드 확인용으로 사용하지 마세요.

## 현재 접근 확인 상태

2026-11-12(한국 시간) 새 게시 환경에서 두 공식 호스트(`www.cs.toronto.edu`, `cave.cs.toronto.edu`) 허용 적용과 공식 페이지 접근 성공을 확인했습니다. 이전 301 리다이렉트 후 403 문제는 해결되었습니다. Python 압축파일은 리다이렉트 목적지인 `https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz`에서 다운로드합니다. 공식 페이지의 MD5와 torchvision 내장 MD5가 같습니다. 다운로드·검증 결과는 아래 실행 기록과 검증 문서에 기록합니다.

학습 없이 준비·검증하려면 다음 명령을 사용하세요.

```bash
python -m tdat.prepare_data --download
# 다운로드된 파일을 다시 검증할 때
python -m tdat.prepare_data
```

결과는 `runs/data-validation.json`에 저장합니다. 원본 압축파일 MD5, torchvision 개별 배치 파일 무결성, 전체 train/test 클래스별 수량, 전처리된 첫 batch와 10% 표본 구성을 검사하며 모델 학습은 하지 않습니다.

## 라이선스

2026-11-12(한국 시간) 공식 페이지 본문을 직접 확인했습니다. 페이지는 데이터셋이 80 Million Tiny Images의 레이블 부분집합이며 Alex Krizhevsky, Vinod Nair, Geoffrey Hinton이 만들었다고 설명합니다. 사용 시 Alex Krizhevsky의 2009년 기술보고서 *Learning Multiple Layers of Features from Tiny Images*를 인용하도록 요청합니다.

확인한 공식 페이지에는 명시적 LICENSE, 저작권 이용허락 또는 MIT/CC/BSD 조항이 없습니다. 따라서 **공식 출처와 인용 요청은 확인했으나 이미지의 명시적 라이선스 조건은 확인되지 않았습니다**. 이는 네트워크 때문에 페이지를 읽지 못한 이전 상태와 다릅니다. 미러의 데이터 카드만으로 원본 조건을 추정하지 않으며 이미지 배포 전 공식 배포자와 원천 이미지 조건을 확인하세요. 이미지는 Git에 커밋하지 않습니다.

## 축소 실험

별도 축소 설정은 seed 기반 층화 비복원 추출로 클래스당 500장, 총 5,000/50,000장(10%)을 사용합니다. 전체 테스트 10,000장은 유지합니다. 실제 인덱스를 저장합니다. 목적은 계산 예산을 제한한 탐색이며 원본 Baseline이 아닙니다. HW2에서 충분한 규모인지 Colab/Kaggle의 실제 자원·사용량 측정으로 정당화해야 합니다. 현재 축소 학습과 GPU 자원 측정은 수행하지 않았습니다. 제안 방법과 Baseline에는 같은 표본을 사용하며 전체 Baseline 설정은 축소하지 않습니다.

## 실제 데이터 검증 완료 기록 — 2026-11-12 한국 시간

공식 파일 170,498,071바이트를 내려받아 MD5 `c58f30108f718f92721af3b95e74349a` 일치를 확인한 후 torchvision으로 압축 해제했습니다. 경로는 `/workspace/Adversarial-Training-Research/data/cifar-10-batches-py`이며 기존 YAML의 `data.root: data`를 그대로 사용합니다.

전체 train 50,000장(클래스별 5,000), test 10,000장(클래스별 1,000), float32 첫 배치 `[128,3,32,32]`, 픽셀 범위 [0,1], int64 레이블 0–9 검증을 통과했습니다. 10% 층화 표본은 클래스별 500장, 총 5,000장이며 seed 0의 표본 재생성 일치를 확인했습니다. 전체 Baseline의 학습 데이터는 축소하지 않았습니다.

`python -m tdat.prepare_data --download`와 이어서 `python -m tdat.prepare_data`를 실행하여 다운로드 없는 재검증도 확인했습니다. 실행 결과는 `runs/data-validation.json`, 요약 사본은 `docs/DATA_VALIDATION.json`에 있습니다. 이미지·압축파일·실행 출력은 Git 제외 대상으로 확인했습니다. 실제 학습이나 전체 강건성 평가 결과는 아닙니다.
