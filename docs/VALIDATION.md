# 검증 기록

초기 검증: 2026-11-11 UTC. 한국어 문서 전환 및 게시 환경 재검증: 2026-11-12 한국 시간.

환경: Linux x86_64, Python 3.12.14, Torch 2.5.1+cpu, torchvision 0.20.1+cpu. 기본 의존성과 커밋으로 고정한 AutoAttack을 설치했습니다. 정확한 패키지 목록은 `environment-linux-cpu.txt`입니다. 이는 Linux CPU 실행 증거이며 플랫폼 공통 lockfile이 아닙니다.

초기 `python -m pytest -q tests` 결과: **6 passed in 5.21s**. 게시 환경에서도 동일 테스트를 다시 실행했습니다. 최신 결과는 아래 재검증 항목에 기록합니다.

1. 실제 원본 CIFAR ResNet18에 synthetic batch 2, RGB 32×32 입력: forward, TDAT 내부·외부·강건 손실, backward, SGD 파라미터 변경, 제안 손실 on/off, 최종 섭동 범위, 모델 체크포인트 저장·복원 및 eval logits 일치를 확인했습니다.
2. 클래스 균형 consistency 손실은 수계산 값 0.75, clean stop-gradient와 adversarial gradient를 확인했습니다.
3. 경량 선형 모델에서 synthetic PGD-20의 픽셀·epsilon 범위와 clean/robust 정확도 경로를 확인했습니다.
4. 특징 추출 hook 적용 전후 logits 일치와 특징 크기 2×512를 확인했습니다.
5. synthetic loader를 주입한 학습·평가 CLI 진입 함수에서 CPU 1 epoch, epoch PGD-10, 체크포인트 optimizer/scheduler 메타데이터, 최종 PGD-20, 시간 및 비 CUDA 메모리 null 기록을 확인했습니다. 이는 smoke 전용 설정입니다.
6. 제공한 YAML 설정 검증과 미지원 프로토콜 변경 시 명시적인 오류 발생을 확인했습니다.

초기 `pip check`: 의존성 충돌 없음. 학습·평가 `--help`: 정상. AutoAttack import: 정상이나 전체 공격 실행은 하지 않았습니다. 원본 스냅샷은 고정된 Git 객체와 바이트 단위로 같으며 `UPSTREAM_SHA256.json`에 해시를 기록했습니다. 생성된 bytecode는 제외합니다.

seed 0 synthetic batch 2의 예시 손실: inner=2.69723, outer CE=3.37014, robust=0.541737, 총합=3.69519. 제안 손실 활성화 시 feature=0.385976, 총합=3.73378입니다. 이는 코드 경로 확인값이며 정확도·재현·개선 증거가 아닙니다.

초기 검증 당시 미검증: 실제 CIFAR 다운로드·데이터 경로, 데이터 라이선스 조건, 실제 학습, CUDA/MPS/Windows, 전체 AutoAttack, 과거 Torch 환경과의 동등성, 논문 Baseline 수치입니다. 아래 최신 기록에서 데이터 경로 상태를 갱신합니다. 실제 GPU 학습은 실행하지 않았고 데이터·체크포인트를 저장소에 추가하지 않았습니다. 첫 커밋 전 실행 기록은 UNCOMMITTED/dirty이며 과제 보고서는 작성하지 않았습니다.

## 이전 게시 환경 재검증(아래 최신 기록으로 갱신)

공식 호스트 허용은 적용되었으나 공식 페이지가 `cave.cs.toronto.edu`로 301 이동합니다. 리다이렉트 후 403이 발생하여 실제 다운로드·라이선스 확인은 계속 미완료입니다. 새 호스트를 허용할 필요가 있으며 기존 호스트 허용이 실패한 것으로 해석하지 않습니다.

게시 환경 재실행: **6 passed in 4.37s**. `pip check`는 의존성 충돌 없음. 한국어 전환 후 원본 28개 파일의 SHA256도 모두 일치했습니다. 설정 키·하이퍼파라미터·실행 코드는 변경하지 않았습니다. 새 도메인과 한국어 시작 지침은 초안에 저장했으며 게시하지 않았습니다.

## 최신 게시 환경 확인 — 2026-11-12 한국 시간

두 공식 호스트 허용 적용과 실제 HTTPS 연결을 확인했습니다. 기존 리다이렉트 후 403 차단은 해결되었습니다. 공식 압축파일 전체 MD5와 torchvision 개별 파일 검증을 통과하고 실제 train/test 구성·전처리·축소 표본을 확인했습니다. `tdat.prepare_data`를 추가하여 학습 없는 준비·검증과 재실행 경로를 제공합니다. 원본 TDAT 파일과 학습·공격 설정은 변경하지 않았습니다.

- 데이터: 공식 파일 170,498,071바이트, MD5 `c58f30108f718f92721af3b95e74349a` 일치.
- 전체 분할: 학습 50,000장·테스트 10,000장, 클래스별 5,000장·1,000장.
- 첫 batch: train/test 모두 `[128,3,32,32]`, float32, 값 [0,1].
- 10% 별도 표본: 5,000장, 클래스별 500장, seed 0 결정성 확인.
- CPU synthetic 테스트: **6 passed in 3.92s**. 실제 데이터 검사와 구분합니다.
- `pip check`: 의존성 충돌 없음. 원본 TDAT 파일 28개 SHA256 일치.
- 공식 페이지 직접 확인: 기술보고서 인용 요청과 원천 데이터 설명은 확인했으나 명시적 라이선스 조항은 찾지 못했습니다.

`docs/DATA_VALIDATION.json`은 실제 검증 출력의 작은 요약 사본이며 데이터 이미지를 포함하지 않습니다. 압축파일·이미지·runs 출력은 Git 제외 상태입니다. GPU/CPU 실제 학습, CUDA/MPS/Windows 실행, 전체 AutoAttack, 논문 Baseline 수치 및 성능 향상은 계속 미검증입니다. 환경은 게시하지 않았습니다.

## 15분 한도 환경·데이터·모델 진단 — 2026-11-12 한국 시간

현재 클라우드는 Torch 2.5.1+cpu이며 CUDA 장치·nvidia-smi·MPS가 없습니다. 로컬 3060 Ti는 직접 접근하지 않았습니다. `tdat.preflight`를 추가해 로컬 CUDA와 현재 CPU에서 같은 진단 경로를 사용할 수 있도록 했습니다. 900초 초과 시 자식 프로세스를 종료하며 제한 경로는 1초 타임아웃 검사로 검증했습니다(종료 코드 124).

실행: `python -m tdat.preflight --device cpu --batch-size 4 --steps 3 --max-seconds 900 --output runs/preflight-cpu-real-20261112-v2`.

실제 CIFAR 첫 학습 4장과 첫 테스트 4장, 원본 ResNet18, 같은 초기 가중치와 seed 0에서 Baseline·제안 손실을 각각 3 step 검사했습니다. 손실·gradient 유한성, optimizer 파라미터 변경, 특징 손실 on/off, PGD 요청 경로·섭동 범위, 모델/optimizer 체크포인트 저장 및 모델 복원 logits 일치를 확인했습니다. 무작위에 가까운 진단 모델이므로 PGD가 early-stop할 수 있으며 전체 20회 반복 또는 강건 정확도 검증으로 해석하지 않습니다. 전체 분할이나 원본 YAML을 바꾸지 않았으며 이 검사는 성능 비교용 축소 실험이 아닙니다.

데이터 전체 무결성·분할·전처리 검증 재실행 통과, `pip check` 충돌 없음, CPU 테스트 **7 passed in 5.04s**. 첫 커밋이 없을 때 Git 명령의 실패 출력을 HEAD 해시로 기록하던 메타데이터 버그를 수정하고 회귀 검사를 추가했습니다. 이제 UNCOMMITTED로 명시합니다.

실제 진단 소요시간과 조건별 손실은 `docs/PREFLIGHT_CPU.json`, 체크포인트는 Git 제외 경로 `runs/preflight-cpu-real-20261112-v2/diagnostic.pt`에 있습니다. 검사는 15분 상한 안에서 조기 완료되었으며 15분 지속 benchmark가 아닙니다. CPU 소량 검사로 3060 Ti 속도·8GB VRAM 적합성·논문 Baseline 재현·성능 향상을 주장하지 않습니다. 해당 CUDA 명령은 README에 제공했으며 아직 로컬 GPU에서 실행하지 않았습니다.
