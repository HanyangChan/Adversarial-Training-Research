# 출처와 원본 충실도 검토

- 공식 구현: https://github.com/bookman233/TDAT, 커밋 `2ce7bb0650584756a8bbb8deca575da4150718c4`.
- 논문: Tong et al., Taxonomy Driven Fast Adversarial Training, AAAI 38(6), 5233–5242 (2024). 원본 PDF는 `vendor/TDAT`에 보존했습니다.
- `vendor/TDAT`는 원본 소스 스냅샷입니다. 생성된 `models/__pycache__` 파일만 제외했으며 소스는 수정·번역하지 않았습니다. 해당 커밋에는 LICENSE 파일이 없으므로 상업적 재배포 허가를 추정하거나 자체 라이선스를 부여하지 않습니다. 재배포 전 저자에게 확인하세요.
- HW2.pdf는 과제 참고 자료입니다. 이 프로젝트는 연구 코드와 절차를 제공하며 보고서 본문을 작성하지 않습니다. Baseline 대비 성능 향상은 아직 입증되지 않았습니다.

## 공개 코드와 논문의 차이

Baseline YAML은 공식 README 명령과 공개 코드에 맞췄습니다. 논문의 수치적 재현이 검증되었다는 뜻은 아닙니다. `lamda=0.6`은 README와 CIFAR10 로그 기준이며 parser 기본값은 0.65입니다. 저장 시작 epoch는 README의 1을 유지했고 로그에서는 104를 사용했습니다. 저장 빈도는 최적화 과정에 영향을 주지 않습니다.

논문 식 (6)과 Algorithm 1은 **tanh**를 쓰지만 TDAT.py는 relaxation에 **math.tan**을 사용합니다. 공개 코드는 batch momentum 가중평균을 구한 뒤 현재 delta의 clamp 결과로 덮어쓰므로 초기화 이후 실질적으로 batch_m이 작동하지 않습니다. 두 동작을 조용히 수정하지 않고 보존했습니다. 외부 손실의 PyTorch label_smoothing은 내부 손실이 잘못된 9개 클래스에 명시적으로 확률을 분배하는 방식과도 다릅니다. 이 비대칭도 유지합니다.

초기 섭동은 sign(uniform[-8/255,8/255])*8/255입니다. 초기 모델 입력은 [0,1]로 자르지 않으며 최종 FGSM 입력만 자릅니다. 섭동은 batch·epoch 사이에 이어집니다. 크기가 덜 찬 마지막 batch(128 기준 80장)는 매 epoch 건너뛰지만 scheduler milestone은 실제 optimizer step 수가 아닌 ceil(50000/128)을 기준으로 합니다. 초기 logits 그래프도 학습 손실에 연결되며 retain_graph와 parameter gradient 초기화 순서를 원본에 맞췄습니다. FP32, AMP 미사용, 항등 정규화입니다.

모델은 torchvision ImageNet ResNet18이 아닌 원본 CIFAR ResNet18입니다. import 시 CUDA 텐서를 생성하는 utils.py를 피하기 위해 resnet.py를 직접 불러옵니다. 장치 이식을 위해 DataParallel을 제거했으며 원본과 단일 장치의 수치적 동등성은 검증하지 않았습니다. Python·NumPy·Torch에 seed를 설정합니다(원본은 Python random 설정이 빠져 있습니다). 장치별 난수 소비와 연산 차이로 플랫폼 간 bitwise 재현을 보장하지 않습니다. 테스트 shuffle은 원본을 따릅니다. CUDA 다중 GPU 학습은 이식 범위 밖입니다.

학습 조건: SGD lr=0.1, momentum=0.9, weight decay=0.0005, 110 epoch, batch=128, 명목 milestone=100/105, lr gamma=0.1, epsilon=8/255, FGSM alpha=8/255, beta=0.6, inner/outer 최솟값=0.15입니다. 매 epoch PGD-10은 random restart 1회, CE 공격 alpha=2/255, epsilon=8/255입니다. PGD-20도 원본 utility처럼 오분류된 표본 갱신을 중단합니다.

논문에는 이 프로젝트에서 요청하지 않은 다른 공격 평가도 있습니다. 최종 AutoAttack은 고정된 공식 구현의 standard ensemble과 Linf 8/255를 사용합니다. 논문 본문에 AA의 버전·restart 세부사항이 모두 명시되어 있지 않으므로 이는 기록된 평가 구현 선택이며 확인된 원본 설정으로 단정하지 않습니다.

원본 README 환경은 Python 3.7.13, Torch 1.13.1, torchvision 0.13입니다. Torch/torchvision은 공식 호환 조합과 다르므로 그대로 설치하지 않았습니다. 이식 환경은 Python 3.10–3.12, Torch 2.5.1/torchvision 0.20.1이며 역사적 실행 환경과 동일하지 않습니다. 저장소에서 확인할 수 없는 보충자료의 하이퍼파라미터를 임의로 추가하지 않았습니다.

## 제안 손실: Class-Aware Feature Consistency

`tdat/consistency.py`는 최종 분류층 직전의 512차원 특징을 사용합니다. clean 특징에는 stop-gradient를 적용합니다. 같은 표본의 clean/adversarial 특징 간 `1 - cosine similarity`를 각 정답 클래스 안에서 평균한 뒤, batch에 존재하는 클래스별 평균을 동일한 가중치로 평균합니다.

목적함수는 `원본 TDAT + weight × consistency`입니다. 초기 가중치 0.1은 제안 실험값이며 논문 설정이 아닙니다. clean 특징 추출은 BN 통계를 추가 갱신하지 않도록 eval 모드에서 하고 adversarial 특징은 train 모드에서 얻습니다. 이 BN 모드 차이를 포함한 설계는 검증할 가설이며 성능 개선이 확인된 것이 아닙니다. 비활성화 시 추가 clean forward는 실행하지 않습니다.

Baseline과 제안 설정의 데이터 분할·seed·체크포인트 선택을 같게 유지하세요. 원본의 테스트 모니터링을 공개하지 않고 테스트 정확도로 모델을 선택하면 안 됩니다.
