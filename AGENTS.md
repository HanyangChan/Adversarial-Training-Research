# 연구 코드 작업 지침

현재 작업의 기존 체크아웃을 사용합니다. 사용자의 명시적 요청 없이는 worktree를 만들지 않습니다.
프로토콜 변경 전 README.md, docs/PROVENANCE.md, docs/DATASET.md를 읽습니다.
`vendor/TDAT` 원본은 보존합니다. Baseline의 특이 동작을 조용히 고치지 않습니다. 새 방법은 `tdat/`의 별도 모듈과 독립적인 설정 스위치로 구현합니다.
사용자의 명시적 요청 없이 실제 GPU 학습을 실행하지 않습니다. 기본 검증은 CPU synthetic 데이터의 `python -m pytest -q tests`입니다.
CPU smoke test 성공은 논문 재현·강건성·성능 향상의 증거가 아닙니다. 실제 실행 범위와 미검증 플랫폼을 구분합니다.
Baseline·축소·제안 설정을 분리합니다. seed, 표본 인덱스, Git 커밋/변경 여부, 하드웨어, 시간, VRAM을 기록하고 동일 예산·분할로 비교합니다.
`data/`, `runs/`, 체크포인트, 토큰과 자격 증명은 커밋하지 않습니다. 학생 대신 과제 보고서 본문을 작성하지 않습니다.
가상환경을 활성화하고 저장소 루트에서 실행합니다. 기본 의존성은 requirements.txt, 선택 AutoAttack 의존성은 requirements-autoattack.txt에 있습니다. Windows DataLoader worker 기본값은 0이며 CLI 진입점에는 main guard를 유지합니다.
프로젝트 안내 문서는 한국어로 작성합니다. 실행 명령, 설정 키, API 식별자와 보존된 원본 파일은 번역하지 않습니다.
