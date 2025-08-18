## Inverse Reinforcement Learning – Grid Path Planning (IRL/DRL/Classic)

이 저장소는 격자(Gridworld) 환경에서의 경로 계획을 다양한 방법으로 비교합니다.

- **수리모델**: Dijkstra, Value Iteration
- **IRL**: MaxEnt IRL 기반 보상 학습 후 탐색(irl_guided_search)
- **DRL**: 개선된 DQN(비교용)
- **분석 출력**: 경로 시각화, 연산량(operations/expansions/inferences), 비교 그래프

### 1) 빠른 시작

- Python 3.9+ 권장
- 필수 패키지 설치(없으면 아래 최소 세트 설치)

```bash
pip install numpy matplotlib torch psutil
```

실행:

```bash
python examples/all_path_planning_comparison.py
```

출력물:

- 경로 시각화: `paths_10x10.png`, `paths_50x50.png`, `paths_70x70.png`, `paths_100x100.png`
- 성능 비교 그래프: `performance_comparison.png`
- 콘솔에 환경별 요약(길이/연산량/추론횟수 등)

### 2) 주요 스크립트

- `examples/all_path_planning_comparison.py`

  - 공통 비교 드라이버. 환경 생성 → Dijkstra/IRL/DQN 실행 → 시각화/요약 출력
  - 기본 장애물 밀도는 20%로 설정되어 있습니다. 상단의 `obstacle_density` 값으로 조정하세요.

- `examples/amr_path_planning_irl.py`

  - `AMRGridworld` 환경, 수리모델 플래너, IRL(맥스엔트) 학습/추론 유틸 포함
  - `create_connected_environment()`가 활성화되어 있어 높은 밀도에서도 경로가 완전히 막히지 않도록 일부 장애물을 제거하며 연결 가능 환경을 보장합니다.

- `examples/improved_dqn_path_planning.py`
  - 비교용 DQN 학습/경로계획 구현. 작은 지도(예: 10x10, 20% 내외)에서 비교가 용이합니다.

### 3) IRL 모델 준비(선택)

IRL 모델이 없으면 IRL 단계에서 “model not found” 메시지가 출력됩니다. 아래로 학습 후 사용하세요.

```bash
python examples/dijkstra_irl_learning.py  # 데모 IRL 학습/저장
```

기본 저장 경로(예시): `saved_models/10x10_dijkstra/irl_model_10x10_Dijkstra.pth`

### 4) 설정 포인트

- 장애물 밀도: `examples/all_path_planning_comparison.py` 하단의 `obstacle_density` 값을 조정
- 평가 격자 크기: `grid_sizes = [10, 50, 70, 100]` 리스트 변경
- DQN 학습 에피소드: 복잡도에 따라 자동 조정되며, 필요 시 `run_algorithms_safe()` 내부 설정을 수정

### 5) 자주 묻는 질문(FAQ)

- Q: 30% 장애물에서 10x10이 막혀요.

  - A: 기본은 `create_connected_environment()`가 경로 연결을 보장합니다. 완전 원시 환경을 쓰려면 해당 호출을 직접 `AMRGridworld(...)`로 바꾸세요(경로가 없을 수 있음).

- Q: DQN이 목표에 못 갑니다.

  - A: 작은 지도(10x10, 15~20% 밀도)에서 에피소드 수를 늘리거나(예: 600~1000) 최소 ε을 너무 빨리 낮추지 않도록 설정하세요. 수리모델/IRL은 전역 최적 경로를 보장/근사하므로 비교 기준으로 사용하십시오.

- Q: CUDA 메모리 오류가 납니다.
  - A: 실행 전 캐시 정리(`clear_gpu_memory()`)가 포함되어 있습니다. 여전히 문제가 있으면 CPU로 실행하거나 배치 크기를 줄이세요.

### 6) 저장소 구조(요약)

```
examples/
  all_path_planning_comparison.py   # 메인 비교 스크립트
  amr_path_planning_irl.py          # 환경/IRL/수리모델
  improved_dqn_path_planning.py     # DQN 비교(학습/경로계획)
irl/
  maxent.py 등                      # IRL 알고리즘 구현
```

### 7) 라이선스

`LICENSE` 파일을 참고하세요.
