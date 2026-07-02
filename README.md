# 외통수

외통수는 한국 장기에서 사람이 직접 AI와 대국할 수 있는 강한 장기 AI를 목표로 하는 프로젝트입니다.

## 현재 구현 범위

- React + TypeScript + Vite 기반 사람 vs AI 장기 UI
- 9x10 장기판 렌더링
- 초/한 및 4가지 차림 선택
- 기물 선택과 합법수 이동
- 장기 룰 엔진
  - 차, 포, 마, 상, 사, 장, 졸 이동
  - 마/상 길막힘
  - 포 넘기 및 포끼리 제한
  - 궁성 대각 이동
  - 장군 및 외통 판정
- 탐색 AI
  - 합법수 생성
  - 평가 함수
  - Negamax
  - Alpha-Beta pruning
  - Iterative deepening
  - 시간 제한 탐색
  - 쉬움/보통/어려움 난이도

## 개발 명령

```bash
npm install
npm run dev
npm run test
npm run build
```

## 방향

초기 엔진은 TypeScript로 구현되어 있으며, 추후 Rust 또는 C++ 엔진 코어로 분리할 수 있도록 UI와 엔진 모듈을 나누어 둡니다.
