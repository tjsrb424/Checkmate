# AI Beta Readiness

외통수 AI는 현재 상급 엔진이 아니라 초보자용 AI 대국 베타입니다. 실수할 수 있으며, Arena 리그와 블런더 fixture로 계속 품질을 확인합니다.

## Beta Gates

- hard가 normal 대비 Arena scoreRate 55% 이상
- hard forfeit 0
- illegal move forfeit 0
- drawRate 70% 미만
- CHO/HAN 결정국 승률 편향 35%~65%
- blunder regression fixture 전체 통과
- hard completedDepth 평균 4 이상
- 후보수 Top N 표시 정상
- 오프닝북 ON/OFF fallback 정상
- AI 오류 발생 시 대국 상태 보존

## Commands

```bash
npm run arena:league:quick
npm run beta:check
```

Strict mode:

```bash
npm run beta:check:strict
```

## Manual Smoke Checklist

1. 새 대국 시작
2. 사람 초/한 모두 테스트
3. AI 선공 테스트
4. 오프닝북 ON/OFF 테스트
5. 무르기/다시두기 테스트
6. 현재 포지션 분석 테스트
7. AI 사고 중 새 대국 테스트
8. hard 난이도 3판 테스트
9. AI가 공짜 차/포를 헌납하는지 확인
10. 외통 위협 대응 확인

## Opening Book Loading Strategy

현재 앱 기본값은 small built-in seed book입니다. 대형 `opening-book.json`은 번들에 직접 import하지 않습니다.

권장 단계:

- 현재: built-in seed 유지
- 다음: `public/opening-book/opening-book.json`을 fetch로 lazy load
- 이후: Worker preload로 book을 한 번만 로드하고 request에는 book id만 전달

## Known Limitations

- readiness는 내부 기준이며 외부 강엔진 비교가 아닙니다.
- quick Arena는 smoke test용이라 통계적 의미가 작습니다.
- 실제 사용자 패배 기보 fixture가 더 필요합니다.
- tacticalSafety는 완전한 SEE가 아니라 1차 근사입니다.
