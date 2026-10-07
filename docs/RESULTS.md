# Local verification results and limitations

Source: the team's final verification and self-play reports dated **2026-08-24**. These are local BattleServer V1.2 tests, not official competition rankings.

| Test | Own health | Reference health | Outcome |
| --- | ---: | ---: | --- |
| Blue, fixed reference scenario | 96.453 | 95.013 | Local win |
| Red, final validation | 100.000 | 94.201 | Local win |

The Blue executable was repeated ten times at the same initial distance (about 912.9 m), speed (about 250.1 m/s), altitude and placement. Every repeat had the same score; the maximum per-frame trajectory difference versus the first run was zero. This demonstrates deterministic repeatability in that one fixed setup. It is not ten independent samples of a scenario-wide win probability.

Self-play across other initial conditions exposed a different result: several 200-second runs completed without a crash but inflicted no damage. At close range, the controller could briefly achieve a small angle while outside the weapon envelope, then overshoot by the time it entered range. Excessive climbs, altitude variation and symmetric stalemates also remained. Candidate micro-correction changes were not included in the final submission when they failed to show a clear improvement.

The official team result was **preliminary-round participation without advancement**.

## Comparing PPO and SAC fairly

The internal evaluation separated two questions:

- Stage-matched ability: can a curriculum policy acquire the target position under the task it was trained on?
- Combat readiness: can it enter the official WEZ, inflict damage, avoid a crash and complete the engagement?

The team's archived SAC keeper did better on its position-acquisition stage, while some PPO candidates performed better on the particular damage/WEZ checks. These objectives and starting conditions differed. They do not justify a general claim that one algorithm outperforms the other.

## 한국어

고정 기준 모델 시험에서 Blue 10회 반복과 Red 최종 시험은 승리했습니다. 그러나 10회 모두 같은 초기 조건이며 궤적도 동일했으므로 다양한 상황의 일반 승률로 해석할 수 없습니다. 별도 자기 대전에서는 피해를 주지 못하는 조준·접근 속도·고도·교착 문제가 확인됐습니다. 공식 대회 결과는 예선 참가이며 본선 진출이나 수상을 주장하지 않습니다.
