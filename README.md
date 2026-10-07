# AI Pilot — Sim2Strike

Participant-written reward functions, observations, curriculum experiments, control overlays and evaluation tools for the 2026 AI Pilot Top Gun competition. This repository is a source portfolio for Team **Sim2Strike**; the team participated in the preliminary round and did not advance.

**LEE SANGYEOP:** team lead, PPO experiments, reward/curriculum design, evaluation and integration. SAC experiments were a team collaboration; teammate SAC models are not presented as Sangyeop's sole work. Some historical code/log labels use `FDSA`; the final team name is Sim2Strike.

## Start here

| Path | What it demonstrates |
| --- | --- |
| `student/v6_ppo_my_reward.py` | Stateful combat reward with stability, geometry, WEZ, damage and safety terms |
| `student/codex_tactical26.py` | Tactical16-compatible observation extended with motion features |
| `experiments/codex_ppo_v6_s1_safety_fixed_010.yaml` | First step of the staged PPO curriculum |
| `experiments/codex_ppo_v6_s2_maneuver_loiter_from_s1_020.yaml` | Maneuvering against a loitering target |
| `experiments/codex_ppo_v6_s3_wez_loiter_from_s2_020.yaml` | WEZ approach stage |
| `experiments/codex_ppo_v6_s4_full_bt_from_s3_020.yaml` | Full BT-opponent stage |
| `tools/evaluate_bundle_suite_fast.py` | Candidate/scenario evaluation with explicit seeds |
| `src/dogfight/ai/bt_gun_opportunity_provider.py` | Attack/defense corrections around an external BT provider |
| `submission_entry.py` | Final local inference configuration; external runtime required |
| `docs/RESULTS.md` | Measured results and their limits |

Older reward versions and experiment variants are retained as an archive of the development work. A YAML file describes an experiment; its presence is not evidence that training completed or that the configuration is the final submission. `init_bundle` references refer to excluded historical checkpoints.

## Environment and execution

The training environment used **Windows, Python 3.11, JSBSim/DLL simulation and Ray RLlib (PPO/SAC)**. Recorded package versions are in `requirements-sdk.txt`; this is the historical environment, not a tested cross-platform installation recipe.

The competition's `Release_260529` SDK, DLLs, assets, starter templates, modified common SDK files, trained checkpoints, reference BT and BattleServer are **external dependencies**. Obtain the organizer release through the competition's authorized distribution. It is not redistributed here. Keep a separate SDK checkout; copy this repository's additions into their corresponding paths there. Do not replace unrelated SDK files.

Within an authorized, compatible SDK environment:

```powershell
python -m pip install -r requirements-sdk.txt
python scripts/run_experiment.py experiments/codex_ppo_v6_s1_safety_fixed_010.yaml --dry-run
python scripts/run_experiment.py experiments/codex_ppo_v6_s1_safety_fixed_010.yaml
```

The example stage uses an earlier `init_bundle`. Supply your own compatible bundle or adapt the curriculum to train from scratch; original checkpoints are not included. Some experiments rely on participant changes to common SDK interfaces that are also excluded. Review the relevant hooks and interface expectations before running; a clean organizer SDK may require additional integration work.

An evaluation example after integrating the tools and obtaining your own trained bundle:

```powershell
python tools/evaluate_bundle_suite_fast.py --candidate ppo=artifacts/models/your_bundle --scenario tail914=experiments/codex_eval_t26_bt_tail914_60s.yaml --episodes 3 --seed-start 10001 --output artifacts/evals/portfolio_eval.csv
```

`submission_entry.py` additionally requires the excluded reference BT host/runtime and the team's final integration interfaces. It documents the submitted configuration; this public archive is **not a standalone runnable copy of the submitted executable**. Do not infer official competition performance from the local tests.

## Verification and research context

All included Python source was checked for syntax during packaging. The full simulation/training stack was not run from this public archive because its SDK/runtime dependencies are excluded. The measured August 2026 results were taken from the original local verification reports, not a new training run.

Reward experiments include participant adaptations of the PHANG-MAN / hierarchical air-combat reward structure described by Adrian P. Pope et al., *Hierarchical Reinforcement Learning for Air-to-Air Combat*, [arXiv:2105.00990](https://arxiv.org/abs/2105.00990). They are adaptations to the competition state/API, with approximations documented in the source; they do not reproduce the paper's full simulator or training results. Optional LAG interfaces require a separately obtained [LAG implementation](https://github.com/liuqh16/LAG) and its own licensing/model files.

No repository-wide open-source license has been assigned; see [LICENSE-NOTE.md](LICENSE-NOTE.md).

## 한국어 소개

2026 AI Pilot Top Gun 대회에 Sim2Strike 팀장으로 참가하며 작성한 보상·관측 함수, 단계별 PPO 실험 설정, 공격·방어 보조 로직과 평가 도구를 정리했습니다. 예선에 참가했으며 본선에는 진출하지 못했습니다. PPO 학습과 평가·통합은 이상엽의 주요 기여이고, SAC 계열 실험은 팀원의 작업과 협업한 내용입니다. 팀 내부의 과거 이름 `FDSA`가 코드에 남아 있습니다.

대표 흐름은 안정 비행 → 기동 → WEZ 진입 → BT 상대 교전입니다. 최종 제출은 PPO 모델 자체가 아니라 외부 기준 BT와 자체 보조 로직을 결합한 에이전트였습니다. 10회 반복 승리는 동일한 초기 조건에서 얻은 결정론적 재현 결과이고, 여러 조건에서의 일반 승률을 의미하지 않습니다. 자기 대전에서는 조준 유지·고도 일반화·교착 문제가 남았습니다.

대회 운영진 SDK·DLL·기준 BT·서버·학습 가중치와 원시 로그는 포함하지 않았습니다. 이 저장소는 본인이 기여한 소스 포트폴리오이며 전체 제출 실행 파일을 재현하는 독립 배포본은 아닙니다. 실행에는 정식 배포받은 대회 환경과 별도의 통합 작업이 필요합니다.
