from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.ai.bt_competition_provider import BTCompetitionProvider
from dogfight.sim.state_schema import StateIndex


class FixedProvider(ActionProvider):
    def reset(self, context: ActionContext | None = None) -> None:
        return None

    def compute_action(self, context: ActionContext) -> ActionResult:
        return ActionResult(
            action=np.asarray([0.60, -0.40, 0.80, 1.00], dtype=np.float32),
            source="reference_bt32",
            info={"bt32_stage": "reference_bt"},
        )


def state(
    *,
    north_m: float,
    yaw_deg: float,
    speed_mps: float,
    time_s: float,
    altitude_m: float = 3000.0,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
) -> np.ndarray:
    value = np.zeros(51, dtype=np.float32)
    value[StateIndex.N] = north_m
    value[StateIndex.D] = -altitude_m
    value[StateIndex.YAW] = yaw_deg
    value[StateIndex.ROLL] = roll_deg
    value[StateIndex.PITCH] = pitch_deg
    value[6] = speed_mps
    value[StateIndex.SIM_TIME] = time_s
    value[StateIndex.ALT] = altitude_m
    value[StateIndex.HEALTH] = 1.0
    return value


def context(own: np.ndarray, target: np.ndarray) -> ActionContext:
    return ActionContext(
        sim=None,
        opponent_sim=None,
        ownship_state=own,
        target_state=target,
        observation=np.zeros(26, dtype=np.float32),
    )


def main() -> None:
    provider = BTCompetitionProvider(FixedProvider())

    # At excessive closure the wrapper must preserve the stable BT output.
    provider.compute_action(
        context(
            state(north_m=0.0, yaw_deg=0.0, speed_mps=350.0, time_s=0.0),
            state(north_m=508.4, yaw_deg=0.0, speed_mps=100.0, time_s=0.0),
        )
    )
    fast = provider.compute_action(
        context(
            state(north_m=0.0, yaw_deg=0.0, speed_mps=350.0, time_s=1.0 / 60.0),
            state(north_m=500.0, yaw_deg=0.0, speed_mps=100.0, time_s=1.0 / 60.0),
        )
    )
    assert fast.source == "fast_closure_guard", fast.source
    np.testing.assert_allclose(fast.action, [0.60, -0.40, 0.36, 1.00], atol=1.0e-6)

    # A true tail-aspect firing line may only receive bounded micro commands.
    provider = BTCompetitionProvider(FixedProvider(), defense_duration_s=3.0)
    provider.compute_action(
        context(
            state(north_m=0.0, yaw_deg=0.0, speed_mps=200.0, time_s=0.0),
            state(north_m=503.3, yaw_deg=0.0, speed_mps=100.0, time_s=0.0),
        )
    )
    shot = provider.compute_action(
        context(
            state(north_m=0.0, yaw_deg=0.0, speed_mps=200.0, time_s=1.0 / 60.0),
            state(north_m=500.0, yaw_deg=0.0, speed_mps=100.0, time_s=1.0 / 60.0),
        )
    )
    assert shot.source == "firing_line_micro", shot.source
    assert abs(float(shot.action[0])) <= 0.25
    assert -0.35 <= float(shot.action[1]) <= 0.20
    assert abs(float(shot.action[2])) <= 0.25

    # A close, aimed pass starts exactly one short reattack turn.
    provider.reset()
    provider.compute_action(
        context(
            state(north_m=0.0, yaw_deg=0.0, speed_mps=300.0, time_s=0.0),
            state(north_m=310.0, yaw_deg=0.0, speed_mps=0.0, time_s=0.0),
        )
    )
    provider.compute_action(
        context(
            state(north_m=0.0, yaw_deg=0.0, speed_mps=300.0, time_s=1.0 / 60.0),
            state(north_m=300.0, yaw_deg=0.0, speed_mps=0.0, time_s=1.0 / 60.0),
        )
    )
    passed = None
    for frame in range(5):
        passed = provider.compute_action(
            context(
                state(
                    north_m=0.0,
                    yaw_deg=0.0,
                    speed_mps=300.0,
                    time_s=(2.0 + frame) / 60.0,
                ),
                state(
                    north_m=-100.0 - 5.0 * frame,
                    yaw_deg=0.0,
                    speed_mps=0.0,
                    time_s=(2.0 + frame) / 60.0,
                ),
            )
        )
    assert passed is not None
    assert passed.source == "post_pass_reattack", passed.source
    assert abs(float(passed.action[0]) - 0.60) <= 0.20
    assert float(passed.action[1]) < 0.0
    np.testing.assert_allclose(passed.action[1:], [-0.40, 0.36, 1.00], atol=1.0e-6)

    # Low energy at safe altitude becomes a descending turn, not a climb spiral.
    provider.reset()
    dive = provider.compute_action(
        context(
            state(north_m=0.0, yaw_deg=90.0, speed_mps=190.0, time_s=1.0),
            state(north_m=1000.0, yaw_deg=0.0, speed_mps=200.0, time_s=1.0),
        )
    )
    assert dive.source in {"energy_dive_left", "energy_dive_right"}, dive.source
    assert float(dive.action[1]) > 0.0
    assert float(dive.action[3]) == 1.0

    # Energy recovery must never override the BT while the opponent is aiming.
    safe_provider = BTCompetitionProvider(FixedProvider())
    threatened_energy = safe_provider.compute_action(
        context(
            state(
                north_m=0.0,
                yaw_deg=0.0,
                speed_mps=190.0,
                time_s=1.0,
                altitude_m=3000.0,
            ),
            state(
                north_m=-1000.0,
                yaw_deg=0.0,
                speed_mps=200.0,
                time_s=1.0,
                altitude_m=3000.0,
            ),
        )
    )
    assert threatened_energy.source == "reference_bt", threatened_energy.source

    # A close tail threat produces a short break; repeated threats change side.
    provider.reset()
    threatened_own = state(
        north_m=0.0,
        yaw_deg=0.0,
        speed_mps=250.0,
        time_s=1.0,
    )
    threatening_target = state(
        north_m=-800.0,
        yaw_deg=0.0,
        speed_mps=250.0,
        time_s=1.0,
    )
    first_break = provider.compute_action(context(threatened_own, threatening_target))
    assert first_break.source.startswith("defense_roll_"), first_break.source
    assert float(first_break.action[3]) == 1.0
    first_sign = provider._defense_turn_sign

    # Let the first maneuver and cooldown expire, then expose the same geometry.
    threatened_own[StateIndex.SIM_TIME] = 5.1
    threatening_target[StateIndex.SIM_TIME] = 5.1
    provider.compute_action(context(threatened_own, threatening_target))
    threatened_own[StateIndex.SIM_TIME] = 8.0
    threatening_target[StateIndex.SIM_TIME] = 8.0
    second_break = provider.compute_action(context(threatened_own, threatening_target))
    assert second_break.source.startswith("defense_roll_"), second_break.source
    assert provider._defense_turn_sign == -first_sign

    # Trigger before WEZ when the opponent is aiming and our nose is displaced.
    provider.reset()
    moderate = provider.compute_action(
        context(
            state(
                north_m=0.0,
                yaw_deg=20.0,
                speed_mps=250.0,
                time_s=1.0,
                altitude_m=1100.0,
            ),
            state(
                north_m=-1500.0,
                yaw_deg=0.0,
                speed_mps=250.0,
                time_s=1.0,
                altitude_m=1100.0,
            ),
        )
    )
    assert moderate.source.startswith("defense_roll_"), moderate.source

    print("bt_competition_provider: all checks passed")


if __name__ == "__main__":
    main()
