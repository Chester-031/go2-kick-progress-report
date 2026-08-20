#!/usr/bin/env python3
from pathlib import Path
import argparse
import json

SCENARIO_BLOCK = """
        # R8.71 deterministic targeted reset cases. These overrides are
        # enabled only when R871_SCENARIO is set.
        scenario_name = os.environ.get("R871_SCENARIO", "")
        if scenario_name:
            scenario_table = {
                "center_front": (0.60, 0.00, -0.90, 0.00, 0.00, False, False, 0),
                "center_back": (0.60, 0.00, -0.90, 0.00, 3.141592653589793, False, False, 0),
                "own_penalty_front": (-2.40, 0.00, -3.10, 0.00, 0.00, False, False, 0),
                "own_penalty_back": (-2.40, 0.00, -3.10, 0.00, 3.141592653589793, False, False, 0),
                "enemy_goal_diagonal_left": (2.10, 1.25, 0.80, 0.20, 0.00, False, False, 0),
                "enemy_goal_diagonal_right": (2.10, -1.25, 0.80, -0.20, 0.00, False, False, 0),
                "own_penalty_back_diagonal": (-2.40, 1.00, -3.00, 0.20, 3.141592653589793, False, False, 0),
                "near_left_wall_avoidable": (0.80, 1.72, -0.20, 1.35, 0.00, True, True, 1),
                "near_right_wall_avoidable": (0.80, -1.72, -0.20, -1.35, 0.00, True, True, -1),
                "near_left_wall_back": (0.80, 1.72, -0.20, 1.55, 3.141592653589793, True, True, 1),
                "near_right_wall_back": (0.80, -1.72, -0.20, -1.55, 3.141592653589793, True, True, -1),
                "corner_candidate_left": (2.00, 1.78, 1.30, 1.55, 0.00, True, False, 1),
                "corner_candidate_right": (2.00, -1.78, 1.30, -1.55, 0.00, True, False, -1),
            }
            if scenario_name not in scenario_table:
                raise RuntimeError(f"Unknown R871_SCENARIO={scenario_name}")
            (
                scenario_ball_x,
                scenario_ball_y,
                scenario_robot_x,
                scenario_robot_y,
                scenario_yaw,
                scenario_wall_case,
                scenario_wall_avoidable,
                scenario_wall_sign,
            ) = scenario_table[scenario_name]
            ball_x_local.fill_(scenario_ball_x)
            ball_y_local.fill_(scenario_ball_y)
            robot_x_local.fill_(scenario_robot_x)
            robot_y_local.fill_(scenario_robot_y)
            robot_yaw.fill_(scenario_yaw)
            print(
                "[R871_SCENARIO] "
                f"name={scenario_name} "
                f"ball=({scenario_ball_x},{scenario_ball_y}) "
                f"robot=({scenario_robot_x},{scenario_robot_y}) "
                f"yaw={scenario_yaw} "
                f"wall_case={int(scenario_wall_case)} "
                f"wall_avoidable={int(scenario_wall_avoidable)} "
                f"wall_sign={scenario_wall_sign}",
                flush=True,
            )

"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    text = Path(args.source).read_text(encoding="utf-8")
    needle = "        origins = self.env_origins[env_ids]\n"
    count = text.count(needle)
    if count != 1:
        raise RuntimeError(f"R871 reset insertion site count={count}")
    text = text.replace(needle, SCENARIO_BLOCK + needle, 1)
    out = Path(args.out)
    out.write_text(text, encoding="utf-8")
    Path(str(out) + ".patch_report.json").write_text(
        json.dumps({"version": "r8_71_v4d_targeted_scan", "training": False, "scenario_count": 13}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("R871_SCENARIO_PATCH_OK")

if __name__ == "__main__":
    main()
