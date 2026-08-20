#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import py_compile
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
    required = (
        'train_go2_soccer_r8_support_skill.py',
        'run_go2_soccer_r8_support_skill.sbatch',
        'install_submit_go2_soccer_r8_support_skill.sh',
        'validate_go2_soccer_r8_support_skill.py',
        'check_go2_soccer_r8_support_skill_result.sh',
        'README.md',
    )
    errors: list[str] = []
    passed: list[str] = []

    for name in required:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f'MISSING:{name}')
        else:
            passed.append(f'file:{name}')

    source = root / 'train_go2_soccer_r8_support_skill.py'
    if source.is_file():
        try:
            py_compile.compile(str(source), doraise=True)
            passed.append('python_compile:train')
        except Exception as exc:
            errors.append(f'PYTHON_COMPILE:train:{exc}')

        text = source.read_text(encoding='utf-8')
        try:
            tree = ast.parse(text)
            names = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            }
            for required_name in (
                'IsolatedSupportTransferSkillEnv',
                '_resolve_r82_contact_sensor',
                '_resolve_r82_foot_sensor_ids',
                '_r82_foot_vertical_loads',
                '_r82_load_state',
                'r82_contact_snapshot',
                'create_r8_runner_cfg',
                'read_r8_schedule',
                'run_r8_training',
                'run_r8_evaluation',
                'main',
            ):
                if required_name not in names:
                    errors.append(f'AST_MISSING:{required_name}')
                else:
                    passed.append(f'ast:{required_name}')

            def class_method(class_name: str, method_name: str):
                for node in tree.body:
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.name == method_name:
                                return item
                return None

            parent_step = class_method('IsolatedDeltaPoseKickEnv', 'kick_step')
            r8_reward = class_method('IsolatedSupportTransferSkillEnv', '_kick_reward')
            if parent_step is None or r8_reward is None:
                errors.append('AST_DIAGNOSTIC_SCHEMA:methods_missing')
            else:
                required_keys: set[str] = set()
                for node in ast.walk(parent_step):
                    if (
                        isinstance(node, ast.Subscript)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == 'tensors'
                        and isinstance(node.slice, ast.Constant)
                        and isinstance(node.slice.value, str)
                    ):
                        required_keys.add(node.slice.value)

                provided_keys: set[str] = set()
                for node in ast.walk(r8_reward):
                    if not isinstance(node, ast.Assign):
                        continue
                    if not any(
                        isinstance(target, ast.Name) and target.id == 'diagnostics'
                        for target in node.targets
                    ):
                        continue
                    if not isinstance(node.value, ast.Dict):
                        continue
                    for key in node.value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            provided_keys.add(key.value)
                missing = sorted(required_keys - provided_keys)
                if missing:
                    errors.append('AST_DIAGNOSTIC_SCHEMA_MISSING:' + ','.join(missing))
                else:
                    passed.append('ast:r8_diagnostic_schema_complete')

            build_obs = class_method('IsolatedSupportTransferSkillEnv', '_build_low_observation')
            if build_obs is None:
                errors.append('AST_MISSING:_build_low_observation')
            else:
                obs_text = ast.unparse(build_obs)
                forbidden_obs_terms = (
                    'r82_contact_sensor',
                    'selected_load',
                    'net_forces_w',
                )
                leaking = [term for term in forbidden_obs_terms if term in obs_text]
                if leaking:
                    errors.append('CONTACT_FORCE_OBSERVATION_LEAK:' + ','.join(leaking))
                else:
                    passed.append('ast:contact_force_not_in_actor_observation')
        except Exception as exc:
            errors.append(f'AST_PARSE:{exc}')

        required_tokens = (
            '# R8_2_FORCE_GROUNDED_UNLOAD_CURRICULUM_PPO',
            '[R8_2_CONTACT_SENSOR]',
            '[R8_2_CONTACT_MAPPING]',
            'data.net_forces_w',
            'selected_foot_vertical_contact_load_relative_to_stance',
            'contact_force=privileged_reward_only_not_actor_observation',
            'actor_observation=70D_teacher_free',
            'low_action=12D_joint_position_residuals',
            'high_level_router_interface=unchanged',
            'soccer_goal_reward=not_used',
            'ball_contact_training=deferred_until_video_review',
            'observation[:, self.KICK_CONTEXT_TEACHER_FLAG] = 0.0',
            '"unload_half"',
            'self.curriculum_stage = self.R8_STAGES[0]',
            '"unload_light"',
            '"lift_5mm"',
            '"lift_1cm"',
            '"lift_2cm"',
            '"lift_4cm"',
            '"swing_3cm"',
            '"swing_6cm"',
            'self.r82_unload_seen',
            'self.max_selected_foot_lift >= 0.040',
            'self.max_airborne_foot_forward >= 0.060',
            'self.max_base_displacement_r8 <= 0.060',
            'self.max_base_height_error_r8 <= 0.030',
            'self.max_upright_error_r8 <= 0.020',
            'force_grounded_success_gated_not_fixed_time',
            'airborne_forward_only',
            '[R8_2_STAGE_GATE]',
            '[R8_2_STAGE_TIMEOUT]',
            '[R8_2_FINAL_GUARD]',
            'completed_env_fraction_informational_only',
            'R8_2_SUPPORT_PILOT_COMPLETE',
            'R8_2_SUPPORT_SKILL_ACCEPTED',
            'R8_EVAL_VIDEO_READY',
        )
        for token in required_tokens:
            if token not in text:
                errors.append(f'TOKEN_MISSING:{token}')
            else:
                passed.append(f'token:{token}')

        forbidden_tokens = (
            'teacher_action_labels=full_12D',
            'R6_TEACHER_FEASIBILITY_ACCEPTED',
            'R8 rollout completion guard failed',
            'self.curriculum_stage = "unload_1cm"',
            'self.max_selected_foot_forward >= 0.060',
        )
        for token in forbidden_tokens:
            if token in text:
                errors.append(f'FORBIDDEN_TOKEN:{token}')
            else:
                passed.append(f'forbidden_absent:{token}')

    for name in (
        'run_go2_soccer_r8_support_skill.sbatch',
        'install_submit_go2_soccer_r8_support_skill.sh',
        'check_go2_soccer_r8_support_skill_result.sh',
    ):
        path = root / name
        if path.is_file():
            result = subprocess.run(
                ['bash', '-n', str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                errors.append(f'SHELL_SYNTAX:{name}:{result.stderr.strip()}')
            else:
                passed.append(f'shell_syntax:{name}')

    worker = root / 'run_go2_soccer_r8_support_skill.sbatch'
    if worker.is_file():
        text = worker.read_text(encoding='utf-8')
        for token in (
            '/scratch/lj2634/envs/isaaclab232_v2',
            '/scratch/lj2634/IsaacLab_232_src_37ddf62',
            '/scratch/lj2634/vendor/libglu-9.0.0',
            'R82_GATE_WINDOW=20',
            'R82_GATE_MIN_EPISODES=256',
            'R82_CONTACT_FORCE_THRESHOLD_N=5.0',
            'R82_MIN_BASELINE_FORCE_N=5.0',
            'R8_INIT_NOISE_STD=0.32',
            '--kick_episode_steps 80',
            '${R8_MAX_ITERATIONS:-1000}',
            'R8_2_SUPPORT_PILOT_COMPLETE',
        ):
            if token not in text:
                errors.append(f'WORKER_TOKEN:{token}')
            else:
                passed.append(f'worker:{token}')

    installer = root / 'install_submit_go2_soccer_r8_support_skill.sh'
    if installer.is_file():
        text = installer.read_text(encoding='utf-8')
        for token in (
            'torch_pr_1066_general',
            'experiments/r8_2_force_grounded_support_skill',
            'runs/go2_soccer_r8_2_force_grounded_support_skill',
            'r82sup-pilot',
            'r82sup-video',
            '--dependency=afterok:',
            'R8_NUM_ENVS=256',
            'R8_MAX_ITERATIONS=1000',
            'latest_r8_2_force_grounded_support_skill.txt',
            'high_level_router_interface=unchanged',
            'ball_contact_training=deferred_until_pilot_video_review',
            'unset SBATCH_ACCOUNT',
        ):
            if token not in text:
                errors.append(f'INSTALLER_TOKEN:{token}')
            else:
                passed.append(f'installer:{token}')

    payload = {
        'counts': {'errors': len(errors), 'warnings': 0, 'passed': len(passed)},
        'errors': errors,
        'passed': passed,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(
        'R8_2_SUPPORT_PACKAGE_AUDIT_COUNTS '
        f'errors={len(errors)} warnings=0 passed={len(passed)}'
    )
    if errors:
        print('R8_2_SUPPORT_PACKAGE_AUDIT_FAILED')
        return 1
    print('R8_2_SUPPORT_PACKAGE_AUDIT_CLEAN')
    print('R8_SUPPORT_PACKAGE_AUDIT_CLEAN')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
