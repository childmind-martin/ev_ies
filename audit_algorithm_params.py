from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent

FIELDS = (
    "algorithm",
    "total_timesteps",
    "seed",
    "learning_rate",
    "gamma",
    "batch_size",
    "buffer_size",
    "tau",
    "policy_delay",
    "action_noise_type",
    "action_noise_sigma",
    "target_policy_noise",
    "target_noise_clip",
    "n_steps",
    "n_epochs",
    "clip_range",
    "ent_coef",
    "ees_terminal_soc_tolerance",
    "penalty_ees_terminal_soc",
)

TRAINING_SCRIPTS = {
    "TD3": ROOT / "train_TD3.py",
    "DDPG": ROOT / "train_DDPG.py",
    "PPO": ROOT / "train_PPO_sb3_direct.py",
}

SCRIPT_KEYS = {
    "total_timesteps": "TOTAL_TIMESTEPS",
    "seed": "SEED",
    "learning_rate": "LEARNING_RATE",
    "gamma": "GAMMA",
    "batch_size": "BATCH_SIZE",
    "buffer_size": "BUFFER_SIZE",
    "tau": "TAU",
    "policy_delay": "POLICY_DELAY",
    "action_noise_type": "ACTION_NOISE_TYPE",
    "action_noise_sigma": "ACTION_NOISE_SIGMA",
    "target_policy_noise": "TARGET_POLICY_NOISE",
    "target_noise_clip": "TARGET_NOISE_CLIP",
    "n_steps": "N_STEPS",
    "n_epochs": "N_EPOCHS",
    "clip_range": "CLIP_RANGE",
    "ent_coef": "ENT_COEF",
}

EES_CONFIG_KEYS = (
    "ees_terminal_soc_tolerance",
    "penalty_ees_terminal_soc",
)


def eval_expr(node: ast.AST, constants: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [eval_expr(item, constants) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(eval_expr(item, constants) for item in node.elts)
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.UnaryOp):
        value = eval_expr(node.operand, constants)
        if isinstance(node.op, ast.USub) and isinstance(value, (int, float)):
            return -value
        if isinstance(node.op, ast.UAdd) and isinstance(value, (int, float)):
            return value
    if isinstance(node, ast.BinOp):
        left = eval_expr(node.left, constants)
        right = eval_expr(node.right, constants)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Pow):
                return left**right
    return None


def read_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def read_module_constants(path: Path) -> dict[str, Any]:
    constants: dict[str, Any] = {}
    tree = read_tree(path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = eval_expr(node.value, constants)
            if value is not None:
                constants[node.targets[0].id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            value = eval_expr(node.value, constants)
            if value is not None:
                constants[node.target.id] = value
    return constants


def read_ies_defaults(path: Path) -> dict[str, Any]:
    tree = read_tree(path)
    defaults: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "IESConfig":
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.value is not None:
                if item.target.id in EES_CONFIG_KEYS:
                    defaults[item.target.id] = eval_expr(item.value, defaults)
    return defaults


def read_ies_overrides(path: Path) -> dict[str, Any]:
    tree = read_tree(path)
    overrides: dict[str, Any] = {}
    constants = read_module_constants(path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_ies_config = (
            isinstance(func, ast.Name) and func.id == "IESConfig"
        ) or (
            isinstance(func, ast.Attribute) and func.attr == "IESConfig"
        )
        if not is_ies_config:
            continue
        for keyword in node.keywords:
            if keyword.arg in EES_CONFIG_KEYS:
                overrides[keyword.arg] = eval_expr(keyword.value, constants)
    return overrides


def build_row(algorithm: str, path: Path, ies_defaults: dict[str, Any]) -> dict[str, Any]:
    constants = read_module_constants(path)
    ies_values = dict(ies_defaults)
    ies_values.update(read_ies_overrides(path))

    row: dict[str, Any] = {"algorithm": algorithm}
    for field, key in SCRIPT_KEYS.items():
        row[field] = constants.get(key)
    for key in EES_CONFIG_KEYS:
        row[key] = ies_values.get(key)
    return row


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    rendered = [[format_value(row.get(field)) for field in FIELDS] for row in rows]
    widths = [
        max(len(field), *(len(row[idx]) for row in rendered))
        for idx, field in enumerate(FIELDS)
    ]
    header = " | ".join(field.ljust(widths[idx]) for idx, field in enumerate(FIELDS))
    sep = "-+-".join("-" * width for width in widths)
    print(header)
    print(sep)
    for row in rendered:
        print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def has_action_noise(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    return "NormalActionNoise" in source and "action_noise=action_noise" in source


def declares_no_action_noise(row: dict[str, Any]) -> bool:
    noise_type = str(row.get("action_noise_type") or "").strip().lower()
    sigma = row.get("action_noise_sigma")
    return noise_type in {"none", "no", "false", "0"} and float(sigma or 0.0) == 0.0


def ppo_has_action_noise(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    return "NormalActionNoise" in source or "action_noise" in source


def unique_values(rows: list[dict[str, Any]], field: str) -> set[Any]:
    return {row.get(field) for row in rows}


def build_warnings(rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    row_by_method = {str(row.get("algorithm")): row for row in rows}
    td3_row = row_by_method.get("TD3", {})
    if not has_action_noise(TRAINING_SCRIPTS["TD3"]) and not declares_no_action_noise(td3_row):
        warnings.append("WARNING: TD3 action noise setting is ambiguous.")
    if not has_action_noise(TRAINING_SCRIPTS["DDPG"]):
        warnings.append("WARNING: DDPG has no explicit exploration action_noise.")
    if ppo_has_action_noise(TRAINING_SCRIPTS["PPO"]):
        warnings.append("WARNING: PPO appears to contain external action noise; PPO should rely on its stochastic policy.")
    if len(unique_values(rows, "total_timesteps")) > 1:
        warnings.append("WARNING: TOTAL_TIMESTEPS is not consistent across algorithms.")
    if len(unique_values(rows, "seed")) > 1:
        warnings.append("WARNING: SEED is not consistent across algorithms.")
    for field in EES_CONFIG_KEYS:
        if len(unique_values(rows, field)) > 1:
            warnings.append(f"WARNING: {field} is not consistent across algorithms.")
    return warnings


def main() -> None:
    ies_defaults = read_ies_defaults(ROOT / "ies_config.py")
    rows = [
        build_row(algorithm, path, ies_defaults)
        for algorithm, path in TRAINING_SCRIPTS.items()
    ]

    print("Algorithm parameter audit")
    print_table(rows)

    warnings = build_warnings(rows)
    if warnings:
        print("\nWarnings")
        for warning in warnings:
            print(warning)
    else:
        print("\nWarnings")
        print("No parameter audit warnings.")


if __name__ == "__main__":
    main()
