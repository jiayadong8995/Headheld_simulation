"""配置加载、深合并、场景覆盖与参数来源（SPEC/USER_MODEL/ASSUMED/MEASURED）标注。"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "config" / "base.yaml"
SCENARIO_CONFIG = ROOT / "config" / "scenarios.yaml"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并 override 到 base 的副本上（dict 合并，其它类型直接替换）。"""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_base(path: str | Path = BASE_CONFIG) -> Dict[str, Any]:
    return load_yaml(path)


def load_scenarios(path: str | Path = SCENARIO_CONFIG) -> Dict[str, Any]:
    return load_yaml(path)


def load_config(base_path: str | Path = BASE_CONFIG,
                overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """加载 base 配置并叠加命令行/代码给定的覆盖（点号分隔键亦可）。"""
    cfg = load_base(base_path)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def apply_dotted_override(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """支持 `a.b.c=3` 形式的覆盖。"""
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def provenance_map(cfg: Dict[str, Any]) -> Dict[str, str]:
    """返回 {参数路径: 来源标签}；未标注者默认 ASSUMED。"""
    prov = cfg.get("provenance", {}) or {}
    out: Dict[str, str] = {}
    for label in ("SPEC", "USER_MODEL", "MEASURED"):
        for entry in prov.get(label, []) or []:
            out[entry] = label
    return out


def provenance_of(cfg: Dict[str, Any], dotted: str) -> str:
    pm = provenance_map(cfg)
    node = pm
    for key in dotted.split("."):
        if isinstance(node, dict):
            if key in node and isinstance(node[key], str):
                return node[key]
            node = node.get(key, {})
    return pm.get(dotted, "ASSUMED")


def resolve_scenario(base_cfg: Dict[str, Any], scenarios_cfg: Dict[str, Any],
                     scenario_id: str) -> Dict[str, Any]:
    """把某个场景的定义覆盖到 base 配置上，得到该次运行完全解析后的配置。"""
    spec = (scenarios_cfg.get("scenarios", {}) or {}).get(scenario_id)
    if spec is None:
        raise KeyError(f"未知场景 {scenario_id}")
    overrides: Dict[str, Any] = {}
    if "friction_schedule" in spec:
        pass  # 由 scenarios 运行器处理
    if "plant_overrides" in spec:
        for key, val in spec["plant_overrides"].items():
            if key in ("wet", "dry", "ideal"):
                continue
            overrides.setdefault("mechanics", {})[key] = val
    if "brush" in spec:
        brush = {k: v for k, v in spec["brush"].items()
                 if k not in ("bias_schedule",)}
        overrides.setdefault("mechanics", {})["brush"] = brush
    if "sensors" in spec:
        sens = {k: v for k, v in spec["sensors"].items() if k != "ideal"}
        for key, val in sens.items():
            overrides.setdefault("sensors", {})[key] = val
    if "actuator" in spec:
        act = {k: v for k, v in spec["actuator"].items() if k != "ideal"}
        if act:
            overrides.setdefault("actuator", {}).update(act)
    cfg = deep_merge(base_cfg, overrides)
    cfg["_scenario"] = scenario_id
    cfg["_scenario_spec"] = spec
    return cfg
