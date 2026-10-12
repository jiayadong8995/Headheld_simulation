"""真值隔离：控制器/估计器/传感器不得读取 plant 真值（任务书 §10-6）。"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SIM = ROOT / "sim"
FORBIDDEN_MODULES = ["controllers.py", "estimator.py", "sensors.py", "actuator.py"]
TRUTH_TOKENS = ("_true", "truth_snapshot", "plant.")


def test_no_plant_import_in_control_path():
    for fname in FORBIDDEN_MODULES:
        src = (SIM / fname).read_text(encoding="utf-8")
        assert "import plant" not in src and "from .plant" not in src, \
            f"{fname} 不得导入 plant"
        assert "truth_snapshot" not in src, f"{fname} 不得访问真值快照"


def test_truth_access_only_through_oracle_dict():
    """任何对真值字段的**下标访问**都必须显式出现在 oracle 表达式中。"""
    import re
    src = (SIM / "controllers.py").read_text(encoding="utf-8")
    pattern = re.compile(r"\[\s*[\"'][a-zA-Z_]*_true[\"']\s*\]")
    offenders = []
    for i, line in enumerate(src.splitlines(), 1):
        if pattern.search(line) and "oracle" not in line:
            offenders.append((i, line.strip()))
    assert not offenders, f"未经 oracle 的真值访问: {offenders}"


def test_deployable_controllers_do_not_require_oracle():
    from sim.controllers import CONTROLLER_CLASSES
    for name in ("C0", "C1", "C2", "C3"):
        assert CONTROLLER_CLASSES[name].requires_oracle is False
    for name in ("C4a", "C4b"):
        assert CONTROLLER_CLASSES[name].requires_oracle is True


def test_oracle_not_used_in_m2_default_runs():
    """默认部署比较（C0–C3）运行时不应接收任何 oracle 数据。"""
    from sim.config import load_base, load_scenarios, resolve_scenario
    from sim.scenarios import run_case
    base = load_base()
    sc = load_scenarios()
    cfg = resolve_scenario(base, sc, "S03")
    res = run_case(cfg, cfg["_scenario_spec"], "C3", seed=5)
    # C3 的信息字典里只允许名义阻力，不允许真值回灌
    assert res.meta["controller"] == "C3"
    assert not any("oracle" in k for k in res.rec.keys())
