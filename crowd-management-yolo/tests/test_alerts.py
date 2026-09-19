import csv
import json

from crowd_management.alerts import AlertManager
from crowd_management.zones import Status, ZoneState


def state(status, count=10, capacity=20, name="Gate"):
    return {name: ZoneState(name, count, capacity, status)}


def test_safe_zone_raises_no_alert():
    am = AlertManager(cooldown=10)
    assert am.evaluate(state(Status.SAFE), now=0) == []


def test_escalation_alerts_immediately_at_each_level():
    am = AlertManager(cooldown=100)
    warn = am.evaluate(state(Status.WARNING, 12), now=0)
    danger = am.evaluate(state(Status.DANGER, 17), now=1)  # far inside the cooldown
    assert [a.level for a in warn] == ["WARNING"]
    assert [a.level for a in danger] == ["DANGER"]
    assert "17/20" in danger[0].message and "85%" in danger[0].message


def test_cooldown_suppresses_repeats_then_allows_them():
    am = AlertManager(cooldown=10)
    assert len(am.evaluate(state(Status.DANGER), now=0)) == 1
    assert am.evaluate(state(Status.DANGER), now=5) == []
    assert len(am.evaluate(state(Status.DANGER), now=10)) == 1


def test_recovery_emits_one_info_alert():
    am = AlertManager(cooldown=10)
    am.evaluate(state(Status.DANGER), now=0)
    recovered = am.evaluate(state(Status.SAFE, 2), now=1)
    assert [a.level for a in recovered] == ["INFO"]
    assert am.evaluate(state(Status.SAFE, 2), now=2) == []


def test_zones_are_tracked_independently():
    am = AlertManager(cooldown=10)
    states = {**state(Status.DANGER, name="A"), **state(Status.SAFE, name="B")}
    alerts = am.evaluate(states, now=0)
    assert [a.zone for a in alerts] == ["A"]


def test_log_file_and_csv_export(tmp_path):
    log = tmp_path / "logs" / "alerts.jsonl"
    am = AlertManager(cooldown=10, log_path=str(log))
    am.evaluate(state(Status.WARNING, 12), now=0)
    am.evaluate(state(Status.DANGER, 17), now=1)
    lines = log.read_text().strip().splitlines()
    assert [json.loads(l)["level"] for l in lines] == ["WARNING", "DANGER"]

    out = tmp_path / "history.csv"
    am.export_csv(str(out))
    rows = list(csv.reader(out.open()))
    assert rows[0][:3] == ["timestamp", "zone", "level"] and len(rows) == 3


def test_reset_forgets_previous_levels():
    am = AlertManager(cooldown=100)
    am.evaluate(state(Status.DANGER), now=0)
    am.reset()
    assert len(am.evaluate(state(Status.DANGER), now=1)) == 1
