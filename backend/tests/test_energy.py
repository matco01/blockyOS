"""Energy per token, meter parsing and costs."""
import pytest

from app import power


@pytest.mark.parametrize("reply,field,watts", [
    ({"id": 0, "apower": 41.5, "voltage": 230}, "", 41.5),                     # Shelly Gen2+
    ({"meters": [{"power": 38.2}]}, "", 38.2),                                  # Shelly Gen1
    ({"StatusSNS": {"ENERGY": {"Power": 55}}}, "", 55.0),                       # Tasmota
    ({"emeters": [{"power": 12.0}]}, "emeters.0.power", 12.0),                  # custom field path
    ({"apower": "n/a"}, "", None),
    ({"power": -3}, "", None),
    ({"meters": []}, "", None),
])
def test_parse_watts(reply, field, watts):
    assert power.parse_watts(reply, field) == watts


def test_meter_url_validation():
    assert power.meter_config({"power_meter_url": ""}) is None
    assert power.meter_config({"power_meter_url": "http://10.0.0.5/status", "power_meter_field": ""}) == ("http://10.0.0.5/status", "")
    with pytest.raises(ValueError):
        power.meter_config({"power_meter_url": "file:///etc/passwd"})


def _samples(n, **w):
    return [dict(w) for _ in range(n)]


def test_energy_per_token_gross_and_net():
    idle = _samples(10, power=20.0, cpu_power=10.0)
    load = _samples(20, power=120.0, cpu_power=30.0)
    e = power.energy_summary(idle, load, tokens=1500, seconds=15.0)  # 100 tok/s
    assert e["primary"] == "parts" and e["tps"] == 100
    parts = e["sources"]["parts"]
    assert parts["j_per_token"] == pytest.approx(1.5)            # 150 W / 100 tok/s
    assert parts["net_j_per_token"] == pytest.approx(1.2)        # (150 - 30) W / 100 tok/s
    assert parts["tokens_per_wh"] == pytest.approx(2400)
    assert e["sources"]["gpu"]["j_per_token"] == pytest.approx(1.2)


def test_wall_meter_wins_and_gpu_only_has_no_parts():
    e = power.energy_summary(_samples(5, power=10.0, wall_power=40.0), _samples(10, power=90.0, wall_power=140.0), 500, 10.0)
    assert e["primary"] == "wall" and "parts" not in e["sources"]
    assert e["sources"]["wall"]["j_per_token"] == pytest.approx(2.8)


def test_no_power_readings():
    e = power.energy_summary(_samples(5), _samples(10), 500, 10.0)
    assert e["primary"] is None and "note" in e
    assert power.energy_summary([], [], 0, 0)["error"]


def test_costs():
    e = power.energy_summary([], _samples(10, wall_power=100.0), 1000, 10.0)  # 1 J/token
    c = power.costs(e, 50.0, {"electricity_price_kwh": 0.3, "device_price": 500, "currency": "EUR"})
    assert c["electricity_per_mtok"] == pytest.approx(1e6 / 3.6e6 * 0.3, rel=1e-3)
    assert c["tps_per_1k_price"] == 100.0 and c["currency"] == "EUR"
    assert power.costs(None, None, {}) == {"currency": ""}
