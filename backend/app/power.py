"""Power and energy: an optional wall-power meter, and energy per token from a dedicated measurement.

Sources, best first:
  wall   a smart plug or meter on the device's power cable (whole device, what you pay for). Any URL that returns JSON
         with the current watts: Shelly Gen2+ (/rpc/Switch.GetStatus?id=0 -> apower), Shelly Gen1 (/status ->
         meters[0].power), Tasmota (/cm?cmnd=Status%208 -> StatusSNS.ENERGY.Power), or a custom JSON field path.
  parts  the GPU's own sensor (nvidia-smi) plus the CPU package (RAPL). Leaves out RAM, disk, fans and the PSU's losses.
The energy phase measures idle power with the model loaded, then power during sustained generation, so results show
both gross energy per token (everything the device drew) and net (above idle: what generating costs on top).
"""
import math
import re

import httpx

_FIELDS = (("apower",), ("meters", 0, "power"), ("StatusSNS", "ENERGY", "Power"), ("ENERGY", "Power"), ("power",), ("watts",))


def _dig(d, path):
    for p in path:
        if isinstance(p, int):
            if not isinstance(d, list) or len(d) <= p:
                return None
        elif not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def parse_watts(data, field: str = "") -> float | None:
    """Watts from a meter's JSON reply: a custom dotted path (e.g. 'emeters.0.power'), else known shapes."""
    paths = [tuple(int(p) if p.isdigit() else p for p in field.split("."))] if field else list(_FIELDS)
    for path in paths:
        v = _dig(data, path)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0:
            return float(v)
    return None


def meter_config(settings: dict) -> tuple[str, str] | None:
    url = (settings.get("power_meter_url") or "").strip()
    if not url:
        return None
    if not re.match(r"^https?://[^\s]+$", url):
        raise ValueError("The power meter address must be an http(s) URL")
    return url, (settings.get("power_meter_field") or "").strip()


async def read_wall(client: httpx.AsyncClient, url: str, field: str = "") -> float | None:
    try:
        r = await client.get(url, timeout=1.5)
        r.raise_for_status()
        return parse_watts(r.json(), field)
    except (httpx.HTTPError, ValueError):
        return None


def _stats(v: list[float]) -> dict | None:
    if not v:
        return None
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return {"avg": round(m, 2), "sd": round(sd, 2), "n": len(v)}


def energy_summary(idle: list[dict], load: list[dict], tokens: int, seconds: float) -> dict:
    """Energy per generated token from telemetry samples taken while idle (model loaded) and while generating."""
    if not tokens or seconds <= 0:
        return {"error": "No tokens were generated"}
    tps = tokens / seconds
    out: dict = {"tokens": tokens, "seconds": round(seconds, 2), "tps": round(tps, 2), "sources": {}}
    sources = {"wall": ["wall_power"], "gpu": ["power"], "cpu": ["cpu_power"], "parts": ["power", "cpu_power"]}
    for name, keys in sources.items():
        def watts(ss):
            return [sum(s[k] for k in keys) for s in ss if all(k in s for k in keys)]
        li, ll = _stats(watts(idle)), _stats(watts(load))
        if not ll or ll["n"] < 3:
            continue
        j = ll["avg"] / tps
        src = {"idle_w": li, "load_w": ll, "j_per_token": round(j, 4), "tokens_per_wh": round(3600 / j, 1) if j else None}
        if li:
            net = max(0.0, ll["avg"] - li["avg"])
            src["net_j_per_token"] = round(net / tps, 4)
        out["sources"][name] = src
    if "parts" in out["sources"] and "cpu" not in out["sources"]:
        out["sources"].pop("parts")  # GPU only: "parts" would just repeat it
    primary = next((k for k in ("wall", "parts", "gpu") if k in out["sources"]), None)
    out["primary"] = primary
    out["primary_label"] = {"wall": "whole device (wall meter)", "parts": "GPU + CPU package (sensors)", "gpu": "GPU only (sensor)"}.get(primary)
    if not primary:
        out["note"] = "No power reading on this device: the GPU driver reports none, CPU energy counters aren't readable, and no wall meter is set (Settings → Energy & cost)."
    return out


def costs(energy: dict | None, tg: float | None, prices: dict) -> dict:
    """Running cost per 1M generated tokens (electricity) and speed per hardware cost, from prices set on the device."""
    out: dict = {"currency": prices.get("currency") or ""}
    src = (energy or {}).get("sources", {}).get((energy or {}).get("primary") or "")
    kwh = prices.get("electricity_price_kwh")
    if src and kwh:
        out["electricity_per_mtok"] = round(src["j_per_token"] * 1e6 / 3.6e6 * float(kwh), 4)
    dev = prices.get("device_price")
    if tg and dev:
        out["tps_per_1k_price"] = round(tg / float(dev) * 1000, 2)
    return out
