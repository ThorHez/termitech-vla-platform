"""Normalize per-interface heartbeat reports and keep observed daily usage."""

from datetime import datetime, timedelta, timezone


DAY_TIMEZONE_NAME = "Asia/Shanghai"
DAY_TIMEZONE = timezone(timedelta(hours=8), DAY_TIMEZONE_NAME)
MAX_INTERFACES = 64
MAX_COUNTER_BYTES = (1 << 64) - 1
RATE_FIELDS = ("rx_bytes_per_second", "tx_bytes_per_second")


def rate(value):
    if type(value) in (int, float) and 0 <= value <= MAX_COUNTER_BYTES:
        return value
    return None


def _counter(value):
    return type(value) is int and 0 <= value <= MAX_COUNTER_BYTES


def _label(value):
    return isinstance(value, str) and 0 < len(value) <= 64 and value.isprintable()


def _report(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get("interfaces"), list):
        return {}
    interfaces = {}
    for item in raw["interfaces"][:MAX_INTERFACES]:
        if not isinstance(item, dict) or not _label(item.get("name")):
            continue
        name = item["name"]
        if name in interfaces or not all(_counter(item.get(key)) for key in ("rx_bytes", "tx_bytes")):
            continue
        ifindex = item.get("ifindex")
        interfaces[name] = {
            "name": name,
            "ifindex": ifindex if type(ifindex) is int and 0 < ifindex < 2**31 else None,
            "rx_bytes": item["rx_bytes"],
            "tx_bytes": item["tx_bytes"],
            **{key: rate(item.get(key)) for key in RATE_FIELDS},
        }
    default_interface = raw.get("default_interface")
    return {
        "boot_id": raw["boot_id"] if _label(raw.get("boot_id")) else None,
        "default_interface": default_interface if _label(default_interface) and default_interface in interfaces else None,
        "interfaces": list(interfaces.values()),
    }


def _day_record(name):
    return {
        "name": name,
        "day_rx_bytes": None,
        "day_tx_bytes": None,
        "day_partial": True,
        "observed_from": None,
        "observed_until": None,
    }


def _timestamp(value):
    if _label(value):
        try:
            if datetime.fromisoformat(value).tzinfo is not None:
                return value
        except ValueError:
            pass
    return None


def _previous_usage(network, date):
    today = network.get("today")
    if not isinstance(today, dict) or today.get("date") != date or not isinstance(today.get("interfaces"), list):
        return {}
    records = {}
    for item in today["interfaces"][:MAX_INTERFACES]:
        if not isinstance(item, dict) or not _label(item.get("name")):
            continue
        record = _day_record(item["name"])
        for key in ("day_rx_bytes", "day_tx_bytes"):
            value = item.get(key)
            if type(value) is int and 0 <= value < 2**128:
                record[key] = value
        for key in ("observed_from", "observed_until"):
            record[key] = _timestamp(item.get(key))
        record["day_partial"] = item.get("day_partial") is not False
        records[record["name"]] = record
    return records


def with_network_metrics(metrics, previous_metrics, reported_at, previous_at, max_gap_seconds):
    """Called under the node's heartbeat lock, before replacing its latest metrics."""
    raw = metrics.get("network")
    previous_raw = (previous_metrics or {}).get("network")
    if raw is None and not previous_raw:
        return metrics
    current = _report(raw)
    previous = _report(previous_raw)
    moment = datetime.fromisoformat(reported_at)
    previous_moment = datetime.fromisoformat(previous_at)
    start_of_day = moment.astimezone(DAY_TIMEZONE).replace(hour=0, minute=0, second=0, microsecond=0)
    date = start_of_day.date().isoformat()
    previous_usage = _previous_usage(previous_raw if isinstance(previous_raw, dict) else {}, date)
    previous_interfaces = {item["name"]: item for item in previous.get("interfaces", [])}
    continuous = (
        current.get("boot_id") is not None
        and current["boot_id"] == previous.get("boot_id")
        and previous_moment >= start_of_day
        and timedelta(0) < moment - previous_moment <= timedelta(seconds=max_gap_seconds)
    )
    usage = {}
    for item in current.get("interfaces", []):
        name = item["name"]
        record = previous_usage.pop(name, _day_record(name))
        baseline = previous_interfaces.get(name)
        if (
            continuous
            and baseline is not None
            and baseline["ifindex"] == item["ifindex"]
            and item["rx_bytes"] >= baseline["rx_bytes"]
            and item["tx_bytes"] >= baseline["tx_bytes"]
        ):
            if record["observed_from"] is None:
                record["observed_from"] = previous_at
                record["day_partial"] = previous_moment != start_of_day
            record["observed_until"] = reported_at
            record["day_rx_bytes"] = (record["day_rx_bytes"] or 0) + item["rx_bytes"] - baseline["rx_bytes"]
            record["day_tx_bytes"] = (record["day_tx_bytes"] or 0) + item["tx_bytes"] - baseline["tx_bytes"]
        else:
            record["day_partial"] = True
        usage[name] = record
    for name, record in previous_usage.items():
        if len(usage) >= MAX_INTERFACES:
            break
        usage[name] = {**record, "day_partial": True}
    result = {key: value for key, value in metrics.items() if key != "network"}
    if current or usage:
        result["network"] = {
            **(current or {"boot_id": None, "default_interface": None, "interfaces": []}),
            "today": {"date": date, "interfaces": list(usage.values())},
        }
    return result


def network_snapshot(metrics):
    report = _report(metrics.get("network"))
    if not report.get("interfaces"):
        return None
    return {
        "default_interface": report["default_interface"],
        "interfaces": [
            {"name": item["name"], **{key: item[key] for key in RATE_FIELDS}} for item in report["interfaces"]
        ],
    }


def network_summary(metrics, end, historical_names, historical_default, freshness_seconds):
    raw = metrics.get("network")
    current = _report(raw)
    date = end.astimezone(DAY_TIMEZONE).date().isoformat()
    today = _previous_usage(raw if isinstance(raw, dict) else {}, date)
    current_names = [item["name"] for item in current.get("interfaces", [])]
    names = set(current_names) | today.keys() | historical_names
    oldest_fresh = end - timedelta(seconds=freshness_seconds)
    records = []
    for name in sorted(names):
        record = today.get(name, _day_record(name))
        observed_until = record["observed_until"]
        if observed_until is None or datetime.fromisoformat(observed_until) < oldest_fresh:
            record["day_partial"] = True
        records.append(record)
    return {
        "default_interface": current["default_interface"] if current_names else historical_default,
        "day_date": date,
        "day_timezone": DAY_TIMEZONE_NAME,
        "interfaces": records,
    }
