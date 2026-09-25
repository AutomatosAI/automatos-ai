"""Chart geometry for the Markets promo, computed from the live data fetched on
2026-09-23 (assets/data/*.json). Every price, odds figure and ledger row on
screen comes from here — nothing is typed by hand (the Markets honesty rule).

Writes assets/charts/{cockpit,plan,replay}.svg (shapes only; all text is HTML)
and assets/data/derived.json (numbers + label positions for build.py).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "assets" / "data"
OUT = ROOT / "assets" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

BULL, BEAR = "#81a9da", "#e96235"          # Automatos Studio Dark: steel up, orange down
PROFIT, LOSS, GOLD = "#90af5a", "#f07a50", "#e96235"
GRID = "rgba(240,232,219,0.07)"

candles = json.loads((DATA / "btc-1h.json").read_text())["data"]["candles"]
ledger = json.loads((DATA / "ledger-resolved.json").read_text())["data"]
ledger_all = json.loads((DATA / "ledger.json").read_text())["data"]
plan_row = ledger_all["rows"][0]


def svg_candles(cs, w, h, lo, hi, pad_x=6, extra=""):
    """Candles scaled into a w×h box over the price domain [lo, hi]."""
    n = len(cs)
    slot = (w - 2 * pad_x) / n
    body = max(3.0, slot * 0.62)
    y = lambda p: h - (p - lo) / (hi - lo) * h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    for i in range(1, 5):
        gy = round(h * i / 5, 1)
        parts.append(f'<line x1="0" y1="{gy}" x2="{w}" y2="{gy}" stroke="{GRID}" stroke-width="1"/>')
    parts.append(extra)
    for i, c in enumerate(cs):
        cx = pad_x + slot * i + slot / 2
        col = BULL if c["close"] >= c["open"] else BEAR
        top, bot = y(max(c["open"], c["close"])), y(min(c["open"], c["close"]))
        parts.append(f'<line x1="{cx:.1f}" y1="{y(c["high"]):.1f}" x2="{cx:.1f}" y2="{y(c["low"]):.1f}" stroke="{col}" stroke-width="2"/>')
        parts.append(f'<rect x="{cx - body / 2:.1f}" y="{top:.1f}" width="{body:.1f}" height="{max(2.0, bot - top):.1f}" fill="{col}" rx="1"/>')
    parts.append("</svg>")
    return "\n".join(parts), y, slot


def domain(cs, extra_prices=(), pad=0.06):
    lo = min([c["low"] for c in cs] + list(extra_prices))
    hi = max([c["high"] for c in cs] + list(extra_prices))
    span = hi - lo
    return lo - span * pad, hi + span * pad


# ── cockpit: the 60 recorded BTC·1h candles ───────────────────────────────────
lo, hi = domain(candles)
svg, _, _ = svg_candles(candles, 900, 560, lo, hi)
(OUT / "cockpit.svg").write_text(svg)

# ── plan: the last 24 candles with the live plan's zone, stop and T1 ─────────
recent = candles[-24:]
t1 = plan_row["targets"][0]
plo, phi = domain(recent, [plan_row["stop"], t1["price"], plan_row["entryLo"], plan_row["entryHi"]], pad=0.08)
W, H = 900, 520
yp = lambda p: H - (p - plo) / (phi - plo) * H
zone = (f'<rect x="0" y="{yp(plan_row["entryHi"]):.1f}" width="{W}" height="{yp(plan_row["entryLo"]) - yp(plan_row["entryHi"]):.1f}" '
        f'fill="{GOLD}" fill-opacity="0.16"/>'
        f'<line x1="0" y1="{yp(plan_row["stop"]):.1f}" x2="{W}" y2="{yp(plan_row["stop"]):.1f}" stroke="{LOSS}" stroke-width="3" stroke-dasharray="14 10"/>'
        f'<line x1="0" y1="{yp(t1["price"]):.1f}" x2="{W}" y2="{yp(t1["price"]):.1f}" stroke="{PROFIT}" stroke-width="3" stroke-dasharray="14 10"/>')
svg, _, _ = svg_candles(recent, W, H, plo, phi, extra=zone)
(OUT / "plan.svg").write_text(svg)

# ── replay: the first 40 candles, revealed one bar at a time ─────────────────
replay = candles[:40]
rlo, rhi = domain(replay)
svg, _, rslot = svg_candles(replay, 900, 520, rlo, rhi)
(OUT / "replay.svg").write_text(svg)

# ── derived numbers (all computed) ───────────────────────────────────────────
last = candles[-1]
hi60 = max(c["high"] for c in candles)
lo60 = min(c["low"] for c in candles)
chg = (last["close"] - candles[0]["open"]) / candles[0]["open"] * 100
fmt = lambda p: f"{p:,.0f}"
prom = plan_row["promised"]

OUTCOME = {"t1": ("T1 FIRST", "profit"), "t2": ("T2 FIRST", "profit"), "stop": ("STOP FIRST", "loss"),
           "horizon": ("HORIZON", "muted"), "invalidated": ("INVALIDATED", "muted"),
           "never_entered": ("NEVER ENTERED", "muted")}
rows = []
for r in ledger["rows"][:8]:
    label, tone = OUTCOME.get(r["outcome"], (str(r["outcome"]).upper(), "muted"))
    rr = r["realizedR"]
    rows.append({
        "chart": f'{r["asset"]} · {r["tf"].upper()}',
        "setup": r["setupId"].replace("_", " ").upper(),
        "outcome": label, "tone": tone,
        "r": "—" if rr is None else (f"+{rr:.2f}R" if rr >= 0 else f"−{abs(rr):.2f}R"),
        "rtone": "muted" if rr is None else ("profit" if rr >= 0 else "loss"),
        "resolved": datetime.fromtimestamp(r["resolvedTs"], timezone.utc).strftime("%d %b %H:%M"),
    })

derived = {
    "last_close": fmt(last["close"]), "hi60": fmt(hi60), "lo60": fmt(lo60),
    "chg60": f"{'+' if chg >= 0 else '−'}{abs(chg):.1f}%",
    "first_ts": datetime.fromtimestamp(candles[0]["time"], timezone.utc).strftime("%d %b %H:%M"),
    "last_ts": datetime.fromtimestamp(last["time"], timezone.utc).strftime("%d %b %H:%M"),
    "atr": fmt(plan_row["atr"]),
    "plan": {
        "title": f'{plan_row["asset"]} · {plan_row["tf"].upper()} · {plan_row["setupId"].replace("_", " ").upper()} · {plan_row["direction"].upper()}',
        "entry": f'{fmt(plan_row["entryLo"])} – {fmt(plan_row["entryHi"])}',
        "stop": fmt(plan_row["stop"]), "t1": fmt(t1["price"]), "t1r": f'{t1["r"]:.1f}R',
        "y_entry_mid": round(yp((plan_row["entryLo"] + plan_row["entryHi"]) / 2), 1),
        "y_stop": round(yp(plan_row["stop"]), 1), "y_t1": round(yp(t1["price"]), 1),
        "n": prom["n"], "pT1": round(prom["pT1"] * 100), "pStop": round(prom["pStop"] * 100),
        "pHorizon": round(prom["pHorizon"] * 100),
    },
    "ledger_total_logged": ledger_all["total"], "ledger_total_resolved": ledger["total"], "rows": rows,
    "last_resolved": rows[0],
    "replay": {"slot": round(rslot, 3), "n": len(replay)},
}
(DATA / "derived.json").write_text(json.dumps(derived, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in derived.items() if k != "rows"}, indent=1, ensure_ascii=False))
for r in rows:
    print(r)
