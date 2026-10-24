"""Render exact SVG diagrams/charts from the checked-in CSVs (stdlib only)."""

import csv
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
BG, PANEL, TEXT, MUTED, GRID = "#09151f", "#102332", "#edf7fa", "#a6bbc9", "#294152"
BLUE, GREEN, GOLD = "#75bfff", "#54e2be", "#ffca7d"


def text(x, y, value, size=16, color=TEXT, anchor="start", weight=400):
    return (f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>')


def line(x1, y1, x2, y2, color=GRID, width=1, dash=""):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>'


def rect(x, y, w, h, fill=PANEL, stroke=GRID, radius=14):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'


def save(name, width, height, content, description):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(description)}">'
           f'<title>{escape(description)}</title><rect width="100%" height="100%" fill="{BG}"/>'
           '<g font-family="Inter, DejaVu Sans, Arial, sans-serif">' + content + '</g></svg>')
    (ASSETS / name).write_text(svg, encoding="utf-8")


def read(path):
    with (ROOT / path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def chart_panel(x, y, width, height, title, series, *, xlabel="updates", ylabel="loss"):
    # Each series: label, color, [(x,y), ...]. Draw a zero-based linear axis.
    left, top, right, bottom = x + 65, y + 90, x + width - 25, y + height - 60
    xmax = max(p[0] for _, _, points in series for p in points) or 1
    ymax = max(p[1] for _, _, points in series for p in points) * 1.1 or 1
    result = rect(x, y, width, height) + text(x + 22, y + 33, title, 20, weight=650)
    for i, (label, color, _) in enumerate(series):
        lx = x + 24 + i * 190
        result += line(lx, y + 55, lx + 22, y + 55, color, 3)
        result += text(lx + 30, y + 60, label, 13, MUTED)
    for i in range(5):
        yy = bottom - (bottom - top) * i / 4
        value = ymax * i / 4
        result += line(left, yy, right, yy)
        result += text(left - 10, yy + 5, f"{value:.1f}", 12, MUTED, "end")
    for i in range(5):
        xx = left + (right - left) * i / 4
        result += text(xx, bottom + 24, f"{xmax*i/4:.0f}", 12, MUTED, "middle")
    result += text((left + right) / 2, bottom + 48, xlabel, 13, MUTED, "middle")
    result += text(left, top - 10, ylabel, 12, MUTED)
    for _, color, points in series:
        coords = " ".join(f"{left+(right-left)*a/xmax:.2f},{bottom-(bottom-top)*b/ymax:.2f}" for a, b in points)
        result += f'<polyline fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" points="{coords}"/>'
        for a, b in points:
            result += f'<circle cx="{left+(right-left)*a/xmax:.2f}" cy="{bottom-(bottom-top)*b/ymax:.2f}" r="3" fill="{color}"/>'
    return result


def banner():
    out = text(52, 54, "SELECTIVE STATE SPACES · FROM SCRATCH", 15, GREEN, weight=650)
    out += text(48, 127, "MAMBA / NUMPY", 62, weight=750)
    out += text(52, 174, "Read the math. Train the model. Carry the state.", 24, MUTED)
    labels = [("NUMPY ONLY", BLUE), ("MANUAL GRADIENTS", GREEN), ("STREAMING INFERENCE", GOLD)]
    for i, (label, color) in enumerate(labels):
        xx = 52 + i * 315
        out += rect(xx, 213, 290, 44, fill=BG, stroke=GRID, radius=22)
        out += text(xx + 145, 241, label, 14, color, "middle", 650)
    # A bounded-state signal motif, purely illustrative.
    for i in range(12):
        xx = 840 + (i % 4) * 42
        yy = 45 + (i // 4) * 38
        out += rect(xx, yy, 27, 24, fill=GREEN if i in (2, 5, 8) else PANEL, stroke=GRID, radius=5)
    save("banner.svg", 1100, 295, out, "Mamba NumPy: selective state spaces from scratch")


def architecture():
    def box(x, y, w, h, title, sub="", color=BLUE):
        s = rect(x, y, w, h, stroke=color)
        s += text(x + w / 2, y + 29, title, 18, TEXT, "middle", 650)
        if sub:
            s += text(x + w / 2, y + 53, sub, 13, MUTED, "middle")
        return s

    def arrow(x1, y1, x2, y2, color=BLUE):
        return line(x1, y1, x2, y2, color, 2) + f'<circle cx="{x2}" cy="{y2}" r="3" fill="{color}"/>'

    s = text(40, 43, "Inside one Mamba residual block", 26, weight=650)
    s += text(40, 71, "E = expand × d_model     N = d_state     R = dt_rank", 15, MUTED)
    s += box(340, 95, 400, 64, "RMSNorm → input projection", "(batch, time, d_model) → 2E")
    s += arrow(450, 159, 280, 198) + arrow(650, 159, 850, 198, GREEN)
    s += box(90, 198, 390, 72, "Causal depthwise convolution", "left-padded K taps → SiLU → u", BLUE)
    s += box(735, 198, 235, 72, "Gate branch", "SiLU(z)", GREEN)
    s += arrow(280, 270, 280, 300)
    s += box(90, 300, 390, 75, "Input-dependent projections", "u → low-rank Δ, Bₜ, Cₜ", BLUE)
    s += arrow(280, 375, 280, 413)
    s += box(65, 413, 490, 104, "Selective state-space recurrence", "hₜ = exp(Δₜ A) ⊙ hₜ₋₁ + Δₜ Bₜ uₜ", GOLD)
    s += text(310, 494, "yₜ = Σ Cₜ hₜ + D ⊙ uₜ", 16, TEXT, "middle")
    s += box(625, 338, 345, 77, "Persistent recurrent state", "SSM: (B,E,N)  ·  conv: (B,K−1,E)", GOLD)
    s += arrow(625, 382, 555, 450, GOLD)
    s += line(852, 270, 1000, 270, GREEN, 2)
    s += line(1000, 270, 1000, 555, GREEN, 2)
    s += line(310, 517, 310, 555, BLUE, 2)
    s += line(310, 555, 520, 555, BLUE, 2)
    s += line(1000, 555, 580, 555, GREEN, 2)
    s += box(520, 533, 60, 47, "×", color=GREEN)
    s += arrow(550, 580, 550, 610)
    s += box(335, 610, 430, 75, "Output projection + residual", "E → d_model; add the block input", BLUE)
    s += text(40, 733, "Stack blocks → final RMSNorm → tied vocabulary head → next-token logits", 18, TEXT)
    s += text(40, 765, "All projections, normalization, state updates, gradients, and optimization run in NumPy.", 14, MUTED)
    save("architecture.svg", 1050, 795, s, "Mamba-1 block architecture and recurrent state shapes")


def training():
    s = text(35, 40, "Real training runs · two small teaching tasks", 26, weight=650)
    s += text(35, 68, "One seed. Fixed evaluation batches. Raw logs and checkpoints are included.", 15, MUTED)
    for index, (folder, name) in enumerate((("text_demo", "Character language model"), ("recall_demo", "Selective recall"))):
        rows = read(f"results/{folder}/metrics.csv")
        series = [("train probe", BLUE, [(int(r["step"]), float(r["train_loss"])) for r in rows]),
                  ("validation", GREEN, [(int(r["step"]), float(r["val_loss"])) for r in rows])]
        s += chart_panel(25 + index * 535, 95, 510, 365, name, series)
    save("training.svg", 1100, 485, s, "Training and validation losses measured in the included demo runs")


def benchmark():
    rows = read("results/benchmark/timings.csv")
    series = [(name, color, [(int(r["length"]), float(r["median_ms"])) for r in rows if r["method"] == name])
              for name, color in (("sequential", BLUE), ("parallel_demo", GOLD), ("streaming", GREEN))]
    s = text(35, 40, "CPU reference implementations", 26, weight=650)
    s += text(35, 68, "Same model and inputs; median wall time. Environment and raw CSV are included.", 15, MUTED)
    s += chart_panel(25, 95, 1000, 390, "End-to-end chunk inference", series,
                     xlabel="sequence length (tokens)", ylabel="milliseconds")
    s += text(40, 516, "The prefix demo uses O(L log L) work; these timings do not represent fused Mamba GPU kernels.", 14, MUTED)
    save("benchmark.svg", 1050, 540, s, "Measured CPU inference timings for three NumPy execution paths")


def recall():
    rows = read("results/recall_evaluation/delay_sweep.csv")
    ranges = [(4, 9), (12, 20), (32, 48), (64, 96)]
    top, bottom, left, right = 125, 365, 95, 1000
    s = text(35, 40, "Does the model use its recurrent state?", 26, weight=650)
    s += text(35, 68, "Held-out procedural queries; trained only on delays 4–9. The ablation is not retrained.", 15, MUTED)
    for i in range(5):
        yy = bottom - i * (bottom - top) / 4
        s += line(left, yy, right, yy)
        s += text(left - 12, yy + 5, f"{25*i}%", 13, MUTED, "end")
    for i, (low, high) in enumerate(ranges):
        center = left + (right - left) * (i + 0.5) / 4
        for j, (mode, color) in enumerate((("trained", GREEN), ("state_readout_disabled", GOLD))):
            row = next(r for r in rows if r["mode"] == mode and int(r["min_delay"]) == low)
            acc = float(row["query_accuracy"])
            xx = center - 64 + j * 66
            h = acc * (bottom - top)
            s += rect(xx, bottom - h, 58, max(h, 0.5), fill=color, stroke=color, radius=3)
            s += text(xx + 29, bottom - h - 9, f"{100*acc:.1f}", 13, TEXT, "middle")
        s += text(center, bottom + 27, f"{low}–{high}", 15, TEXT, "middle")
    chance_y = bottom - 0.25 * (bottom - top)
    s += line(left, chance_y, right, chance_y, BLUE, 2, "6 5")
    s += text((left + right) / 2, bottom + 55, "intervening distractor tokens", 14, MUTED, "middle")
    s += text(110, 106, "Trained model", 14, GREEN) + text(350, 106, "State readout disabled", 14, GOLD)
    s += text(700, 106, "Dashed line: 25% chance", 14, BLUE)
    s += text(40, 464, "Accuracy depends on this task and delay. See raw counts and limitations in docs/experiments.md.", 14, MUTED)
    save("recall.svg", 1050, 490, s, "Selective-recall accuracy by delay, with trained state-readout ablation")


def main():
    ASSETS.mkdir(exist_ok=True)
    banner()
    architecture()
    training()
    benchmark()
    recall()
    print("Rendered SVG assets in", ASSETS)


if __name__ == "__main__":
    main()
