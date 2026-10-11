"""Build every figure and table used in the report from results/.

    python src/analysis.py        # -> report/figures/*, report/tables/*.md, results/all_runs.csv
"""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, ROOT, dilate, load_annotations, load_sample, splits, to_np  # noqa: E402

FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"
MAC_BACKUP = ROOT / "results_mac_backup"      # dev runs first produced on the M2 Pro (reproducibility check)

# fixed colour per method (entity), reference categorical palette order
COLORS = {"lama": "#2a78d6", "sd15": "#eb6834", "hybrid": "#1baf7a", "sdxl": "#eda100",
          "ip2p": "#e87ba4", "telea": "#008300", "kontext": "#4a3aa7", "identity": "#8a8983"}
NAMES = {"identity": "Input (no edit)", "telea": "Telea", "lama": "LaMa", "sd15": "SD1.5-Inpaint",
         "sdxl": "SDXL-Inpaint", "ip2p": "InstructPix2Pix", "kontext": "FLUX.1 Kontext",
         "hybrid": "P1: LaMa→SD1.5 (ours)"}
SHORT = {"identity": "Input", "telea": "Telea", "lama": "LaMa", "sd15": "SD1.5", "sdxl": "SDXL",
         "ip2p": "IP2P", "kontext": "Kontext", "hybrid": "P1"}
ORDER = ["identity", "telea", "lama", "sd15", "sdxl", "hybrid", "ip2p", "kontext"]
MASK_FREE = {"ip2p", "kontext"}
N_TEST = 160
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TYPES = ["promo_claim", "rank_badge", "store_watermark", "contact_info", "exaggerated_effect", "qr_code"]
TYPE_NAMES = {"promo_claim": "Promo claim", "rank_badge": "Rank badge", "store_watermark": "Store watermark",
              "contact_info": "Contact info", "exaggerated_effect": "Effect claim", "qr_code": "QR code"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 9.5, "axes.titlecolor": INK,
    "figure.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 2, "lines.markersize": 5,
})


def S(exp, run, root=RESULTS):
    p = root / exp / run / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def per_image(exp, run):
    p = RESULTS / exp / run / "metrics.csv"
    if not p.exists():
        return {}
    with open(p) as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def fmt(v, k):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if k in ("text_res", "qr_res", "new_text_img"):
        return f"{100 * v:.1f}"
    if k.startswith("psnr"):
        return f"{v:.2f}"
    return f"{v:.4f}" if k.startswith("lpips") else f"{v:.3f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + [":-:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ tables

# metric column headers: name on the first line, unit and direction (↓ lower / ↑ higher is better) on the second
HEAD = {"text_res": "Text res.<br>(%) ↓", "qr_res": "QR res.<br>(%) ↓", "new_text_img": "New text<br>(%) ↓",
        "psnr_in": "PSNR-in<br>(dB) ↑", "lpips_in": "LPIPS-in<br>↓", "psnr_out": "PSNR-out<br>(dB) ↑",
        "lpips_out": "LPIPS-out<br>↓", "ssim": "SSIM<br>↑"}
MAIN_COLS = [(k, HEAD[k]) for k in ("text_res", "qr_res", "new_text_img", "psnr_in", "lpips_in", "psnr_out",
                                    "lpips_out", "ssim")]


def table_main():
    rows = []
    for m in ORDER:
        s = S("main", m)
        if s:
            name = NAMES[m] + ("" if s["n"] == N_TEST else f" (n={s['n']})")
            rows.append([name] + [fmt(s.get(k), k) for k, _ in MAIN_COLS])
    return md_table(["Method"] + [h for _, h in MAIN_COLS], rows)


def run_median_time(m):
    """Median seconds/image over the test run, first image dropped (it includes model loading)."""
    p = RESULTS / "main" / m / "meta.json"
    if not p.exists():
        return None
    meta = json.loads(p.read_text())
    ts = [meta["times"][i] for i in meta["ids"] if i in meta["times"]][1:]
    return statistics.median(ts) if ts else None


def table_speed():
    bench = {}
    dev = "NVIDIA A100"
    for f in sorted((RESULTS / "speed").glob("*.json")):
        d = json.loads(f.read_text())
        bench, dev = d["seconds_per_image"], d["device"]
    front = pareto_front()
    rows = []
    for m in ORDER[1:]:
        med = run_median_time(m)
        if med is None and m not in bench:
            continue
        res, steps = INFER_CONFIG[m]
        rows.append([NAMES[m], res, steps, f"{med:.2f}" if med is not None else "–",
                     f"{bench[m]:.2f}" if m in bench else "–", "yes" if m in front else "no"])
    return md_table(["Method", "Inference resolution<br>(long side)", "Denoising steps",
                     "Median s / img<br>(159 test images)", "Mean s / img<br>(10-image benchmark)",
                     "On the Pareto<br>front?"], rows)


def pareto_front():
    """Methods for which no other method is at least as fast and at least as good on both LPIPS measures."""
    pts = {}
    for m in ORDER[1:]:
        s, t = S("main", m), run_median_time(m)
        if s and t is not None:
            pts[m] = (t, s["lpips_in"], s["lpips_out"])
    return {m for m, p in pts.items()
            if not any(q != p and all(qi <= pi for qi, pi in zip(q, p)) for k, q in pts.items() if k != m)}


# each model's practical inference configuration (src/methods.py); used to interpret the speed table
INFER_CONFIG = {"telea": ("512 px", "– (one pass)"), "lama": ("512 px", "– (one pass)"),
                "sd15": ("512 px", "30"), "sdxl": ("1024 px", "29 (30 × strength 0.99)"),
                "hybrid": ("512 px", "27 (30 × strength 0.9) + LaMa"), "ip2p": ("512 px", "30"),
                "kontext": ("1024 px", "28")}


def table_hparams():
    p = RESULTS / "selected_hparams.json"
    if not p.exists():
        return ""
    h = json.loads(p.read_text())
    return md_table(["Hyper-parameter selected on dev", "Value", "Rule"], [
        ["Mask dilation, ground-truth masks", f"{h['oracle_dilate']} px", "lowest LaMa LPIPS-in within 0.5 pt of the best text residual"],
        ["P1 refinement strength", f"{h['hybrid_strength']}", "lowest LPIPS-in with text residual and new-text rate no worse than LaMa"],
    ])


def table_ablation():
    labels = {"default": "Full (paste-back, prompt, negative prompt)", "no_paste": "− paste-back",
              "no_negative": "− negative prompt", "empty_prompt": "− positive prompt (empty)",
              "feather4": "+ 4 px seam feathering"}
    rows = []
    for r, lab in labels.items():
        s = S("ablation", r)
        if s:
            rows.append([lab] + [fmt(s.get(k), k) for k in ("text_res", "new_text_img", "lpips_in", "lpips_out", "psnr_out", "ssim")])
    return md_table(["SD1.5-Inpaint variant (dev)"] + [HEAD[k] for k in ("text_res", "new_text_img", "lpips_in", "lpips_out",
                                                                     "psnr_out", "ssim")], rows)


def type_sets():
    anns = load_annotations()
    return {pid: {v["type"] for v in a["violations"]} for pid, a in anns.items()}


def table_per_type():
    """Text residual by type (the QR column is QR residual)."""
    rows = []
    for m in ORDER:
        s = S("main", m)
        if s:
            rows.append([NAMES[m]] + [fmt(s.get(f"tr_{t}"), "text_res") for t in TYPES[:-1]] + [fmt(s.get("qr_res"), "qr_res")])
    return md_table(["Residual (%) ↓"] + [TYPE_NAMES[t] for t in TYPES], rows)


def psnr_in_by_type():
    ts = type_sets()
    out = {}
    for m in ORDER[1:]:
        rows = per_image("main", m)
        if rows:
            out[m] = {t: float(np.mean([float(r["psnr_in"]) for i, r in rows.items() if t in ts[i]])) for t in TYPES}
    return out


def table_per_type_quality():
    q = psnr_in_by_type()
    ts = type_sets()
    ids = splits()["test"]
    n = {t: sum(t in ts[i] for i in ids) for t in TYPES}
    rows = [[NAMES[m]] + [f"{q[m][t]:.2f}" for t in TYPES] for m in q]
    out = md_table(["PSNR-in (dB) ↑"] + [f"{TYPE_NAMES[t]} (n={n[t]})" for t in TYPES], rows)
    # watermark opacity split, LaMa vs Kontext (report Sec. 4.5)
    anns = load_annotations()
    wm = {i: [v for v in anns[i]["violations"] if v["type"] == "store_watermark"][0]["alpha"]
          for i in ids if "store_watermark" in ts[i]}
    lam, kon = per_image("main", "lama"), per_image("main", "kontext")
    if lam and kon:
        for lab, sel in (("alpha < 80", [i for i, a in wm.items() if a < 80]), ("alpha >= 80", [i for i, a in wm.items() if a >= 80])):
            l = np.mean([float(lam[i]["psnr_in"]) for i in sel])
            k = np.mean([float(kon[i]["psnr_in"]) for i in sel])
            out += f"\nWatermark {lab} (n={len(sel)}): LaMa {l:.2f} dB, Kontext {k:.2f} dB ({k - l:+.1f} dB)\n"
    return out


def table_dataset():
    anns = load_annotations()
    counts = {t: 0 for t in TYPES}
    ptypes = set()
    n_two = 0
    for a in anns.values():
        ptypes.add(a.get("product_type"))
        n_two += len(a["violations"]) == 2
        for v in a["violations"]:
            counts[v["type"]] += 1
    rows = [[TYPE_NAMES[t], str(counts[t])] for t in TYPES]
    import collections
    per = collections.Counter(a.get("product_type") for a in anns.values()).values()
    extra = (f"\n{len(anns)} images from {len(ptypes)} product types ({min(per)}–{max(per)} images per type); "
             f"{n_two} images carry two violations.\n")
    return md_table(["Violation type", "Instances"], rows) + extra


# ------------------------------------------------------------------ statistics

COMPARISONS = [
    ("main/lama", "main/hybrid", "lpips_in", "P1 vs LaMa, fill quality"),
    ("main/lama", "main/hybrid", "lpips_out", "P1 vs LaMa, background"),
    ("main/sd15", "main/hybrid", "lpips_in", "P1 vs SD1.5, fill quality"),
    ("main/sd15", "main/sdxl", "lpips_in", "SDXL vs SD1.5, fill quality"),
    ("main/lama", "main/kontext", "lpips_in", "Kontext vs LaMa, fill quality"),
    ("main/lama", "main/kontext", "lpips_out", "Kontext vs LaMa, background"),
]


def paired_stats(a, b, key, n_boot=10000, seed=0):
    from scipy.stats import wilcoxon
    ra, rb = per_image(*a.split("/")), per_image(*b.split("/"))
    ids = sorted(set(ra) & set(rb))
    if not ids:
        return None
    x = np.array([float(ra[i][key]) for i in ids])
    y = np.array([float(rb[i][key]) for i in ids])
    d = y - x
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(y, x).pvalue)
    return {"n": len(ids), "a": x.mean(), "b": y.mean(), "diff": d.mean(), "lo": lo, "hi": hi, "p": p}


def table_stats():
    rows = []
    for a, b, key, label in COMPARISONS:
        r = paired_stats(a, b, key)
        if not r:
            continue
        p = "< 0.001" if r["p"] < 1e-3 else f"{r['p']:.3f}"
        rows.append([label, {"lpips_in": "LPIPS-in", "lpips_out": "LPIPS-out"}.get(key, key),
                     f"{r['a']:.4f}", f"{r['b']:.4f}", f"{r['diff']:+.4f}", f"[{r['lo']:+.4f}, {r['hi']:+.4f}]", p])
    return md_table(["Comparison (B vs A, test, paired)", "Metric", "A", "B", "Δ = B − A", "95% bootstrap CI", "Wilcoxon p"], rows)


def table_repro():
    """Same dev experiments generated on the M2 Pro (MPS) and on the A100 (CUDA)."""
    if not MAC_BACKUP.exists():
        return ""
    pairs = []
    for p in sorted(RESULTS.glob("*/*/summary.json")):
        exp, run = p.parent.parent.name, p.parent.name
        if exp == "main" or exp.startswith("_"):
            continue
        mac = S(exp, run, MAC_BACKUP)
        if mac:
            pairs.append((mac, json.loads(p.read_text())))
    if not pairs:
        return ""
    rows = []
    for k, lab in (("lpips_in", "LPIPS-in"), ("lpips_out", "LPIPS-out"), ("text_res", "Text residual")):
        d = [abs(a[k] - b[k]) for a, b in pairs]
        sc = 100 if k == "text_res" else 1
        from scipy.stats import spearmanr
        rho = spearmanr([a[k] for a, _ in pairs], [b[k] for _, b in pairs]).statistic
        unit = " pt" if k == "text_res" else ""
        f = ".2f" if k == "text_res" else ".4f"
        rows.append([lab, f"{sc * np.mean(d):{f}}{unit}", f"{sc * np.max(d):{f}}{unit}", f"{rho:.3f}"])
    return md_table([f"Metric ({len(pairs)} dev runs, M2 Pro vs A100)", "Mean abs. difference", "Max abs. difference", "Spearman ρ across runs"], rows)


def write_tables():
    TAB.mkdir(parents=True, exist_ok=True)
    for name, fn in [("main", table_main), ("ablation", table_ablation), ("per_type", table_per_type),
                     ("per_type_quality", table_per_type_quality), ("speed", table_speed), ("hparams", table_hparams),
                     ("dataset", table_dataset), ("stats", table_stats), ("repro", table_repro),
                     ("localisation", table_localisation)]:
        try:
            (TAB / f"{name}.md").write_text(fn())
            print("ok  table", name)
        except Exception as e:
            print("ERR table", name, e)
    rows = []
    for p in sorted(RESULTS.glob("*/*/summary.json")):
        s = json.loads(p.read_text())
        rows.append({"exp": p.parent.parent.name, "run": p.parent.name,
                     **{k: v for k, v in s.items() if not isinstance(v, dict)}})
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("exp", "run"), k))
        with open(RESULTS / "all_runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


# ------------------------------------------------------------------ where the editors edit (report Sec. 4.4)

EDIT_TAU = 0.1        # a pixel counts as changed when one of its channels moves by more than 0.1 (0-1 scale)


def _change(a, b):
    """Largest per-channel absolute difference, averaged over 3x3 pixels to ignore isolated noise."""
    d = np.abs(to_np(a) - to_np(b)).max(-1)
    return np.asarray(Image.fromarray((d * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(1)), np.float32) / 255


def edit_maps(m, pid, tau=EDIT_TAU):
    """Boolean maps for one test output: violation pixels (those the overlay visibly changed), pixels the method
    changed, the region a mask-based method may repaint (ground-truth mask with the selected dilation), and pixels
    that ended up closer to the clean image than to the violated input."""
    clean, violated, gt = load_sample(pid)
    out = Image.open(RESULTS / "main" / m / "images" / f"{pid}.png").convert("RGB")
    if out.size != violated.size:
        out = out.resize(violated.size, Image.LANCZOS)
    h = json.loads((RESULTS / "selected_hparams.json").read_text())
    viol = (np.asarray(gt) > 0) & (_change(violated, clean) > tau)
    changed = _change(out, violated) > tau
    allowed = np.asarray(dilate(gt, h["oracle_dilate"])) > 0
    o, v, c = to_np(out), to_np(violated), to_np(clean)
    closer = np.abs(o - c).sum(-1) < np.abs(o - v).sum(-1)
    return viol, changed, allowed, closer, violated, out


def localisation(m, tau=EDIT_TAU, types=None):
    """Pooled over all test images: share of violation pixels changed / restored, share of changed pixels
    outside the allowed region, and that outside area as a share of the image. types: only these violation types."""
    anns = load_annotations()
    t = dict(viol=0, found=0, restored=0, changed=0, outside=0, area=0)
    for pid in splits()["test"]:
        viol, changed, allowed, closer, violated, _ = edit_maps(m, pid, tau)
        if types is not None:
            W = violated.size[0]
            s = W / anns[pid]["width"]
            keep = np.zeros_like(viol)
            for v in anns[pid]["violations"]:
                if v["type"] in types:
                    x0, y0, x1, y1 = v["bbox"]
                    keep[max(0, int(y0 * s) - 2):int(y1 * s) + 3, max(0, int(x0 * s) - 2):int(x1 * s) + 3] = True
            viol = viol & keep
        t["viol"] += viol.sum()
        t["found"] += (changed & viol).sum()
        t["restored"] += (closer & viol).sum()
        t["changed"] += changed.sum()
        t["outside"] += (changed & ~allowed).sum()
        t["area"] += viol.size
    return {"found": t["found"] / t["viol"], "restored": t["restored"] / t["viol"],
            "outside_share": t["outside"] / max(t["changed"], 1), "outside_area": t["outside"] / t["area"]}


def table_localisation():
    rows = []
    for m, label in (("lama", "LaMa (given the ground-truth mask, for reference)"), ("kontext", "FLUX.1 Kontext"),
                     ("ip2p", "InstructPix2Pix")):
        if not (RESULTS / "main" / m / "images").exists():
            continue
        r = localisation(m)
        rows.append([label, f"{100 * r['found']:.1f}", f"{100 * r['restored']:.1f}", f"{100 * r['outside_share']:.1f}",
                     f"{100 * r['outside_area']:.2f}"])
    return md_table(["Method (test set)", "Violation pixels<br>changed (%) ↑", "Violation pixels<br>restored (%) ↑",
                     "Changes outside the<br>violation (% of changes) ↓", "Area changed outside<br>(% of image) ↓"], rows)


def localisation_details():
    """Numbers quoted in the text of Sec. 4.4: other thresholds and the split by violation type."""
    for m in ("kontext", "ip2p"):
        for tau in (0.05, 0.2):
            r = localisation(m, tau)
            print(f"{m} tau={tau}: changed {100 * r['found']:.1f}%, outside {100 * r['outside_share']:.1f}% of changes, "
                  f"{100 * r['outside_area']:.2f}% of image")
        for ty in TYPES:
            r = localisation(m, types={ty})
            print(f"{m} {ty}: changed {100 * r['found']:.1f}%, restored {100 * r['restored']:.1f}%")


# ------------------------------------------------------------------ figures

SWEEP_METRICS = [("text_res", "Text residual (%) ↓", 100), ("new_text_img", "Images with new text (%) ↓", 100),
                 ("lpips_in", "LPIPS inside mask ↓", 1), ("lpips_out", "LPIPS outside mask ↓", 1)]


def sweep_series(exp, prefix, values):
    xs, ss = [], []
    for v in values:
        s = S(exp, f"{prefix}{v}")
        if s:
            xs.append(v)
            ss.append(s)
    return xs, ss


def plot_sweep_row(axes, series, xlabel):
    """series: [(method_key, xs, summaries)]; one metric per panel, never two y-axes."""
    for ax, (k, lab, scale) in zip(axes, SWEEP_METRICS):
        for m, xs, ss in series:
            if not xs:
                continue
            ys = [s[k] * scale for s in ss]
            ax.plot(xs, ys, "-o", color=COLORS[m], label=SHORT[m])
            if len(series) > 1:
                ax.annotate(SHORT[m], (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                            color=INK2, fontsize=7, va="center")
        ax.set_title(lab, fontsize=8.5)
        top = max((s[k] * scale for _, _, ss in series for s in ss), default=0)
        ax.set_ylim(0, max(top * 1.15, 1.0 if scale == 100 else 0.005))   # zero baseline: no exaggerated slopes
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_xticks(series[0][1])
        ax.tick_params(labelsize=7.5)
    if len(series) > 1:
        axes[0].legend(frameon=False, fontsize=7)


SWEEPS_A = [
    ("SD1.5 guidance", lambda: [("sd15", *sweep_series("sweep_guidance", "g", [1, 3, 5, 7.5, 12]))], "guidance scale"),
    ("SD1.5 steps", lambda: [("sd15", *sweep_series("sweep_steps", "s", [10, 20, 30, 50]))], "sampling steps"),
    ("Dilation (GT mask)", lambda: [("sd15", *sweep_series("sweep_dilation", "sd15_d", [0, 4, 8, 16, 32])),
                                   ("lama", *sweep_series("sweep_dilation", "lama_d", [0, 4, 8, 16, 32]))], "dilation (px)"),
]
SWEEPS_B = [
    ("P1 strength", lambda: [("hybrid", *sweep_series("sweep_strength", "hy", [0.3, 0.5, 0.7, 0.9]))], "refinement strength s"),
    ("IP2P image guid.", lambda: [("ip2p", *sweep_series("sweep_ip2p_img", "ig", [1.0, 1.25, 1.5, 2.0]))], "image guidance s_I"),
    ("IP2P text guid.", lambda: [("ip2p", *sweep_series("sweep_ip2p_txt", "g", [3, 7.5, 12]))], "text guidance s_T"),
    ("Kontext guidance", lambda: [("kontext", *sweep_series("sweep_kontext", "g", [1.5, 2.5, 4.0]))], "guidance"),
]


def fig_sweeps():
    for name, specs in (("sweeps_a", SWEEPS_A), ("sweeps_b", SWEEPS_B)):
        specs = [(t, f(), xl) for t, f, xl in specs]
        specs = [s for s in specs if any(xs for _, xs, _ in s[1])]
        if not specs:
            continue
        fig, axes = plt.subplots(len(specs), 4, figsize=(10, 2.05 * len(specs)), squeeze=False)
        for row, (title, series, xl) in zip(axes, specs):
            plot_sweep_row(row, series, xl)
            row[0].text(-0.45, 0.5, title, transform=row[0].transAxes, rotation=90, va="center", ha="center",
                        fontsize=9, color=INK, fontweight="bold")
        fig.tight_layout(h_pad=1.0)
        fig.savefig(FIG / f"{name}.png")
        plt.close(fig)


def fig_tradeoff():
    """Fill quality vs background damage (log-log): mask-based methods with the ground-truth mask, editors without."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for m in ORDER[1:]:
        s = S("main", m)
        if not s:
            continue
        x, y = s["lpips_in"], s["lpips_out"]
        ax.scatter(x, y, s=70, color=COLORS[m], edgecolor="white", linewidth=2, zorder=4)
        off = {"sdxl": (-34, 4), "telea": (6, -12), "hybrid": (6, 5), "lama": (6, -12)}.get(m, (6, 5))
        ax.annotate(SHORT[m] + (" (no mask)" if m in MASK_FREE else ""), (x, y), xytext=off,
                    textcoords="offset points", fontsize=8, color=INK, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("LPIPS inside mask ↓  (fill quality)")
    ax.set_ylabel("LPIPS outside mask ↓  (background damage)")
    fig.tight_layout()
    fig.savefig(FIG / "tradeoff.png")
    plt.close(fig)


def fig_per_type():
    """PSNR inside the mask by violation type (images containing that type)."""
    q = psnr_in_by_type()
    ms = [m for m in ["telea", "lama", "sd15", "sdxl", "hybrid", "kontext"] if m in q]
    if not ms:
        return
    fig, ax = plt.subplots(figsize=(10, 2.9))
    x = np.arange(len(TYPES))
    w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        ax.bar(x - 0.4 + w * (i + 0.5), [q[m][t] for t in TYPES], w * 0.88, color=COLORS[m], label=SHORT[m])
    ax.set_xticks(x, [TYPE_NAMES[t] for t in TYPES])
    ax.set_ylabel("PSNR inside mask (dB) ↑")
    ax.set_ylim(10, None)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=7.5, ncol=len(ms), loc="upper center", bbox_to_anchor=(0.5, 1.17))
    fig.tight_layout()
    fig.savefig(FIG / "per_type.png")
    plt.close(fig)


def _thumb(img, w):
    return img.resize((w, round(img.height * w / img.width)), Image.LANCZOS)


def _grid(rows, col_titles, path, w=170, row_labels=None, font_px=15):
    """rows: list of lists of PIL images (same length as col_titles); optional row labels on the left."""
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_px)
    except OSError:
        font = ImageFont.load_default()
    gap, head = 4, font_px + 9
    left = 0
    if row_labels:
        left = 44 if font_px == 15 else round(max(font.getlength(t) for t in row_labels)) + 2 * font_px
    heights = [max(_thumb(i, w).height for i in r) for r in rows]
    W = left + len(col_titles) * (w + gap) - gap
    H = head + sum(heights) + gap * len(rows)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for j, t in enumerate(col_titles):
        tw = d.textlength(t, font=font)
        d.text((left + j * (w + gap) + (w - tw) / 2, 3), t, fill=INK, font=font)
    y = head
    for r_i, (r, h) in enumerate(zip(rows, heights)):
        if row_labels:
            d.text((6, y + h / 2 - 0.6 * font_px), row_labels[r_i], fill=INK, font=font)
        for j, im in enumerate(r):
            t = _thumb(im, w)
            sheet.paste(t, (left + j * (w + gap), y + (h - t.height) // 2))
        y += h + gap
    sheet.save(path, quality=92)


def pick_examples(n=6):
    """One test image per violation type where possible, single-violation first."""
    anns = load_annotations()
    test = splits()["test"]
    chosen = []
    for t in TYPES:
        for pid in test:
            vs = [v["type"] for v in anns[pid]["violations"]]
            if t in vs and pid not in chosen and len(vs) == 1:
                chosen.append(pid)
                break
    return chosen[:n]


def fig_qualitative():
    ms = [m for m in ["telea", "lama", "sd15", "sdxl", "hybrid", "ip2p", "kontext"] if (RESULTS / "main" / m / "images").exists()]
    if not ms:
        return
    rows = []
    for pid in pick_examples():
        clean, violated, _ = load_sample(pid)
        ims = [violated]
        for m in ms:
            p = RESULTS / "main" / m / "images" / f"{pid}.png"
            ims.append(Image.open(p).convert("RGB") if p.exists() else Image.new("RGB", clean.size, "white"))
        rows.append(ims + [clean])
    _grid(rows, ["Input"] + [SHORT[m] for m in ms] + ["Ground truth"], FIG / "qualitative.jpg", w=150)


def fig_localisation(n=3):
    """Where the editors change the image: green = violation pixel changed, orange = violation pixel left as it
    was, blue = pixel changed outside the region a mask-based method may repaint."""
    stats = []
    for pid in splits()["test"]:
        viol, changed, allowed, _, _, _ = edit_maps("kontext", pid)
        if viol.sum():
            stats.append(((changed & ~allowed).mean(), (changed & viol).sum() / viol.sum(), pid))
    stats.sort()
    picks = [stats[len(stats) // 2][2], stats[-1][2]]                  # median and largest area changed outside
    picks.append(min((x for x in stats if x[2] not in picks), key=lambda x: x[1])[2])   # fewest violation pixels changed
    rows = []
    for pid in picks[:n]:
        row = []
        for m in ("kontext", "ip2p"):
            viol, changed, allowed, _, violated, out = edit_maps(m, pid)
            base = np.asarray(violated.convert("L").convert("RGB")).astype(np.float32) * 0.45 + 140
            base[changed & viol] = (27, 175, 122)
            base[viol & ~changed] = (235, 104, 52)
            base[changed & ~allowed] = (42, 120, 214)
            if not row:
                row.append(violated)
            row += [out, Image.fromarray(base.clip(0, 255).astype(np.uint8))]
        rows.append(row)
    _grid(rows, ["Input", "Kontext", "Kontext: changes", "IP2P", "IP2P: changes"], FIG / "localisation.jpg", w=150)


def fig_dataset():
    """One test image per violation type: one row per type; columns clean / violated / ground-truth mask."""
    rows, labels = [], []
    anns = load_annotations()
    for pid in pick_examples():
        clean, violated, mask = load_sample(pid)
        overlay = Image.composite(Image.new("RGB", clean.size, (235, 104, 52)), clean.point(lambda v: v // 2 + 100), mask)
        rows.append([clean, violated, overlay])
        labels.append(TYPE_NAMES[anns[pid]["violations"][0]["type"]])
    _grid(rows, ["Clean original", "Violated input", "Ground-truth mask (orange)"], FIG / "dataset.jpg", w=400,
          row_labels=labels, font_px=30)


FAILURES = [("sdxl", "text_res", "SDXL re-draws the badge with new text"),
            ("sd15", "lpips_in", "SD1.5 invents a round object in the round hole"),
            ("ip2p", "text_res", "InstructPix2Pix keeps the text and recolours the product's own logo"),
            ("kontext", "lpips_out", "Kontext re-renders the product slightly displaced"),
            ("lama", "lpips_in", "LaMa's worst case: a large badge hole filled with a blurred patch")]


def fig_failures():
    """Worst test case of each failure mode, by the metric the mode shows up in."""
    cases = []
    for m, k, title in FAILURES:
        rows = per_image("main", m)
        cand = sorted((r for r in rows.values() if not math.isnan(float(r[k]))), key=lambda r: -float(r[k]))
        for r in cand:
            if r["id"] not in [c[1] for c in cases]:
                cases.append((m, r["id"], title))
                break
    if not cases:
        return
    cols = []
    for m, pid, _ in cases:
        clean, violated, _ = load_sample(pid)
        cols.append([violated, Image.open(RESULTS / "main" / m / "images" / f"{pid}.png").convert("RGB"), clean])
    # one column per case, rows = input / output / ground truth: wide and short, fits one page
    _grid([list(r) for r in zip(*cols)], [f"({chr(97 + i)}) {SHORT[m]}" for i, (m, _, _) in enumerate(cases)],
          FIG / "failures.jpg", w=160, row_labels=["In", "Out", "GT"])
    (FIG / "failures.json").write_text(json.dumps([{"method": m, "id": pid, "mode": t} for m, pid, t in cases], indent=2))


KONTEXT_MODES = [("sweep_kontext/g2.5", "71kIETMF8jL"),   # dev: words removed, badge disc kept
                 ("sweep_kontext/g2.5", "71Mf2wNTgsL"),   # dev: white background turned grey
                 ("main/kontext", "71bLB7hpmAL")]         # test: product re-rendered slightly displaced


def fig_kontext_modes():
    """Kontext's three failure modes with |output - ground truth| x4 to show where pixels changed."""
    from PIL import ImageChops
    rows = []
    for run, pid in KONTEXT_MODES:
        p = RESULTS / run / "images" / f"{pid}.png"
        if not p.exists():
            return
        clean, violated, _ = load_sample(pid)
        out = Image.open(p).convert("RGB")
        rows.append([violated, out, clean, ImageChops.difference(out, clean).point(lambda v: min(255, 4 * v))])
    _grid(rows, ["Input", "Kontext", "Ground truth", "abs. difference x4"], FIG / "kontext_modes.jpg", w=170,
          row_labels=["(a)", "(b)", "(c)"])


def fig_metric_explainer(pid="61GhLWzgVUL", run="main/sd15"):
    """How one output is scored: two composites isolate the erased region and the background."""
    p = RESULTS / run / "images" / f"{pid}.png"
    if not p.exists():
        return
    clean, violated, mask = load_sample(pid)
    out = Image.open(p).convert("RGB")
    fill = Image.composite(out, clean, mask)          # output inside the mask, original outside
    back = Image.composite(clean, out, mask)          # original inside the mask, output outside
    ov = Image.composite(Image.new("RGB", clean.size, (235, 104, 52)), violated.point(lambda v: v // 2 + 100), mask)
    _grid([[violated, ov, out, fill, back, clean]],
          ["Input", "Mask (orange)", "Output", "Scored for fill", "Scored for background", "Ground truth"],
          FIG / "metric_explainer.jpg", w=150)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    write_tables()
    for fn in (fig_dataset, fig_sweeps, fig_tradeoff, fig_per_type, fig_qualitative, fig_localisation,
               fig_failures, fig_kontext_modes, fig_metric_explainer):
        try:
            fn()
            print("ok  fig  ", fn.__name__)
        except Exception as e:
            print("ERR fig  ", fn.__name__, e)


if __name__ == "__main__":
    main()
