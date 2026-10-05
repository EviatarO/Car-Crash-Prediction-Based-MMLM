"""Builds the 2026-10 reasoning-path plan deck (title + 5 slides, dark theme).

Style, palette and helpers copied from build_status_presentation_2026-08-22.py (the August
semantic-guidance deck). Slides stay sparse; the speaker notes carry the detail, one bullet
per line.

Every AP and per-TTE count is recomputed from the per-clip score files and asserted against
the numbers printed on the slides, so a stale figure can never be silently embedded.

    python student_training/scripts/build_reasoning_path_deck_2026-10.py
"""
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls, qn
from pptx.util import Inches, Pt
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "outputs" / "a1_compress256" / "scores"
OUT = ROOT / "reports" / "presentations" / "2026-10_reasoning-path-plan.pptx"

# ---- palette (identical to the August deck)
BG = RGBColor(0x1C, 0x23, 0x40)
PANEL = RGBColor(0x24, 0x30, 0x60)
PANEL_DK = RGBColor(0x20, 0x2A, 0x50)
CYAN = RGBColor(0x00, 0xBF, 0xFF)
CYAN_DK = RGBColor(0x00, 0x96, 0xCC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MUTED = RGBColor(0xA0, 0xB4, 0xCC)
GREEN = RGBColor(0x00, 0xE6, 0x76)
ORANGE = RGBColor(0xFF, 0xA7, 0x26)
RED = RGBColor(0xFF, 0x6B, 0x6B)
FROZEN_LINE = RGBColor(0x6E, 0x82, 0xA8)
SW, SH = 10.0, 5.625
FOOTER = "CCP-MMLM · Reasoning Path for Collision Anticipation"

# ---------------------------------------------------------------- verified data
# Correct at threshold 0.5: positives = TP per TTE bucket (group 0/1/2 = 0.5/1.0/1.5 s),
# negatives = TN. BADAS-Open = the published checkpoint with its published (crop) preprocessing.
EXPECTED = {
    "private": {"n": 677,
                "A0": {"ap": 0.8535, "g0": (135, 142), "g1": (104, 117), "g2": (69, 79), "neg": (209, 339)},
                "A1": {"ap": 0.9128, "g0": (129, 142), "g1": (101, 117), "g2": (56, 79), "neg": (284, 339)}},
    "public": {"n": 667,
               "A0": {"ap": 0.8711, "g0": (127, 142), "g1": (105, 115), "g2": (71, 77), "neg": (196, 333)},
               "A1": {"ap": 0.9096, "g0": (126, 142), "g1": (105, 115), "g2": (52, 77), "neg": (262, 333)}},
}


def _load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _y(r):
    v = r.get("gt_verdict", r.get("ground_truth"))
    return 1 if v in ("YES", 1, "1", True) else 0


def _summ(rows):
    y = [_y(r) for r in rows]
    s = [float(r["score"]) for r in rows]
    out = {"ap": average_precision_score(y, s), "n": len(rows)}
    for g in ("g0", "g1", "g2", "neg"):
        sel = [(yy, ss) for r, yy, ss in zip(rows, y, s)
               if (yy == 0 if g == "neg" else (yy == 1 and f"g{r['group']}" == g))]
        ok = sum(int((ss >= 0.5) == bool(yy)) for yy, ss in sel)
        out[g] = (ok, len(sel))
    return out


def verify():
    pub_a1 = _load(SCORES / "public_compress256_a1" / "A1-compress256.jsonl")
    pub_ids = {r["video_id"] for r in pub_a1}
    pooled = _load(SCORES / "A1_compress256_pooled.jsonl")
    priv_a1 = [r for r in pooled if r["video_id"] not in pub_ids]
    got = {
        "private": {"A0": _summ(_load(SCORES / "private_crop" / "A0.jsonl")), "A1": _summ(priv_a1)},
        "public": {"A0": _summ(_load(SCORES / "public_crop" / "A0.jsonl")), "A1": _summ(pub_a1)},
    }
    for split, exp in EXPECTED.items():
        for arm in ("A0", "A1"):
            g, e = got[split][arm], exp[arm]
            if g["n"] != exp["n"]:
                raise SystemExit(f"N MISMATCH {split}/{arm}: {g['n']} vs {exp['n']}")
            if abs(g["ap"] - e["ap"]) > 0.0006:
                raise SystemExit(f"AP MISMATCH {split}/{arm}: {g['ap']:.4f} vs {e['ap']}")
            for k in ("g0", "g1", "g2", "neg"):
                if g[k] != e[k]:
                    raise SystemExit(f"COUNT MISMATCH {split}/{arm}/{k}: {g[k]} vs {e[k]}")
    print("  [verify] AP + per-TTE correct counts reproduce from the per-clip score files")


# ---------------------------------------------------------------- primitives
def _bg(slide):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG


def rect(slide, x, y, w, h, color, shape=MSO_SHAPE.RECTANGLE):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def outline(slide, x, y, w, h, color, dash=False, width=1.0, fill=None, shape=MSO_SHAPE.RECTANGLE):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = color
    sh.line.width = Pt(width)
    if dash:
        sh.line.dash_style = MSO_LINE.DASH
    sh.shadow.inherit = False
    return sh


def text(slide, x, y, w, h, runs, size=11, color=WHITE, bold=False, align=PP_ALIGN.LEFT,
         font="Calibri", italic=False, space_after=4, line_spacing=None, anchor=None):
    """runs: str, or list of paragraphs; each paragraph is a str or a list of (txt, dict)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.03)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    if anchor:
        tf.vertical_anchor = anchor
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        if line_spacing:
            p.line_spacing = line_spacing
        pieces = [(para, {})] if isinstance(para, str) else para
        for txt, ov in pieces:
            r = p.add_run(); r.text = txt
            r.font.size = Pt(ov.get("size", size))
            r.font.bold = ov.get("bold", bold)
            r.font.italic = ov.get("italic", italic)
            r.font.color.rgb = ov.get("color", color)
            r.font.name = ov.get("font", font)
    return tb


def notes(slide, lines):
    """One bullet per line. Never wraps - read live while presenting."""
    tf = slide.notes_slide.notes_text_frame
    tf.text = lines[0]
    for l in lines[1:]:
        tf.add_paragraph().text = l


def content_slide(prs, title, subline=None, page=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(s)
    rect(s, 0.35, 0.22, 0.07, 0.50, CYAN)
    t_size = 25 if len(title) <= 52 else max(19, 25 * 52 / len(title))
    text(s, 0.50, 0.13, 9.20, 0.60, title, size=t_size, color=CYAN, bold=True)
    rect(s, 0.35, 0.84, 9.30, 0.012, CYAN_DK)
    if subline:
        text(s, 0.50, 0.88, 9.20, 0.40, subline, size=11, color=MUTED)
    text(s, 0.50, 5.29, 7.00, 0.30, FOOTER, size=8.5, color=MUTED)
    if page:
        text(s, 8.80, 5.29, 0.80, 0.30, str(page), size=8.5, color=MUTED, align=PP_ALIGN.RIGHT)
    return s


def card(slide, x, y, w, h, title, bullets, accent=CYAN, sub=None, body_size=9.5, fill=PANEL):
    rect(slide, x, y, w, h, fill)
    rect(slide, x, y, w, 0.05, accent)
    text(slide, x + 0.13, y + 0.10, w - 0.26, 0.30, title, size=12, color=accent, bold=True)
    top = y + 0.42
    if sub:
        text(slide, x + 0.13, y + 0.38, w - 0.26, 0.26, sub, size=9, color=MUTED)
        top = y + 0.68
    text(slide, x + 0.13, top, w - 0.26, h - (top - y) - 0.08,
         [f"• {b}" for b in bullets], size=body_size, color=WHITE, space_after=3)


def hero(slide, x, y, w, h, headline, tail, accent=GREEN, head_size=20, tail_size=12):
    rect(slide, x, y, w, h, PANEL_DK)
    rect(slide, x, y, 0.07, h, accent)
    text(slide, x + 0.20, y + 0.06, w - 0.30, h - 0.10,
         [[(headline, {"size": head_size, "bold": True, "color": accent}),
           (tail, {"size": tail_size, "color": WHITE})]], size=12, anchor=MSO_ANCHOR.MIDDLE)


def table(slide, rows, cols_w, x, y, h, font=8.5, header_font=8.5, align_first_left=True,
          bold_first_col=False, wrap_left_cols=()):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                                   Inches(sum(cols_w)), Inches(h))
    tbl = shape.table
    tbl.first_row = False
    for j, w in enumerate(cols_w):
        tbl.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            fill = "202A50" if i == 0 else ("243060" if i % 2 else "1F2950")
            cell._tc.get_or_add_tcPr().append(
                parse_xml(f'<a:solidFill {nsdecls("a")}><a:srgbClr val="{fill}"/></a:solidFill>'))
            cell.text = str(val)
            cell.margin_left = Inches(0.06); cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.025); cell.margin_bottom = Inches(0.025)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                left = (j == 0 and align_first_left) or j in wrap_left_cols
                p.alignment = PP_ALIGN.LEFT if left else PP_ALIGN.CENTER
                for r in p.runs:
                    r.font.size = Pt(header_font if i == 0 else font)
                    r.font.name = "Calibri"
                    r.font.bold = (i == 0) or (bold_first_col and j == 0)
                    r.font.color.rgb = CYAN if i == 0 else WHITE
    return tbl


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=1.5, dash=False, kind=MSO_CONNECTOR.STRAIGHT):
    c = slide.shapes.add_connector(kind, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = MSO_LINE.ROUND_DOT
    ln = c.line._get_or_add_ln()
    tail = ln.find(qn("a:tailEnd"))
    if tail is None:
        tail = parse_xml(f'<a:tailEnd {nsdecls("a")} type="triangle" w="med" len="med"/>')
        ln.append(tail)
    return c


def block(slide, x, y, w, h, title, size_txt, kind="frozen", title_size=9.5, size_size=8):
    """Diagram block. kind: frozen | trained | lora | io."""
    if kind == "trained":
        sh = outline(slide, x, y, w, h, ORANGE, width=2.0, fill=PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        tag, tag_c = "TRAINED", ORANGE
    elif kind == "lora":
        sh = outline(slide, x, y, w, h, ORANGE, dash=True, width=1.75, fill=PANEL,
                     shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        tag, tag_c = "S1 frozen · S2 LoRA", ORANGE
    elif kind == "io":
        sh = outline(slide, x, y, w, h, CYAN_DK, width=1.0, fill=PANEL_DK, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        tag, tag_c = None, None
    else:
        sh = outline(slide, x, y, w, h, FROZEN_LINE, width=1.0, fill=PANEL_DK,
                     shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        tag, tag_c = "FROZEN", FROZEN_LINE
    sh.adjustments[0] = 0.12
    paras = [[(title, {"size": title_size, "bold": True, "color": WHITE})],
             [(size_txt, {"size": size_size, "color": MUTED})]]
    if tag:
        paras.append([(tag, {"size": 7, "bold": True, "color": tag_c})])
    text(slide, x + 0.04, y + 0.02, w - 0.08, h - 0.04, paras, align=PP_ALIGN.CENTER,
         space_after=1, anchor=MSO_ANCHOR.MIDDLE)


# ================================================================= SLIDES
def s00_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    rect(s, 0, 0, SW, 0.08, CYAN)
    rect(s, 0, SH - 0.08, SW, 0.08, CYAN)
    rect(s, 0, 0, 0.08, SH, CYAN_DK)
    rect(s, 7.40, 0, 2.60, 2.60, PANEL_DK)
    text(s, 0.50, 0.80, 9.00, 1.10, "CCP-MMLM", size=48, color=WHITE, bold=True)
    rect(s, 0.50, 1.90, 6.50, 0.06, CYAN)
    text(s, 0.50, 1.99, 8.60, 0.90,
         "Reasoning Path for Collision Anticipation\nExplaining V-JEPA2 alerts from the "
         "encoder's own features", size=15, color=MUTED)
    rect(s, 0.50, 3.05, 9.00, 1.42, PANEL)
    rect(s, 0.50, 3.05, 9.00, 0.05, CYAN)
    text(s, 0.62, 3.13, 8.70, 0.30, "Research goal", size=12, color=CYAN, bold=True)
    text(s, 0.62, 3.45, 8.76, 1.00,
         "Keep the collision predictor exactly as it is (AP ~0.91), and add a language path "
         "that explains each alert from the same video features that produced the score - "
         "one vision pass, and an explanation whose link to the decision can be measured.",
         size=11, color=WHITE)
    text(s, 0.50, 4.62, 7.00, 0.35, "MSc Thesis - Plan Update  |  October 2026", size=11.5, color=CYAN)
    text(s, 0.50, 4.92, 5.00, 0.42, "Eviatar Ohayon", size=15, color=WHITE, bold=True)
    notes(s, [
        "Opening framing:",
        "- The crash predictor is finished and frozen: AP 0.913 private / 0.910 public.",
        "- Next goal: explain each alert in text, generated from the predictor's own features.",
        "- Today: where we stand, why the idea is new, why the June attempt failed, the design, the timeline.",
        "- The ask: approve a one-week go/no-go before we invest the full three weeks."])


GROUPS = (("TTE 0.5s", "g0"), ("TTE 1s", "g1"), ("TTE 1.5s", "g2"), ("Negatives", "neg"))


def _cm_table(s, split, label, y):
    """Correct-at-0.5 table for one split: BADAS-Open, A1-compress256, and the change."""
    d = EXPECTED[split]
    fmt = lambda c: f"{c[0]}/{c[1]}  ({round(100 * c[0] / c[1])}%)"
    rows = [[label] + [f"{t}  ·  {d['A0'][g][1]} clips" for t, g in GROUPS] + ["AP"],
            ["BADAS-Open"] + [fmt(d["A0"][g]) for _, g in GROUPS] + ["0.853" if split == "private" else "0.871"],
            ["A1-compress256 (ours)"] + [fmt(d["A1"][g]) for _, g in GROUPS] + [f"{d['A1']['ap']:.3f}"],
            ["Change (correct clips)"] + [f"{d['A1'][g][0] - d['A0'][g][0]:+d}" for _, g in GROUPS]
            + [f"{d['A1']['ap'] - (0.853 if split == 'private' else 0.871):+.3f}"]]
    tbl = table(s, rows, [1.75, 1.55, 1.55, 1.55, 1.55, 1.15], 0.45, y, 1.18, font=9.5, header_font=9,
                bold_first_col=True)
    for j in range(1, len(rows[0])):  # colour the change row: green = better, red = worse
        val = rows[3][j]
        col = GREEN if val.startswith("+") and val.strip("+0.") else (RED if val.startswith("-") else MUTED)
        for para in tbl.cell(3, j).text_frame.paragraphs:
            for r in para.runs:
                r.font.color.rgb = col
                r.font.bold = True
    for j in range(len(rows[0])):  # highlight our row
        for para in tbl.cell(2, j).text_frame.paragraphs:
            for r in para.runs:
                r.font.color.rgb = ORANGE
                r.font.bold = True


def s01_status(prs):
    s = content_slide(prs, "Where We Stand - Crash Path Frozen at AP ~0.91",
                      "Fine-tuned BADAS-Open on full-frame input: far fewer false alarms, small recall loss at 1.5 s.", 1)
    hero(s, 0.45, 1.27, 9.10, 0.55, "AP 0.913 / 0.910",
         "     Private / Public     -     BADAS-Open (published): 0.853 / 0.871",
         head_size=19, tail_size=11.5)  # 0.8535 is reported as the historical 0.853
    _cm_table(s, "private", "Private · 677", 2.02)
    _cm_table(s, "public", "Public · 667", 3.42)
    text(s, 0.45, 4.72, 9.10, 0.45,
         "Correct at threshold 0.5. TTE columns = collisions alarmed (true positives) per "
         "time-to-event; Negatives = correctly silent (true negatives). AP is threshold-free.",
         size=8.5, color=MUTED, italic=True)
    notes(s, [
        "What the numbers say:",
        "- AP rose from 0.853 to 0.913 on private and 0.871 to 0.910 on public - AP is threshold-free.",
        "- The tables are at threshold 0.5: how many clips each model gets right, per time-to-event bucket.",
        "- Most of the gain is on negatives: +75 private / +66 public correctly silent - about a third fewer false alarms.",
        "- Early horizon costs a little: at 1.5 s, -13 private / -19 public - the known 1.5 s ceiling, accepted.",
        "- Model = A1-compress256: BADAS-Open + LoRA, full frame squashed to 256x256 instead of a centre crop.",
        "- This path is now frozen. Nothing in the reasoning work changes the score."])


def s02_novelty(prs):
    s = content_slide(prs, "The Goal - Explain From the Features That Decided",
                      "Most systems explain with a second set of eyes. We explain with the predictor's own.", 2)
    cards = [
        ("Regular MLLM", "InternVL / Qwen3-VL", MUTED,
         ["pixels -> its OWN image encoder -> projector -> LLM",
          "Encoder trained with text, together with the LLM",
          "Explanation = a second opinion, not the predictor's reason"]),
        ("BADAS-Reason", "Nexar, BADAS-2.0", CYAN,
         ["BADAS picks the peak frame + object boxes",
          "Qwen3-VL re-reads ONE cropped frame with its own encoder",
          "Two vision passes; motion seen only through one frame"]),
        ("Ours", "reasoning path", GREEN,
         ["The SAME V-JEPA2 tokens that produced the score -> projector -> LLM",
          "One vision pass, 2 s of motion",
          "Faithfulness measurable: explanation and alert share evidence"]),
    ]
    for i, (t, sub, acc, b) in enumerate(cards):
        card(s, 0.45 + i * 3.08, 1.35, 2.92, 2.15, t, b, accent=acc, sub=sub, body_size=10,
             fill=PANEL if i < 2 else PANEL_DK)
    hero(s, 0.45, 3.78, 9.10, 0.95, "Open problem, named by Nexar.",
         "  BADAS-2.0 lists latent-based reasoning - language read directly from the "
         "predictor's features - as future work (p.8). No published system does it for collisions.",
         accent=GREEN, head_size=14, tail_size=11)
    notes(s, [
        "Why this is different - the key distinction:",
        "- A regular MLLM has its own vision encoder, trained WITH text, so its features are already word-like.",
        "- Its explanation comes from that encoder - not from whatever model raised the alarm.",
        "- BADAS-Reason (Nexar) is the same pattern: BADAS only chooses the frame; Qwen3-VL looks again.",
        "- Ours: the language model reads the predictor's own V-JEPA2 tokens. No second look.",
        "- Consequence 1: one vision pass - the explanation costs no extra encoding.",
        "- Consequence 2: faithfulness can be tested - hide the named object, check the score drops.",
        "- The hard part: V-JEPA2 never saw text, so the projector must learn the translation (next slides)."])


def s03_architecture(prs):
    s = content_slide(prs, "Architecture - One Encoder, Two Parallel Paths",
                      "The prediction path is untouched. The reasoning path reads the same tokens.", 3)
    # shared front (left column)
    lx, lw = 0.40, 1.75
    block(s, lx, 1.30, lw, 0.58, "Dashcam window", "16 frames · 1280×720", kind="io")
    block(s, lx, 2.10, lw, 0.58, "Preprocess", "compress256 -> 16×3×256×256", kind="io")
    block(s, lx, 2.90, lw, 0.80, "V-JEPA2 ViT-L", "+ LoRA (A1) · 24 layers · 304M", kind="frozen")
    block(s, lx, 3.92, lw, 0.62, "Patch tokens", "2,048 × 1024  (8×16×16)", kind="io")
    for y1, y2 in ((1.88, 2.10), (2.68, 2.90), (3.70, 3.92)):
        arrow(s, lx + lw / 2, y1, lx + lw / 2, y2)
    # rows
    bx0, bw, gap = 2.65, 1.55, 0.22
    xs = [bx0 + k * (bw + gap) for k in range(4)]
    up_y, up_h = 1.48, 0.80
    lo_y, lo_h = 3.42, 0.95
    text(s, bx0, 1.17, 6.9, 0.26, "PREDICTION PATH  ·  unchanged, frozen", size=9.5, color=CYAN, bold=True)
    text(s, bx0, 4.42, 6.9, 0.26, "REASONING PATH  ·  new", size=9.5, color=GREEN, bold=True)
    up = [("+ predicted tokens", "2,560 × 1024", "frozen"),
          ("Attentive probe", "2,560 -> 1 query", "frozen"),
          ("Classifier", "1024 -> 2", "frozen"),
          ("P(collision)", "alert score", "io")]
    lo = [("2×2 Merger", "Qwen3-VL structure\n4×1024 -> 2560", "trained"),
          ("Visual tokens", "512 × 2560\n+ 3D position (t,h,w)", "io"),
          ("Qwen3-VL-4B LM", "language model only\n(its ViT dropped)", "lora"),
          ("Explanation", "scene · agents · risk", "io")]
    for k in range(4):
        block(s, xs[k], up_y, bw, up_h, up[k][0], up[k][1], kind=up[k][2])
        block(s, xs[k], lo_y, bw, lo_h, lo[k][0], lo[k][1], kind=lo[k][2], size_size=7.5)
        if k < 3:
            arrow(s, xs[k] + bw, up_y + up_h / 2, xs[k + 1], up_y + up_h / 2)
            arrow(s, xs[k] + bw, lo_y + lo_h / 2, xs[k + 1], lo_y + lo_h / 2)
    # fork from tokens to both rows
    fx = lx + lw
    arrow(s, fx, 4.10, xs[0], up_y + up_h / 2, color=CYAN, kind=MSO_CONNECTOR.ELBOW)
    arrow(s, fx, 4.30, xs[0], lo_y + lo_h / 2, color=GREEN, kind=MSO_CONNECTOR.ELBOW)
    # consistency check
    cx = xs[3] + bw / 2
    arrow(s, cx, up_y + up_h, cx, lo_y, color=GREEN, dash=True, width=1.25)
    text(s, cx + 0.05, 2.55, 0.95, 0.6, "verdict must\nagree with\nscore", size=7.5, color=GREEN, italic=True)
    # legend
    ly = 4.80
    outline(s, 2.65, ly, 0.28, 0.17, FROZEN_LINE, fill=PANEL_DK)
    text(s, 2.98, ly - 0.04, 0.9, 0.25, "frozen", size=8, color=MUTED)
    outline(s, 3.80, ly, 0.28, 0.17, ORANGE, width=2.0, fill=PANEL)
    text(s, 4.13, ly - 0.04, 1.3, 0.25, "trained from scratch", size=8, color=MUTED)
    outline(s, 5.55, ly, 0.28, 0.17, ORANGE, dash=True, width=1.75, fill=PANEL)
    text(s, 5.88, ly - 0.04, 2.3, 0.25, "LoRA in stage 2 (fine-tune) only", size=8, color=MUTED)
    outline(s, 8.10, ly, 0.28, 0.17, CYAN_DK, fill=PANEL_DK)
    text(s, 8.43, ly - 0.04, 1.2, 0.25, "data / tensor", size=8, color=MUTED)
    notes(s, [
        "Walk it left to right:",
        "- One window: 16 frames, squashed to 256x256 (compress256), into the frozen V-JEPA2 ViT-L + our LoRA.",
        "- Output: 2,048 tokens = 8 time steps x 16 x 16 positions, each 1,024 numbers.",
        "- Upper path = today's predictor, untouched: probe + classifier -> collision score.",
        "- Lower path = new. Merger: groups each 2x2 block of tokens -> 512 tokens of 2,560 (the LLM's input size).",
        "- The merger copies Qwen3-VL's own projector design; only its weights are new (trained from scratch).",
        "- Each token keeps its (time, row, column) position, given to the LLM the way Qwen3-VL expects it.",
        "- LLM = Qwen3-VL-4B's language model: trained to read video tokens. Its own vision encoder is discarded.",
        "- Stage 1 (coarse): only the merger trains. Stage 2 (fine-tune): + LoRA on the language model.",
        "- Dotted arrow: the explanation's verdict is checked against the score - the faithfulness link."])


def s04_then_now(prs):
    s = content_slide(prs, "Why June Failed - and What Changes Now",
                      "The June attempt (e4) learned something, then collapsed. Each cause has a fix.", 4)
    rows = [["", "June attempt (e4)", "Now"],
            ["Language model", "Qwen3-4B, text-only: never saw visual tokens",
             "Qwen3-VL-4B's language model: reads video tokens + 3D positions"],
            ["Projector", "64-query resampler, no position information",
             "Qwen3-VL's 2×2 merger: 512 tokens, position kept"],
            ["Training", "One step: 89 scenes / 267 windows",
             "Coarse -> fine: MM-AU (1,962 crash videos, ~7.8k windows) -> Nexar"],
            ["Text", "Whole-clip captions",
             "Coarse: MM-AU human labels. Fine: V12 window captions"],
            ["Preprocessing", "Centre crop (hid side traffic)", "Full frame (compress256)"],
            ["Success test", "Validation loss -> collapsed to 2 templates",
             "Wrong-video test, no-video floor, diversity, facts"]]
    table(s, rows, [1.55, 3.55, 4.00], 0.45, 1.30, 2.95, font=9, header_font=9.5,
          bold_first_col=True, wrap_left_cols=(1, 2))
    hero(s, 0.45, 4.42, 9.10, 0.72, "The facts are in the features.",
         "  Probe on frozen tokens (Oct 2026): token position left/centre/right 86-96% (chance 33%) · "
         "agent side 60% (33%) · closing gap 53% (25%).", accent=GREEN, head_size=13, tail_size=10)
    notes(s, [
        "What went wrong in June, plainly:",
        "- The projector-only stage DID learn: perplexity -48% vs a random projector. The collapse came in fine-tuning.",
        "- Text-only Qwen3 had never seen visual tokens; we fed them into generic placeholder slots.",
        "- 64 resampler queries with no position: 'left / right / ahead' had no way through.",
        "- 267 windows from 89 scenes, one caption style: the model learned the style, not the video.",
        "- We paused in June to fix the crash path - preprocessing turned out to be the biggest AP lever.",
        "What is different now:",
        "- An LLM built to read video tokens, a projector that keeps position, and a coarse stage on public crash video.",
        "- MM-AU's DADA part has exact crash timing and a known 30 fps, so windows end 0.5 / 1 / 1.5 s before impact.",
        "- Success is tested by swapping the video: if the text does not get worse, the video is being ignored.",
        "- Precedent: Meta aligned V-JEPA2 to an LLM, but with 18-88.5M pairs - hence the one-week gate."])


def s05_timeline(prs):
    s = content_slide(prs, "Timeline - Three Weeks, Decision After Week One",
                      "Coarse on MM-AU, fine-tune on Nexar. Nothing is spent before the week-1 decision.", 5)
    rows = [["Week", "Required outcome", "Tasks"],
            ["Week #1\nOct 4-10", "GO / NO-GO: the LLM uses V-JEPA2 features through the merger",
             "Merger + native Qwen3-VL input · MM-AU DADA: windows + label sentences + encode (pod) · "
             "Coarse: merger only · Fine: + LoRA on 1,761 Nexar V12 windows · Gates on held-out Nexar · "
             "Access requests sent"],
            ["Decision\nend of week 1", "GO / enlarge / stop",
             "GO -> weeks 2-3 as below · Works but weak -> enlarge coarse data (MM-AU CAP, BDD-X, "
             "TAU / DRAMA) -> 5-6 weeks · NO-GO -> re-plan before any spend"],
            ["Week #2\nOct 11-17", "Complete Nexar fine-tune text",
             "V12 captions + detection boxes for all 4,446 Nexar windows (2,685 new, ~$100) · "
             "Retrain coarse -> fine"],
            ["Week #3\nOct 18-24", "Evaluated reasoning model",
             "Held-out Nexar: wrong-video gap, diversity, facts, verdict agrees with score · "
             "Sample review · Write-up"]]
    table(s, rows, [1.25, 2.55, 5.30], 0.45, 1.30, 3.30, font=8.8, header_font=9.5,
          bold_first_col=True, wrap_left_cols=(1, 2))
    hero(s, 0.45, 4.72, 9.10, 0.45, "Week-1 gate:",
         "  true video must beat a swapped video and blank video on held-out Nexar text - "
         "otherwise stop.", accent=ORANGE, head_size=12, tail_size=10.5)
    notes(s, [
        "Timeline and odds:",
        "- Week 1 answers one question: does the language model really use our features? ~65-70% chance of a clear answer.",
        "- Week 1 is tight: new input code for Qwen3-VL + streaming 131 GB of MM-AU DADA on the pod.",
        "- Fallback inside week 1: run the Nexar fine-tune check first, add the coarse stage next.",
        "- Pass criteria: true video beats swapped video (confidence interval above 0) and blank video;",
        "-   generated texts differ across clips (>= 45 of 50); side and closing gap beat the no-video floor.",
        "- Decision after week 1: three weeks as planned, or enlarge the coarse data -> 5-6 weeks.",
        "- Three weeks to an evaluated model: ~60%. The full version with all datasets and ablations: 5-6 weeks.",
        "- Cost: ~$100 teacher (Nexar captions only, at V12's logged $0.0365 per window) + ~15-25 pod GPU-hours."])


def main():
    verify()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
    s00_title(prs)
    s01_status(prs)
    s02_novelty(prs)
    s03_architecture(prs)
    s04_then_now(prs)
    s05_timeline(prs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"  [done] {OUT}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
