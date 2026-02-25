# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# import joblib
# import numpy as np

# from models.fuzzy_model import fuzzy_risk
# from models.hybrid_model import hybrid_score

# from fastapi.responses import FileResponse
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
# from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
# from reportlab.lib import colors
# from reportlab.lib.units import inch
# from reportlab.lib.pagesizes import A4
# import os


# # pdf
# from reportlab.platypus import (
#     SimpleDocTemplate,
#     Paragraph,
#     Spacer,
#     Table,
#     TableStyle
# )
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.lib import colors
# from reportlab.lib.units import inch
# from reportlab.lib.pagesizes import A4

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# model = joblib.load("saved_models/ann_model.pkl")
# scaler = joblib.load("saved_models/scaler.pkl")
# feature_columns = joblib.load("saved_models/feature_columns.pkl")
# feature_means = joblib.load("saved_models/feature_means.pkl")

# REQUIRED_FIELDS = ["Admission grade", "Age at enrollment"]

# class StudentInput(BaseModel):
#     data: dict

# def evaluate_feature_risk(value, mean):
#     if value > mean * 1.2:
#         return "Safe"
#     elif value > mean * 0.8:
#         return "Worry"
#     else:
#         return "Critical"

# @app.post("/predict")
# def predict(student: StudentInput):

#     input_data = student.data

#     for field in REQUIRED_FIELDS:
#         if field not in input_data:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"{field} is required"
#             )

#     final_input = {}

#     for col in feature_columns:
#         if col in input_data:
#             final_input[col] = input_data[col]
#         else:
#             final_input[col] = feature_means[col]

#     input_array = np.array([list(final_input.values())])
#     input_scaled = scaler.transform(input_array)

#     ann_prob = model.predict_proba(input_scaled)[0][1]

#     fuzzy_score_value = fuzzy_risk(final_input)

#     final_score, category = hybrid_score(fuzzy_score_value, ann_prob)

#     # ---- Major Risk Factors ----
#     major_factors = []

#     for key in final_input:
#         if abs(final_input[key] - feature_means[key]) > feature_means[key] * 0.5:
#             major_factors.append(key)

#     major_factors = major_factors[:5]

#     # ---- Feature Risk Levels ----
#     feature_risk_levels = {}

#     for key in final_input:
#         feature_risk_levels[key] = evaluate_feature_risk(
#             final_input[key],
#             feature_means[key]
#         )

#     # ---- Simulated Trend ----
#     simulated_trend = [
#         {"month": "Month 1", "risk": max(final_score - 15, 0)},
#         {"month": "Month 2", "risk": max(final_score - 5, 0)},
#         {"month": "Month 3", "risk": final_score},
#     ]

#     return {
#         "fuzzyScore": fuzzy_score_value,
#         "annProbability": float(ann_prob),
#         "finalScore": float(final_score),
#         "category": category,
#         "majorFactors": major_factors,
#         "featureRiskLevels": feature_risk_levels,
#         "trend": simulated_trend
#     }

# def generate_pdf_report(result_data, filename="risk_report.pdf"):

#     doc = SimpleDocTemplate(filename, pagesize=A4)
#     elements = []

#     styles = getSampleStyleSheet()
#     title_style = styles["Heading1"]
#     normal_style = styles["Normal"]

#     # ---------- HEADER ----------
#     elements.append(Paragraph("<b>Student Dropout Risk Report</b>", title_style))
#     elements.append(Spacer(1, 0.4 * inch))

#     # ---------- RISK SUMMARY TABLE ----------
#     risk_score = f"{result_data['finalScore']:.2f}"
#     category = result_data["category"]

#     if category == "High":
#         category_color = colors.red
#     elif category == "Medium":
#         category_color = colors.orange
#     else:
#         category_color = colors.green

#     summary_data = [
#         ["Final Risk Score", risk_score],
#         ["Risk Category", category]
#     ]

#     summary_table = Table(summary_data, colWidths=[200, 200])
#     summary_table.setStyle(TableStyle([
#         ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
#         ("BACKGROUND", (0, 1), (-1, 1), category_color),
#         ("TEXTCOLOR", (0, 1), (-1, 1), colors.white),
#         ("GRID", (0, 0), (-1, -1), 1, colors.grey),
#         ("ALIGN", (1, 0), (1, -1), "CENTER"),
#         ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
#         ("FONTSIZE", (0, 0), (-1, -1), 12),
#     ]))

#     elements.append(summary_table)
#     elements.append(Spacer(1, 0.5 * inch))

#     # ---------- MAJOR FACTORS ----------
#     elements.append(Paragraph("<b>Major Risk Factors</b>", styles["Heading2"]))
#     elements.append(Spacer(1, 0.2 * inch))

#     factor_data = [[factor] for factor in result_data["majorFactors"]]

#     if not factor_data:
#         factor_data = [["No major risk factors identified"]]

#     factor_table = Table(factor_data, colWidths=[400])
#     factor_table.setStyle(TableStyle([
#         ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
#         ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
#         ("FONTSIZE", (0, 0), (-1, -1), 11),
#     ]))

#     elements.append(factor_table)
#     elements.append(Spacer(1, 0.5 * inch))

#     # ---------- TREND SECTION ----------
#     elements.append(Paragraph("<b>Risk Trend Overview</b>", styles["Heading2"]))
#     elements.append(Spacer(1, 0.2 * inch))

#     trend_data = [["Month", "Risk Score"]]

#     for item in result_data["trend"]:
#         trend_data.append([item["month"], f"{item['risk']:.2f}"])

#     trend_table = Table(trend_data, colWidths=[200, 200])
#     trend_table.setStyle(TableStyle([
#         ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
#         ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
#         ("ALIGN", (1, 1), (-1, -1), "CENTER"),
#         ("FONTSIZE", (0, 0), (-1, -1), 11),
#     ]))

#     elements.append(trend_table)
#     elements.append(Spacer(1, 0.5 * inch))

#     # ---------- FOOTER ----------
#     elements.append(
#         Paragraph(
#             "<i>This report is generated using a Hybrid Soft Computing Model (ANN + Fuzzy Logic).</i>",
#             normal_style
#         )
#     )

#     doc.build(elements)

# @app.post("/download-report")
# def download_report(student: StudentInput):

#     # Reuse predict logic
#     prediction = predict(student)

#     filename = "student_risk_report.pdf"

#     generate_pdf_report(prediction, filename)

#     return FileResponse(
#         path=filename,
#         filename="Student_Risk_Report.pdf",
#         media_type="application/pdf"
#     )    

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

from models.fuzzy_model import fuzzy_risk
from models.hybrid_model import hybrid_score

from fastapi.responses import FileResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("saved_models/ann_model.pkl")
scaler = joblib.load("saved_models/scaler.pkl")
feature_columns = joblib.load("saved_models/feature_columns.pkl")
feature_means = joblib.load("saved_models/feature_means.pkl")

REQUIRED_FIELDS = ["Admission grade", "Age at enrollment"]

class StudentInput(BaseModel):
    data: dict

def evaluate_feature_risk(value, mean):
    if value > mean * 1.2:
        return "Safe"
    elif value > mean * 0.8:
        return "Worry"
    else:
        return "Critical"

@app.post("/predict")
def predict(student: StudentInput):

    input_data = student.data

    for field in REQUIRED_FIELDS:
        if field not in input_data:
            raise HTTPException(
                status_code=400,
                detail=f"{field} is required"
            )

    final_input = {}

    for col in feature_columns:
        if col in input_data:
            final_input[col] = input_data[col]
        else:
            final_input[col] = feature_means[col]

    input_array = np.array([list(final_input.values())])
    input_scaled = scaler.transform(input_array)

    ann_prob = model.predict_proba(input_scaled)[0][1]

    fuzzy_score_value = fuzzy_risk(final_input)

    final_score, category = hybrid_score(fuzzy_score_value, ann_prob)

    # ---- Major Risk Factors ----
    major_factors = []

    for key in final_input:
        if abs(final_input[key] - feature_means[key]) > feature_means[key] * 0.5:
            major_factors.append(key)

    major_factors = major_factors[:5]

    # ---- Feature Risk Levels ----
    feature_risk_levels = {}

    for key in final_input:
        feature_risk_levels[key] = evaluate_feature_risk(
            final_input[key],
            feature_means[key]
        )

    # ---- Simulated Trend ----
    simulated_trend = [
        {"month": "Month 1", "risk": max(final_score - 15, 0)},
        {"month": "Month 2", "risk": max(final_score - 5, 0)},
        {"month": "Month 3", "risk": final_score},
    ]

    return {
        "fuzzyScore": fuzzy_score_value,
        "annProbability": float(ann_prob),
        "finalScore": float(final_score),
        "category": category,
        "majorFactors": major_factors,
        "featureRiskLevels": feature_risk_levels,
        "trend": simulated_trend
    }


# ─────────────────────────────────────────────────────────────────────────────
# PDF GENERATION  (UI only changed — all logic above is untouched)
# ─────────────────────────────────────────────────────────────────────────────

# Register fonts once at module load
try:
    pdfmetrics.registerFont(TTFont('Serif',      '/usr/share/fonts/truetype/freefont/FreeSerif.ttf'))
    pdfmetrics.registerFont(TTFont('SerifBold',  '/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf'))
    pdfmetrics.registerFont(TTFont('SerifItalic','/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf'))
    pdfmetrics.registerFont(TTFont('Sans',       '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'))
    pdfmetrics.registerFont(TTFont('SansBold',   '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'))
    pdfmetrics.registerFont(TTFont('SansItalic', '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf'))
    _FONTS_LOADED = True
except Exception:
    _FONTS_LOADED = False  # fall back to Helvetica if fonts not present

def _f(name, fallback):
    """Return font name if custom fonts loaded, else Helvetica variant."""
    return name if _FONTS_LOADED else fallback

# ── Palette ──────────────────────────────────────────────────────────────────
_INK        = colors.HexColor("#1a1814")
_INK_SOFT   = colors.HexColor("#252118")
_PARCHMENT  = colors.HexColor("#f5f0e8")
_GOLD       = colors.HexColor("#b8933f")
_GOLD_DIM   = colors.HexColor("#2a2318")
_MUTED      = colors.HexColor("#8a7f6e")
_RED        = colors.HexColor("#c0392b")
_AMBER      = colors.HexColor("#b8760a")
_GREEN      = colors.HexColor("#2e7d52")
_WHITE      = colors.white

PAGE_W, PAGE_H = A4
_ML = 18 * mm          # margin left
_MR = 18 * mm          # margin right
_CW = PAGE_W - _ML - _MR  # content width


def _category_palette(category):
    if category == "High":
        return _RED,   colors.HexColor("#2e1a16"), colors.HexColor("#8b3a2a"), colors.HexColor("#f1948a"), "CRITICAL RISK"
    elif category == "Medium":
        return _AMBER, colors.HexColor("#2a2110"), colors.HexColor("#7a5a10"), colors.HexColor("#d4aa5a"), "MODERATE RISK"
    else:
        return _GREEN, colors.HexColor("#162418"), colors.HexColor("#2e7d52"), colors.HexColor("#82c99a"), "LOW RISK"


def _draw_background(c):
    c.setFillColor(_INK)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(_INK_SOFT)
    c.rect(0, PAGE_H - 70 * mm, PAGE_W, 70 * mm, fill=1, stroke=0)
    # corner brackets
    c.setStrokeColor(_GOLD)
    c.setLineWidth(0.6)
    s = 12 * mm
    for bx, by, dx, dy in [
        (_ML,             PAGE_H - 10*mm,  1,  -1),
        (PAGE_W - _MR,    PAGE_H - 10*mm, -1,  -1),
        (_ML,             10*mm,           1,   1),
        (PAGE_W - _MR,    10*mm,          -1,   1),
    ]:
        c.line(bx, by, bx + dx * s, by)
        c.line(bx, by, bx, by + dy * s)


def _gold_rule(c, y, x=None, w=None, weight=0.5):
    x = x or _ML
    w = w or _CW
    c.setStrokeColor(_GOLD)
    c.setLineWidth(weight)
    c.line(x, y, x + w, y)


def _dim_rule(c, y):
    c.setStrokeColor(_GOLD_DIM)
    c.setLineWidth(0.4)
    c.line(_ML, y, _ML + _CW, y)


def _section_header(c, label, y):
    _gold_rule(c, y + 1.5 * mm, weight=0.4)
    c.setFont(_f("SansBold", "Helvetica-Bold"), 6.5)
    c.setFillColor(_GOLD)
    c.drawString(_ML, y - 5 * mm, label.upper())
    _dim_rule(c, y - 8 * mm)
    return y - 13 * mm


def _gauge_bar(c, x, y, w, h, value, accent):
    c.setFillColor(_GOLD_DIM)
    c.roundRect(x, y, w, h, 1.5 * mm, fill=1, stroke=0)
    fill_w = max(4 * mm, (value / 100.0) * w)
    c.setFillColor(accent)
    c.roundRect(x, y, fill_w, h, 1.5 * mm, fill=1, stroke=0)
    c.setStrokeColor(_GOLD)
    c.setLineWidth(0.4)
    c.roundRect(x, y, w, h, 1.5 * mm, fill=0, stroke=1)
    c.setStrokeColor(colors.HexColor("#3a3428"))
    c.setLineWidth(0.3)
    for tick in [25, 50, 75]:
        tx = x + (tick / 100.0) * w
        c.line(tx, y, tx, y + h)


def _footer(c, page_num):
    _gold_rule(c, 16 * mm, weight=0.4)
    c.setFont(_f("SansItalic", "Helvetica-Oblique"), 6.5)
    c.setFillColor(_MUTED)
    c.drawCentredString(PAGE_W / 2, 10 * mm,
        f"Generated by Hybrid Soft Computing Model (ANN + Fuzzy Logic)  ·  Page {page_num}")


def generate_pdf_report(result_data, filename="risk_report.pdf"):

    final_score         = result_data["finalScore"]
    category            = result_data["category"]
    major_factors       = result_data.get("majorFactors", [])
    feature_risk_levels = result_data.get("featureRiskLevels", {})
    trend               = result_data.get("trend", [])
    fuzzy_score         = result_data.get("fuzzyScore", None)
    ann_prob            = result_data.get("annProbability", None)

    accent, bg_col, border_col, text_col, badge_text = _category_palette(category)

    c = canvas.Canvas(filename, pagesize=A4)
    c.setTitle("Student Dropout Risk Report")
    c.setAuthor("Hybrid Soft Computing Model")

    # ════════════════════════════════════════════════════
    # PAGE 1
    # ════════════════════════════════════════════════════
    _draw_background(c)

    # ── Masthead ─────────────────────────────────────────
    c.setFont(_f("Sans", "Helvetica"), 7)
    c.setFillColor(_GOLD)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 14 * mm,
        "STUDENT DROPOUT RISK INTELLIGENCE SYSTEM")
    _gold_rule(c, PAGE_H - 17 * mm, weight=0.8)
    _dim_rule(c, PAGE_H - 18.5 * mm)

    c.setFont(_f("SerifBold", "Times-Bold"), 28)
    c.setFillColor(_PARCHMENT)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 34 * mm, "Risk Assessment Report")

    c.setFont(_f("SerifItalic", "Times-Italic"), 11)
    c.setFillColor(_MUTED)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 43 * mm,
        "Hybrid Soft Computing Model  ·  ANN + Fuzzy Logic")

    _gold_rule(c, PAGE_H - 49 * mm, weight=0.8)
    _dim_rule(c, PAGE_H - 50.5 * mm)

    # ── Score Card ───────────────────────────────────────
    card_top = PAGE_H - 52 * mm
    card_h   = 54 * mm
    c.setFillColor(colors.HexColor("#22201a"))
    c.roundRect(_ML, card_top - card_h, _CW, card_h, 3 * mm, fill=1, stroke=0)
    c.setStrokeColor(_GOLD)
    c.setLineWidth(0.7)
    c.roundRect(_ML, card_top - card_h, _CW, card_h, 3 * mm, fill=0, stroke=1)

    mid_y = card_top - card_h / 2

    # Big score
    score_x = _ML + 18 * mm
    c.setFont(_f("SansBold", "Helvetica-Bold"), 6.5)
    c.setFillColor(_MUTED)
    c.drawString(score_x, mid_y + 9 * mm, "FINAL RISK SCORE")
    c.setFont(_f("SerifBold", "Times-Bold"), 52)
    c.setFillColor(accent)
    c.drawString(score_x, mid_y - 7 * mm, f"{final_score:.1f}")
    c.setFont(_f("Serif", "Times-Roman"), 14)
    c.setFillColor(_MUTED)
    c.drawString(score_x, mid_y - 15 * mm, "/ 100")

    # Divider
    div_x = _ML + 68 * mm
    c.setStrokeColor(colors.HexColor("#3a3428"))
    c.setLineWidth(0.5)
    c.line(div_x, card_top - card_h + 7 * mm, div_x, card_top - 7 * mm)

    # Right side
    rx = div_x + 10 * mm
    rw = _CW - (div_x - _ML) - 16 * mm

    # Badge
    bw, bh = rw * 0.72, 8 * mm
    c.setFillColor(accent)
    c.roundRect(rx, mid_y + 3 * mm, bw, bh, 2 * mm, fill=1, stroke=0)
    c.setFont(_f("SansBold", "Helvetica-Bold"), 7.5)
    c.setFillColor(_WHITE)
    c.drawCentredString(rx + bw / 2, mid_y + 5.5 * mm, badge_text)

    # Bar
    _gauge_bar(c, rx, mid_y - 6 * mm, rw, 5 * mm, final_score, accent)
    c.setFont(_f("Sans", "Helvetica"), 6)
    c.setFillColor(_MUTED)
    for tv, tl in [(0, "0"), (50, "50"), (100, "100")]:
        c.drawCentredString(rx + (tv / 100.0) * rw, mid_y - 10 * mm, tl)

    # Sub-scores
    if fuzzy_score is not None:
        c.setFont(_f("Sans", "Helvetica"), 6.5)
        c.setFillColor(_MUTED)
        c.drawString(rx, mid_y - 16 * mm, "FUZZY SCORE")
        c.setFont(_f("SansBold", "Helvetica-Bold"), 9)
        c.setFillColor(_PARCHMENT)
        c.drawString(rx, mid_y - 21 * mm, f"{fuzzy_score:.2f}")
    if ann_prob is not None:
        ax = rx + rw * 0.5
        c.setFont(_f("Sans", "Helvetica"), 6.5)
        c.setFillColor(_MUTED)
        c.drawString(ax, mid_y - 16 * mm, "ANN PROBABILITY")
        c.setFont(_f("SansBold", "Helvetica-Bold"), 9)
        c.setFillColor(_PARCHMENT)
        c.drawString(ax, mid_y - 21 * mm, f"{ann_prob:.2%}")

    cur_y = card_top - card_h - 10 * mm

    # ── Major Risk Factors ────────────────────────────────
    cur_y = _section_header(c, "Major Risk Factors", cur_y)
    cur_y -= 2 * mm

    if major_factors:
        cols = 2
        col_w = _CW / cols
        for i, factor in enumerate(major_factors):
            col = i % cols
            row = i // cols
            fx = _ML + col * col_w
            fy = cur_y - row * 9 * mm
            pw, ph = col_w - 6 * mm, 7 * mm
            c.setFillColor(bg_col)
            c.roundRect(fx, fy - 1.5 * mm, pw, ph, 1.5 * mm, fill=1, stroke=0)
            c.setStrokeColor(border_col)
            c.setLineWidth(0.4)
            c.roundRect(fx, fy - 1.5 * mm, pw, ph, 1.5 * mm, fill=0, stroke=1)
            c.setFillColor(accent)
            c.circle(fx + 4 * mm, fy + 2 * mm, 1.4 * mm, fill=1, stroke=0)
            c.setFont(_f("Sans", "Helvetica"), 8)
            c.setFillColor(text_col)
            c.drawString(fx + 8 * mm, fy + 1 * mm, factor[:38])
        rows = (len(major_factors) - 1) // cols + 1
        cur_y -= rows * 9 * mm + 6 * mm
    else:
        c.setFont(_f("SansItalic", "Helvetica-Oblique"), 8)
        c.setFillColor(_MUTED)
        c.drawString(_ML, cur_y, "No major risk factors identified.")
        cur_y -= 10 * mm

    # ── Trend Bar Chart ───────────────────────────────────
    cur_y -= 4 * mm
    cur_y = _section_header(c, "Risk Trend Overview", cur_y)
    cur_y -= 2 * mm

    if trend:
        chart_h      = 28 * mm
        chart_bottom = cur_y - chart_h
        c.setStrokeColor(colors.HexColor("#3a3428"))
        c.setLineWidth(0.4)
        c.line(_ML, chart_bottom, _ML + _CW, chart_bottom)
        bar_sec_w = _CW / len(trend)
        for i, point in enumerate(trend):
            bx = _ML + i * bar_sec_w + bar_sec_w * 0.2
            bw = bar_sec_w * 0.6
            bh = max((point["risk"] / 100.0) * chart_h, 1 * mm)
            c.setFillColor(accent)
            c.roundRect(bx, chart_bottom, bw, bh, 1 * mm, fill=1, stroke=0)
            c.setFont(_f("SansBold", "Helvetica-Bold"), 7.5)
            c.setFillColor(_PARCHMENT)
            c.drawCentredString(bx + bw / 2, chart_bottom + bh + 2 * mm, f"{point['risk']:.1f}")
            c.setFont(_f("Sans", "Helvetica"), 6.5)
            c.setFillColor(_MUTED)
            c.drawCentredString(bx + bw / 2, chart_bottom - 5 * mm, point["month"])
        cur_y = chart_bottom - 10 * mm

    _footer(c, 1)

    # ════════════════════════════════════════════════════
    # PAGE 2 – Feature Risk Grid
    # ════════════════════════════════════════════════════
    if feature_risk_levels:
        c.showPage()
        _draw_background(c)
        c.setFont(_f("Sans", "Helvetica"), 7)
        c.setFillColor(_GOLD)
        c.drawCentredString(PAGE_W / 2, PAGE_H - 14 * mm,
            "STUDENT DROPOUT RISK INTELLIGENCE SYSTEM")
        _gold_rule(c, PAGE_H - 17 * mm, weight=0.8)
        c.setFont(_f("SerifBold", "Times-Bold"), 20)
        c.setFillColor(_PARCHMENT)
        c.drawCentredString(PAGE_W / 2, PAGE_H - 32 * mm, "Feature Risk Classification")
        _gold_rule(c, PAGE_H - 38 * mm, weight=0.4)

        status_styles = {
            "Critical": (colors.HexColor("#2e1a16"), colors.HexColor("#8b3a2a"), colors.HexColor("#f1948a")),
            "Worry":    (colors.HexColor("#2a2110"), colors.HexColor("#7a5a10"), colors.HexColor("#d4aa5a")),
            "Safe":     (colors.HexColor("#162418"), colors.HexColor("#2e7d52"), colors.HexColor("#82c99a")),
        }

        grid_cols = 3
        pad       = 3 * mm
        cell_w    = (_CW - pad * (grid_cols - 1)) / grid_cols
        cell_h    = 11 * mm
        gap_y     = 4 * mm
        top_y     = PAGE_H - 46 * mm
        col_idx   = 0
        row_idx   = 0
        page_num  = 2

        for feature, status in feature_risk_levels.items():
            col = col_idx % grid_cols
            cx  = _ML + col * (cell_w + pad)
            cy  = top_y - row_idx * (cell_h + gap_y)

            if cy - cell_h < 20 * mm:
                _footer(c, page_num)
                c.showPage()
                _draw_background(c)
                _gold_rule(c, PAGE_H - 17 * mm, weight=0.8)
                top_y    = PAGE_H - 28 * mm
                row_idx  = 0
                page_num += 1
                cy = top_y

            sb_col, sb_border, sb_text = status_styles.get(status, status_styles["Safe"])
            c.setFillColor(sb_col)
            c.roundRect(cx, cy - cell_h, cell_w, cell_h, 1.5 * mm, fill=1, stroke=0)
            c.setStrokeColor(sb_border)
            c.setLineWidth(0.4)
            c.roundRect(cx, cy - cell_h, cell_w, cell_h, 1.5 * mm, fill=0, stroke=1)
            c.setFillColor(sb_border)
            c.circle(cx + 3.5 * mm, cy - cell_h / 2, 1.5 * mm, fill=1, stroke=0)
            feat_label = feature[:22] + ("..." if len(feature) > 22 else "")
            c.setFont(_f("Sans", "Helvetica"), 7)
            c.setFillColor(_PARCHMENT)
            c.drawString(cx + 7 * mm, cy - cell_h / 2 + 1.2 * mm, feat_label)
            c.setFont(_f("SansBold", "Helvetica-Bold"), 5.5)
            c.setFillColor(sb_text)
            c.drawString(cx + 7 * mm, cy - cell_h / 2 - 3.5 * mm, status.upper())

            col_idx += 1
            if col_idx % grid_cols == 0:
                row_idx += 1

        _footer(c, page_num)

    c.save()


@app.post("/download-report")
def download_report(student: StudentInput):

    # Reuse predict logic
    prediction = predict(student)

    filename = "student_risk_report.pdf"

    generate_pdf_report(prediction, filename)

    return FileResponse(
        path=filename,
        filename="Student_Risk_Report.pdf",
        media_type="application/pdf"
    )