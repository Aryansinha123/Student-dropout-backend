# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Optional
# from fastapi.middleware.cors import CORSMiddleware
# import joblib
# import numpy as np

# from models.fuzzy_model import fuzzy_risk
# from models.hybrid_model import hybrid_score

# app = FastAPI()

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load models
# model = joblib.load("saved_models/ann_model.pkl")
# scaler = joblib.load("saved_models/scaler.pkl")
# feature_columns = joblib.load("saved_models/feature_columns.pkl")
# feature_means = joblib.load("saved_models/feature_means.pkl")

# REQUIRED_FIELDS = [
#     "Admission grade",
#     "Age at enrollment"
# ]

# class StudentInput(BaseModel):
#     data: dict

# @app.post("/predict")
# def predict(student: StudentInput):

#     input_data = student.data

#     # Required validation
#     for field in REQUIRED_FIELDS:
#         if field not in input_data:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"{field} is required"
#             )

#     # Fill missing fields
#     final_input = {}
#     for col in feature_columns:
#         if col in input_data:
#             final_input[col] = input_data[col]
#         else:
#             final_input[col] = feature_means[col]

#     # ANN prediction
#     input_array = np.array([list(final_input.values())])
#     input_scaled = scaler.transform(input_array)

#     ann_prob = model.predict_proba(input_scaled)[0][1]

#     # Fuzzy score
#     fuzzy_score_value = fuzzy_risk(final_input)

#     # Hybrid
#     final_score, category = hybrid_score(fuzzy_score_value, ann_prob)

#     return {
#         "fuzzyScore": fuzzy_score_value,
#         "annProbability": float(ann_prob),
#         "finalScore": float(final_score),
#         "category": category
#     }
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

from models.fuzzy_model import fuzzy_risk
from models.hybrid_model import hybrid_score

from fastapi.responses import FileResponse
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
import os


# pdf
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

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
# def generate_pdf_report(result_data, filename="risk_report.pdf"):

#     doc = SimpleDocTemplate(filename, pagesize=A4)
#     elements = []

#     styles = getSampleStyleSheet()
#     title_style = styles["Heading1"]
#     normal_style = styles["Normal"]

#     # Title
#     elements.append(Paragraph("Student Dropout Risk Report", title_style))
#     elements.append(Spacer(1, 0.5 * inch))

#     # Risk Summary
#     elements.append(Paragraph(f"Final Risk Score: {result_data['finalScore']:.2f}", normal_style))
#     elements.append(Paragraph(f"Category: {result_data['category']}", normal_style))
#     elements.append(Spacer(1, 0.3 * inch))

#     # Major Factors
#     elements.append(Paragraph("Major Risk Factors:", styles["Heading2"]))
#     factor_list = [
#         ListItem(Paragraph(factor, normal_style))
#         for factor in result_data["majorFactors"]
#     ]
#     elements.append(ListFlowable(factor_list, bulletType="bullet"))
#     elements.append(Spacer(1, 0.3 * inch))

#     # Trend Summary
#     elements.append(Paragraph("Trend Overview:", styles["Heading2"]))
#     for item in result_data["trend"]:
#         elements.append(
#             Paragraph(f"{item['month']} Risk: {item['risk']}", normal_style)
#         )

#     doc.build(elements)  

def generate_pdf_report(result_data, filename="risk_report.pdf"):

    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    normal_style = styles["Normal"]

    # ---------- HEADER ----------
    elements.append(Paragraph("<b>Student Dropout Risk Report</b>", title_style))
    elements.append(Spacer(1, 0.4 * inch))

    # ---------- RISK SUMMARY TABLE ----------
    risk_score = f"{result_data['finalScore']:.2f}"
    category = result_data["category"]

    if category == "High":
        category_color = colors.red
    elif category == "Medium":
        category_color = colors.orange
    else:
        category_color = colors.green

    summary_data = [
        ["Final Risk Score", risk_score],
        ["Risk Category", category]
    ]

    summary_table = Table(summary_data, colWidths=[200, 200])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("BACKGROUND", (0, 1), (-1, 1), category_color),
        ("TEXTCOLOR", (0, 1), (-1, 1), colors.white),
        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 12),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 0.5 * inch))

    # ---------- MAJOR FACTORS ----------
    elements.append(Paragraph("<b>Major Risk Factors</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    factor_data = [[factor] for factor in result_data["majorFactors"]]

    if not factor_data:
        factor_data = [["No major risk factors identified"]]

    factor_table = Table(factor_data, colWidths=[400])
    factor_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
    ]))

    elements.append(factor_table)
    elements.append(Spacer(1, 0.5 * inch))

    # ---------- TREND SECTION ----------
    elements.append(Paragraph("<b>Risk Trend Overview</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    trend_data = [["Month", "Risk Score"]]

    for item in result_data["trend"]:
        trend_data.append([item["month"], f"{item['risk']:.2f}"])

    trend_table = Table(trend_data, colWidths=[200, 200])
    trend_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
    ]))

    elements.append(trend_table)
    elements.append(Spacer(1, 0.5 * inch))

    # ---------- FOOTER ----------
    elements.append(
        Paragraph(
            "<i>This report is generated using a Hybrid Soft Computing Model (ANN + Fuzzy Logic).</i>",
            normal_style
        )
    )

    doc.build(elements)

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