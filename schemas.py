from pydantic import BaseModel, Field
from typing import List

class AIAnalysisResponse(BaseModel):
    status: str = Field(..., example="CRITICAL ALERT")
    color: str = Field(..., example="#dc3545")
    analysis: str = Field(..., example="Patient shows signs of severe acute respiratory distress.")
    suggestions: List[str] = Field(..., example=["Administer oxygen", "Check vitals"])

class LabFindingRow(BaseModel):
    test: str = Field(..., example="White Blood Cell Count (WBC)")
    value: str = Field(..., example="12.4 x10^3 / µL")
    range: str = Field(..., example="4.5 - 11.0")
    flag: str = Field(..., example="HIGH")
    severity: str = Field(..., example="#dc3545")

class LabReportAnalysisResponse(BaseModel):
    patientName: str = Field(..., example="Jane Doe")
    reportDate: str = Field(..., example="2026-07-05")
    laboratory: str = Field(..., example="Core Health Diagnostics Labs")
    findings: List[LabFindingRow]
    aiSummary: str = Field(..., example="Structured overview summary goes here.")

class BloodBiomarkerRow(BaseModel):
    test: str = Field(..., description="Name of the blood test, e.g., Hemoglobin")
    value: str = Field(..., description="Extracted numerical value or quantitative state")
    range: str = Field(..., description="Normal demographic reference range")
    flag: str = Field(..., description="NORMAL, LOW, or HIGH triage classification status")
    severity: str = Field(..., description="Hex color formatting string matching clinical urgency")

class BloodReportAnalysisResponse(BaseModel):
    patientName: str = Field(..., example="Jane Doe")
    reportDate: str = Field(..., example="2026-07-05")
    laboratory: str = Field(..., example="Core Health Diagnostics Labs")
    findings: List[BloodBiomarkerRow]
    aiSummary: str = Field(..., description="Comprehensive diagnosis conclusion and clinical correlation advice.")

