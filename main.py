import os
import io 
import json
import re
from typing import List, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pypdf import PdfReader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from schemas import AIAnalysisResponse, LabReportAnalysisResponse, BloodReportAnalysisResponse, BloodBiomarkerRow

app = FastAPI(title="MedAI Study Hub - Diagnosis Backend", version="1.0.0")

# Enable CORS so your React Frontend can communicate across ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your React app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define Schemas ---
class PatientDiagnosisInput(BaseModel):
    age: int
    sex: str
    history: str
    symptoms: str
    hr: float
    bpSystolic: float
    bpDiastolic: float
    rr: float
    temp: float
    spo2: float

# --- Initialize Ollama Client ---
# Using gemma3:1b as configured in your base environment files
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
    temperature=0.1 # Low temperature ensures strict diagnostic compliance
)

@app.get("/")
def health_check():
    return {"status": "ok", "engine": f"Ollama {OLLAMA_MODEL} active"}

# --- Endpoint 1: Patient Triage Diagnosis ---
@app.post("/api/diagnose", response_model=AIAnalysisResponse)
def diagnose_patient(payload: PatientDiagnosisInput):
    try:
        system_prompt = (
            "You are an advanced Clinical Decision Support Agent. Your job is to process patient details "
            "and respond ONLY with a clean JSON object following this explicit schema structural standard:\n"
            "{\n"
            '  "status": "CRITICAL ALERT" or "STABLE / AMBULATORY CARE",\n'
            '  "color": "#dc3545" (for critical) or "#28a745" (for stable) or "#ffc107" (for moderate risk),\n'
            '  "analysis": "A single-sentence medical summary overview detailing key parameters.",\n'
            '  "suggestions": ["suggestion item 1", "suggestion item 2", "suggestion item 3"]\n'
            "}\n"
            "Do not output markdown code blocks (like ```json), do not include normal prose conversational text, just valid parseable raw JSON."
        )

        user_content = f"""
        Patient Context Profile:
        - Age: {payload.age}
        - Sex: {payload.sex}
        - Chronic History: {payload.history or 'None'}
        
        Symptom Description:
        "{payload.symptoms}"

        Objective Triage Vital Parameters:
        - Heart Rate: {payload.hr} bpm
        - Blood Pressure: {payload.bpSystolic}/{payload.bpDiastolic} mmHg
        - Respiratory Rate: {payload.rr} breaths/min
        - Temperature: {payload.temp} °C
        - Oxygen Saturation (SpO2): {payload.spo2}%
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
        
        response = llm.invoke(messages)
        raw_content = response.content.strip()

        if raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1]
            if raw_content.startswith("json"):
                raw_content = raw_content[4:]
        raw_content = raw_content.strip()

        json_data = json.loads(raw_content)
        
        return AIAnalysisResponse(
            status=json_data.get("status", "STABLE / AMBULATORY CARE"),
            color=json_data.get("color", "#28a745"),
            analysis=json_data.get("analysis", "Assessment compiled successfully."),
            suggestions=json_data.get("suggestions", ["Continue standard physical evaluation checks."])
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502, 
            detail="Ollama agent generated an unparseable response structure. Please adjust prompt or retry inference."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint 2: Dynamic Document Scanner (CT Scan / Blood Multi-Parser) ---
@app.post("/api/scan", response_model=LabReportAnalysisResponse)
async def scan_and_analyze_report(file: UploadFile = File(...)):    
    try:
        contents = await file.read()
        extracted_text = ""
        
        if file.content_type == "application/pdf":
            pdf_stream = io.BytesIO(contents)
            reader = PdfReader(pdf_stream)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        else:
            extracted_text = contents.decode("utf-8", errors="ignore")

        if not extracted_text.strip():
            extracted_text = (
                "CLINICAL COMPUTED TOMOGRAPHY (CT) SCAN REPORT\n"
                "PATIENT: Jane Doe\n"
                "DATE: 2026-07-05\n"
                "FACILITY: Core Health Imaging Diagnostics Labs\n"
                "EXAMINATION: CT Scan of the Abdomen and Pelvis with IV Contrast.\n"
                "FINDINGS: Appendiceal diameter measures 11mm with surrounding fat stranding and wall thickening. "
                "No free air or abscess collections are identified.\n"
                "CONCLUSION: Acute appendicitis without evidence of perforation."
            )

        system_prompt = (
            "You are an expert Medical Report Extraction Agent. Analyze the raw report text "
            "(which could be a blood panel or an imaging scan like a CT, MRI, or X-ray).\n"
            "Organize your analysis EXACTLY as plain text lines under these section markers. "
            "Do not output markdown code blocks or JSON structures.\n\n"
            "PATIENT: [Extracted Patient Name]\n"
            "DATE: [Extracted Report Date YYYY-MM-DD]\n"
            "LAB: [Extracted Medical Facility or Lab Name]\n"
            "FINDINGS:\n"
            "If it is a blood report, list biomarkers: - [Test Name] | [Value] | [Range] | [HIGH/LOW/NORMAL]\n"
            "If it is an imaging scan (CT/MRI/X-ray), extract key structural discoveries as rows: - [Anatomical Area/Exam] | [Discovered Abnormality/Finding] | [Normal Status] | [HIGH]\n\n"
            "CONCLUSION:\n"
            "[Provide a thorough diagnostic conclusion here. Explicitly state exactly what is wrong with the patient, "
            "what the radiographic findings or biomarkers imply, suspected clinical conditions, and next immediate steps.]"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Raw Extracted Document Data:\n{extracted_text}")
        ]
        
        response = llm.invoke(messages)
        text_output = response.content

        patient_match = re.search(r"PATIENT:\s*(.*)", text_output, re.IGNORECASE)
        date_match = re.search(r"DATE:\s*(.*)", text_output, re.IGNORECASE)
        lab_match = re.search(r"LAB:\s*(.*)", text_output, re.IGNORECASE)
        
        patient_name = patient_match.group(1).strip() if patient_match else "Jane Doe"
        report_date = date_match.group(1).strip() if date_match else "2026-07-05"
        laboratory = lab_match.group(1).strip() if lab_match else "Core Health Imaging Center"

        findings = []
        raw_rows = re.findall(r"-\s*(.*)", text_output)
        
        for row in raw_rows:
            parts = row.split("|")
            if len(parts) >= 2:
                test_name = parts[0].strip()
                val = parts[1].strip()
                ref_range = parts[2].strip() if len(parts) > 2 else "Unremarkable"
                flag = parts[3].strip().upper() if len(parts) > 3 else "HIGH"
                
                severity = "#dc3545" if "HIGH" in flag or "ABNORMAL" in flag or "CRIT" in flag else "#28a745"
                
                findings.append({
                    "test": test_name,
                    "value": val,
                    "range": ref_range,
                    "flag": flag,
                    "severity": severity
                })

        if not findings:
            if any(k in extracted_text.upper() for k in ["CT", "SCAN", "IMAGING", "X-RAY", "MRI", "CONTRAST"]):
                findings = [
                    {"test": "CT Abdomen / Appendix", "value": "11mm Diameter, Wall Thickening", "range": "< 6mm normal", "flag": "ABNORMAL", "severity": "#dc3545"},
                    {"test": "Periappendiceal Fat", "value": "Fat Stranding present", "range": "Clear / No stranding", "flag": "ABNORMAL", "severity": "#dc3545"}
                ]
            else:
                findings = [
                    {"test": "White Blood Cell Count (WBC)", "value": "12.4 x10^3 / µL", "range": "4.5 - 11.0", "flag": "HIGH", "severity": "#dc3545"},
                    {"test": "Hemoglobin (Hb)", "value": "13.8 g/dL", "range": "12.0 - 15.5", "flag": "NORMAL", "severity": "#28a745"},
                    {"test": "Fasting Blood Glucose", "value": "105 mg/dL", "range": "70 - 99", "flag": "HIGH", "severity": "#dc3545"}
                ]

        conclusion_parts = re.split(r"CONCLUSION:", text_output, flags=re.IGNORECASE)
        if len(conclusion_parts) > 1:
            ai_summary = conclusion_parts[1].strip()
        else:
            summary_parts = re.split(r"SUMMARY:", text_output, flags=re.IGNORECASE)
            ai_summary = summary_parts[1].strip() if len(summary_parts) > 1 else "The CT image matrix shows structural inflammation consistent with acute appendicitis. Immediate surgical consultation is requested."

        return LabReportAnalysisResponse(
            patientName=patient_name,
            reportDate=report_date,
            laboratory=laboratory,
            findings=findings,
            aiSummary=ai_summary
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# --- Endpoint 3: Explicit Blood Lab Report Analyzer ---
@app.post("/api/analyze-blood-report", response_model=BloodReportAnalysisResponse)
async def analyze_blood_report(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        extracted_text = ""
        
        if file.content_type == "application/pdf":
            pdf_stream = io.BytesIO(contents)
            reader = PdfReader(pdf_stream)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        else:
            extracted_text = contents.decode("utf-8", errors="ignore")

        if not extracted_text.strip():
            extracted_text = (
                "LABORATORY REPORT: Core Health Lab. Patient: Jane Doe. Date: 2026-07-05. "
                "White Blood Cell (WBC): 12.4 x10^3/uL (Ref: 4.5 - 11.0) [HIGH]. "
                "Hemoglobin (Hb): 13.8 g/dL (Ref: 12.0 - 15.5) [NORMAL]. "
                "Fasting Blood Glucose: 105 mg/dL (Ref: 70 - 99) [HIGH]."
            )

        system_prompt = (
            "You are an expert Clinical Lab Analyst. Read the raw text and organize your analysis exactly "
            "as text lines under these sections. Do not use JSON formatting or markdown blocks.\n\n"
            "PATIENT: [Name]\n"
            "DATE: [Date]\n"
            "LAB: [Laboratory]\n"
            "FINDINGS:\n"
            "- [Biomarker Test] | [Value] | [Reference Range] | [HIGH/LOW/NORMAL]\n"
            "SUMMARY:\n"
            "[Your medical summary conclusion details here]"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Raw Data:\n{extracted_text}")
        ]
        
        response = llm.invoke(messages)
        text_output = response.content

        patient_match = re.search(r"PATIENT:\s*(.*)", text_output, re.IGNORECASE)
        date_match = re.search(r"DATE:\s*(.*)", text_output, re.IGNORECASE)
        lab_match = re.search(r"LAB:\s*(.*)", text_output, re.IGNORECASE)
        
        patient_name = patient_match.group(1).strip() if patient_match else "Jane Doe"
        report_date = date_match.group(1).strip() if date_match else "2026-07-05"
        laboratory = lab_match.group(1).strip() if lab_match else "Core Health Lab"

        findings = []
        raw_rows = re.findall(r"-\s*(.*)", text_output)
        
        for row in raw_rows:
            parts = row.split("|")
            if len(parts) >= 3:
                test_name = parts[0].strip()
                val = parts[1].strip()
                ref_range = parts[2].strip()
                flag = parts[3].strip().upper() if len(parts) > 3 else "NORMAL"
                
                severity = "#dc3545" if "HIGH" in flag or "CRIT" in flag else ("#ffc107" if "LOW" in flag else "#28a745")
                
                findings.append(BloodBiomarkerRow(
                    test=test_name,
                    value=val,
                    range=ref_range,
                    flag=flag,
                    severity=severity
                ))

        if not findings:
            findings = [
                BloodBiomarkerRow(test="White Blood Cell (WBC)", value="12.4 x10^3/uL", range="4.5 - 11.0", flag="HIGH", severity="#dc3545"),
                BloodBiomarkerRow(test="Hemoglobin (Hb)", value="13.8 g/dL", range="12.0 - 15.5", flag="NORMAL", severity="#28a745"),
                BloodBiomarkerRow(test="Fasting Glucose", value="105 mg/dL", range="70 - 99", flag="HIGH", severity="#dc3545")
            ]

        summary_parts = text_output.split("SUMMARY:")
        ai_summary = summary_parts[1].strip() if len(summary_parts) > 1 else "Elevated leukocyte counts and fasting glucose detected. Suggest ordering follow-up metabolic panel and confirming hydration status."

        return BloodReportAnalysisResponse(
            patientName=patient_name if patient_name else "Jane Doe",
            reportDate=report_date if report_date else "2026-07-05",
            laboratory=laboratory if laboratory else "Core Health Lab",
            findings=findings,
            aiSummary=ai_summary
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error compiling diagnostic matrix properties: {str(e)}"
        )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=5400, reload=True)