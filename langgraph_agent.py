"""
langgraph_agent.py
-------------------
LangGraph-based Clinical Decision Support (CDS) agent with RAG.
"""

import os
import uuid
from typing import TypedDict, List, Dict

from langgraph.graph import StateGraph, END
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "cdss_assessments"
LLM_MODEL = "gpt-4o-mini"

# Read provider toggle configuration
USE_OPENAI = os.getenv("USE_OPENAI", "true").lower() == "true"

# ---------------------------------------------------------------------------
# 1. State schema
# ---------------------------------------------------------------------------
class CDSState(TypedDict, total=False):
    symptoms: List[str]
    vitals: Dict[str, float]

    vital_flags: List[str]
    similar_cases: List[str]        
    possible_conditions: List[str]  
    llm_reasoning: str              
    risk_level: str                 
    risk_score: int
    recommendation: str
    alerts: List[str]


# ---------------------------------------------------------------------------
# 2. Reference data
# ---------------------------------------------------------------------------
VITAL_RANGES = {
    "heart_rate":        (60, 100),
    "systolic_bp":       (90, 120),
    "diastolic_bp":      (60, 80),
    "temperature":       (97.0, 99.5),
    "spo2":              (95, 100),
    "respiratory_rate":  (12, 20),
}


# ---------------------------------------------------------------------------
# 3. Shared clients (Conditional Setup based on USE_OPENAI)
# ---------------------------------------------------------------------------
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

_vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=_embeddings,
    persist_directory=CHROMA_DIR,
)

if USE_OPENAI:
    from langchain_openai import ChatOpenAI
    _llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
else:
    from langchain_ollama import ChatOllama
    _llm = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "gemma3:1b"),
        temperature=0
    )


# ---------------------------------------------------------------------------
# 4. Nodes
# ---------------------------------------------------------------------------
def intake(state: CDSState) -> CDSState:
    state["symptoms"] = [s.strip().lower() for s in state.get("symptoms", [])]
    state["vitals"] = state.get("vitals", {})
    state["alerts"] = []
    return state


def analyze_vitals(state: CDSState) -> CDSState:
    flags = []
    for name, value in state["vitals"].items():
        if name not in VITAL_RANGES:
            continue
        low, high = VITAL_RANGES[name]
        if value < low:
            flags.append(f"LOW {name.replace('_', ' ')}: {value} (normal {low}-{high})")
        elif value > high:
            flags.append(f"HIGH {name.replace('_', ' ')}: {value} (normal {low}-{high})")
    state["vital_flags"] = flags
    return state


def retrieve_similar_cases(state: CDSState) -> CDSState:
    query = f"Symptoms: {', '.join(state['symptoms'])}. Vitals flags: {', '.join(state['vital_flags'])}"
    try:
        results = _vectorstore.similarity_search(query, k=3)
        state["similar_cases"] = [doc.page_content for doc in results]
    except Exception:
        state["similar_cases"] = []
    return state


def analyze_symptoms(state: CDSState) -> CDSState:
    context_block = (
        "\n".join(f"- {c}" for c in state["similar_cases"])
        if state["similar_cases"] else "No similar past cases found."
    )

    prompt = f"""You are a clinical decision support assistant. Given the patient
data below, list the most likely possible conditions (comma-separated, most
likely first, max 5) and one sentence of clinical reasoning.

Symptoms: {', '.join(state['symptoms']) or 'none reported'}
Abnormal vitals: {', '.join(state['vital_flags']) or 'none'}

Similar past cases on record:
{context_block}

Respond in this exact format:
CONDITIONS: condition1, condition2, condition3
REASONING: <one sentence>
"""
    response = _llm.invoke(prompt)
    text = response.content

    conditions: List[str] = []
    reasoning = ""
    for line in text.splitlines():
        if line.upper().startswith("CONDITIONS:"):
            conditions = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    state["possible_conditions"] = conditions or ["Unable to determine from model output"]
    state["llm_reasoning"] = reasoning or text.strip()
    return state


def assess_risk(state: CDSState) -> CDSState:
    score = 0
    score += len(state["vital_flags"]) * 2
    score += len(state["symptoms"])

    critical_symptoms = {"chest pain", "shortness of breath", "confusion"}
    if critical_symptoms.intersection(state["symptoms"]):
        score += 5

    critical_vitals = any(
        kw in flag for flag in state["vital_flags"]
        for kw in ("spo2", "heart rate", "systolic bp")
    )
    if critical_vitals:
        score += 4

    if score >= 10:
        level = "critical"
    elif score >= 6:
        level = "high"
    elif score >= 3:
        level = "moderate"
    else:
        level = "low"

    state["risk_score"] = score
    state["risk_level"] = level
    return state


def escalate(state: CDSState) -> CDSState:
    state["alerts"].append("URGENT: Immediate clinician review recommended.")
    state["recommendation"] = (
        f"Risk level: {state['risk_level'].upper()} (score {state['risk_score']}). "
        f"Vital concerns: {', '.join(state['vital_flags']) or 'none'}. "
        f"Possible conditions to rule out: {', '.join(state['possible_conditions'])}. "
        f"Reasoning: {state['llm_reasoning']} "
        "Recommend immediate physician evaluation, continuous monitoring, "
        "and consider emergency protocols."
    )
    return state


def recommend(state: CDSState) -> CDSState:
    state["recommendation"] = (
        f"Risk level: {state['risk_level'].upper()} (score {state['risk_score']}). "
        f"Vital concerns: {', '.join(state['vital_flags']) or 'none'}. "
        f"Possible conditions: {', '.join(state['possible_conditions'])}. "
        f"Reasoning: {state['llm_reasoning']} "
        "Recommend routine clinical assessment and standard monitoring."
    )
    return state


def store_to_vectordb(state: CDSState) -> CDSState:
    doc_text = (
        f"Symptoms: {', '.join(state['symptoms'])}. "
        f"Vital flags: {', '.join(state['vital_flags']) or 'none'}. "
        f"Risk: {state['risk_level']} (score {state['risk_score']}). "
        f"Conditions considered: {', '.join(state['possible_conditions'])}. "
        f"Recommendation: {state['recommendation']}"
    )
    _vectorstore.add_texts(
        texts=[doc_text],
        metadatas=[{
            "risk_level": state["risk_level"],
            "risk_score": state["risk_score"],
            "symptoms": ", ".join(state["symptoms"]),
        }],
        ids=[str(uuid.uuid4())],
    )
    return state


def route_by_risk(state: CDSState) -> str:
    return "escalate" if state["risk_level"] in ("high", "critical") else "recommend"


def build_graph():
    graph = StateGraph(CDSState)

    graph.add_node("intake", intake)
    graph.add_node("analyze_vitals", analyze_vitals)
    graph.add_node("retrieve_similar_cases", retrieve_similar_cases)
    graph.add_node("analyze_symptoms", analyze_symptoms)
    graph.add_node("assess_risk", assess_risk)
    graph.add_node("escalate", escalate)
    graph.add_node("recommend", recommend)
    graph.add_node("store_to_vectordb", store_to_vectordb)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "analyze_vitals")
    graph.add_edge("analyze_vitals", "retrieve_similar_cases")
    graph.add_edge("retrieve_similar_cases", "analyze_symptoms")
    graph.add_edge("analyze_symptoms", "assess_risk")
    graph.add_conditional_edges(
        "assess_risk",
        route_by_risk,
        {"escalate": "escalate", "recommend": "recommend"},
    )
    graph.add_edge("escalate", "store_to_vectordb")
    graph.add_edge("recommend", "store_to_vectordb")
    graph.add_edge("store_to_vectordb", END)

    return graph.compile()


cds_graph = build_graph()


def run_cdss_graph(symptoms: List[str], vitals: Dict[str, float]) -> CDSState:
    initial_state: CDSState = {
        "symptoms": symptoms,
        "vitals": vitals,
    }
    final_state = cds_graph.invoke(initial_state)
    return final_state