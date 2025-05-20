import json
import logging
import re
from typing import Dict, List, Any, Optional, Union

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List  # ensure List is imported

from src.tags import BENEFICIARIES_TAXONOMY  # add this import at the top if not already present

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1.  DATA MODELS
# -----------------------------------------------------------------------------

class CommitmentWithClass(BaseModel):
    """Model for a commitment with its classification and status."""
    text: str = Field(description="The commitment text")
    commitment_class: str = Field(description="Commitment classification – one of Declarative/Programmatic/Financed/Implemented")
    implementation_status: Optional[str] = Field(
        description="Implementation status – one of Not-started/Ongoing/Completed/Delayed/Cancelled",
        default=None,
    )


class KPI(BaseModel):
    """Key‑Performance Indicator with target value."""
    kpi: str = Field(description="KPI description")
    target_value: float = Field(description="Target value as number")
    unit: str = Field(description="Unit, incl. reference year if present")


class ExtraMetadata(BaseModel):
    """Principal strategic metadata captured for each agreement/event."""
    # --- core identification
    lead_country_iso: Optional[str] = Field(description="ISO‑3 country code of leading country", default=None)
    agreement_type: Optional[str] = Field(description="Type of agreement/document", default=None)
    legal_bindingness: Optional[str] = Field(description="Legal bindingness level", default=None)
    coverage_scope: Optional[str] = Field(description="Geographic scope of document", default=None)

    # --- alignment & governance
    eu_policy_alignment: List[str] = Field(description="EU policies referenced", default_factory=list)
    review_schedule: Optional[str] = Field(description="Periodicity of review", default=None)
    monitoring_body: Optional[str] = Field(description="Body/committee in charge of monitoring", default=None)

    # --- financial & beneficiaries
    budget_amount_eur: Optional[float] = Field(description="Total budget committed in EUR", default=None)
    financing_instrument: Optional[str] = Field(description="Type of financing mechanism", default=None)
    financing_source: Optional[str] = Field(description="Origin of the funds – EU budget / Member‑State / IFI / Private / Mixed", default=None)
    beneficiary_group_raw:  List[str]                    = Field(..., default_factory=list)
    beneficiary_group:      List[Dict[str, str]]        = Field(
        description="List of normalized beneficiary objects: {category, label}", 
        default_factory=list
    )

    # --- timeline
    start_date: Optional[str] = Field(description="Start date (ISO‑8601)", default=None)
    end_date: Optional[str] = Field(description="End date (ISO‑8601)", default=None)

    # --- commitments & KPIs
    commitment_details: List[CommitmentWithClass] = Field(description="Detailed commitment analysis", default_factory=list)
    kpi_list: List[KPI] = Field(description="List of quantifiable KPIs", default_factory=list)

    # --- derived metrics
    implementation_degree_pct: Optional[float] = Field(description="Average implementation degree 0‑100", default=None)
    actionability_score: Optional[float] = Field(description="Composite score of readiness 0‑100", default=None)

    # --- reach
    country_list_iso: List[str] = Field(description="All countries mentioned (ISO-3)", default_factory=list)
    budget_detected: bool = Field(description="Flag if a budget was detected", default=True)
    # ─── new SDG alignment field ────────────────────────────────────────────
    sdg_alignment: List[str] = Field(
        description="Detected SDG mentions (e.g., ['ODS 9','ODS 17'])",
        default_factory=list
    )
# -----------------------------------------------------------------------------
# 2.  GENERIC OUTPUT MODELS FOR PARSERS (1 attribute each)
# -----------------------------------------------------------------------------

class LeadCountryOutput(BaseModel):
    lead_country_iso: Optional[str] = None

class AgreementTypeOutput(BaseModel):
    agreement_type: Optional[str] = None

class LegalBindingnessOutput(BaseModel):
    legal_bindingness: Optional[str] = None

class ReviewScheduleOutput(BaseModel):
    review_schedule: Optional[str] = None

class EUPolicyAlignmentOutput(BaseModel):
    eu_policy_alignment: List[str] = []

# ─── new SDG alignment parser model ───────────────────────────────────────
class SDGAlignmentOutput(BaseModel):
    sdg_alignment: List[str] = []

class CommitmentsOutput(BaseModel):
    commitments: List[CommitmentWithClass] = []

class BudgetInfoOutput(BaseModel):
    budget_amount_eur: Optional[float] = None
    financing_instrument: Optional[str] = None

class ResponsibleEntityOutput(BaseModel):
    responsible_entity: Optional[str] = None

class TimelineOutput(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class KPIListOutput(BaseModel):
    kpi_list: List[KPI] = []

class CoverageScopeOutput(BaseModel):
    coverage_scope: Optional[str] = None

class MonitoringBodyOutput(BaseModel):
    monitoring_body: Optional[str] = None

class FinancingSourceOutput(BaseModel):
    financing_source: Optional[str] = None

class BeneficiaryGroupOutput(BaseModel):
    beneficiary_group_raw: List[str] = []

class CountryListOutput(BaseModel):
    country_list_iso: List[str] = []

class NormalizedBeneficiaryGroupOutput(BaseModel):
    normalized_beneficiary_group: List[Dict[str, str]] = []

# -----------------------------------------------------------------------------
# 3.  EXTRACTION HELPERS (LLM‑based)
# -----------------------------------------------------------------------------

def _invoke(prompt: PromptTemplate, llm, parser, **kwargs):
    """Utility to run the chain and trap errors centrally."""
    chain = prompt | llm | parser
    try:
        return chain.invoke(kwargs)
    except Exception as exc:
        logger.error("Extraction error for %s – %s", parser.pydantic_object.__class__.__name__, str(exc))
        return None

# -----------------------------------------------------------------------------
# 3.1  CORE FIELD EXTRACTORS (UPDATED PROMPTS)
# -----------------------------------------------------------------------------

def extract_lead_country(text: str, llm) -> Optional[str]:
    parser = PydanticOutputParser(pydantic_object=LeadCountryOutput)
    prompt = PromptTemplate(
        template="""
System: You are an expert in multilingual legal‑political analysis. Return only an ISO‑3 oficial country code.
User: Identify the host country or the country presiding over this agreement. If several appear, choose the one labelled *chair*, *host* or *president* in the preamble.
Text: {text}
{format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:5000])
    return result.lead_country_iso if result else None


def extract_agreement_type(title_and_preamble: str, llm) -> Optional[str]:
    parser = PydanticOutputParser(pydantic_object=AgreementTypeOutput)
    prompt = PromptTemplate(
        template="""
System: Classify the type of legal instrument.
User: Return ONE label from ["Declaration","MoU","Roadmap","Ministerial Communiqué","Investment Programme","Legal Treaty"]. If unsure → "Unknown".
Text: {text}
{format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=title_and_preamble)
    return result.agreement_type if result else None


def extract_legal_bindingness(full_text: str, llm) -> Optional[str]:
    parser = PydanticOutputParser(pydantic_object=LegalBindingnessOutput)
    prompt = PromptTemplate(
        template="""
System: You are an international‑law expert.
User: Using modal verbs, classify bindingness as "Non-binding", "Politically-binding", or "Legally-binding".
Text: {text}
{format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=full_text[:12000])
    return result.legal_bindingness if result else None


def extract_review_schedule(text: str, llm) -> Optional[str]:
    parser = PydanticOutputParser(pydantic_object=ReviewScheduleOutput)
    prompt = PromptTemplate(
        template="""
System: Extract the document’s review frequency.
User: Return one of ["Annual","Biennial","Triennial","Quarterly","None"].
      Map phrases like "every two years" or "mid-term review" to "Biennial".
Text: {text}
{format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:8000])
    return (result.review_schedule if result else None) or "None"


def extract_eu_policy_alignment(text: str, llm) -> List[str]:
    """
    Extract references to EU policies mentioned in the text.
    Returns a list of EU policy names that are explicitly mentioned.
    """
    parser = PydanticOutputParser(pydantic_object=EUPolicyAlignmentOutput)
    prompt = PromptTemplate(
        template="""
System: Extract references to EU policies.
User: Return an array of EU policies detected from this fixed list:
["Global Gateway","NDICI-Global Europe","Digital Decade","Horizon Europe","EU Cyber Strategy"]

If none are found, return an empty array.
Text: {text}
{format_instructions}
""",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:12000])
    return result.eu_policy_alignment if result else []

# -----------------------------------------------------------------------------
# 3.2  COMMITMENTS & IMPLEMENTATION
# -----------------------------------------------------------------------------

def analyze_commitments(commitments: List[str], llm) -> List[CommitmentWithClass]:
    if not commitments:
        return []
    parser = PydanticOutputParser(pydantic_object=CommitmentsOutput)
    prompt = PromptTemplate(
        template="""
System: Classify commitments.
User: For each commitment in the JSON list below, return:
  * commitment_class → "Declarative","Programmatic","Financed","Implemented"
  * implementation_status → "Not-started","Ongoing","Completed","Delayed","Cancelled"
Commitments JSON: {commitments}
{format_instructions}
        """,
        input_variables=["commitments"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, commitments=json.dumps(commitments))
    if result:
        return result.commitments
    # fallback – unclassified
    return [CommitmentWithClass(text=c, commitment_class="Unclassified") for c in commitments]


# -----------------------------------------------------------------------------
# 3.3  FINANCIAL EXTRACTION
# -----------------------------------------------------------------------------

def extract_budget_info(financial_text: str, llm) -> Dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=BudgetInfoOutput)
    prompt = PromptTemplate(
        template="""
System: Extract total budget and financing instrument.
User: Always return JSON with keys `budget_amount_eur` (float) and `financing_instrument`.
      • Convert amounts in euros, including expressions like 'million euro' or 'billion euro'.
      • For amounts in USD, convert using 1 USD = 1.1 EUR.
      • If multiple monetary figures exist, SUM them.
      • Few-shot example: "an investment of €35 million through a blended-finance facility" should return 35000000 and "Blended Finance".
      • If no amount is found, return 0.0 and "Unspecified".
Possible instruments: ["Grant", "Loan", "Blended Finance", "Guarantee", "Equity", "Unspecified"].
Text: {text}
{format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=financial_text)
    return {
        "budget_amount_eur": (result.budget_amount_eur if result else 0.0),
        "financing_instrument": (result.financing_instrument if result else "Unspecified"),
    }


def extract_financing_source(fin_text: str, llm) -> Optional[str]:
    parser = PydanticOutputParser(pydantic_object=FinancingSourceOutput)
    prompt = PromptTemplate(
        template="""
System: Identify the origin of funds.
User: Return one of ["EU budget","Member-State","IFI","Private","Mixed","Unknown"].
Look for phrases like "funded by", "co‑financed by", "contributed by".
Text: {text}
{format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=fin_text[:6000])
    return result.financing_source if result else None

# -----------------------------------------------------------------------------
# 3.4  GOVERNANCE / BENEFICIARIES / COUNTRY LIST / MONITORING
# -----------------------------------------------------------------------------

def extract_monitoring_body(text: str, llm) -> Optional[str]:
    parser = PydanticOutputParser(pydantic_object=MonitoringBodyOutput)
    prompt = PromptTemplate(
        template="""
System: Extract the name of the committee or body responsible for monitoring.
User: If not explicit, return null.
Text: {text}
{format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:8000])
    return result.monitoring_body if result and result.monitoring_body else "Unspecified"


def extract_beneficiary_group(text: str, llm) -> List[str]:
    parser = PydanticOutputParser(pydantic_object=BeneficiaryGroupOutput)
    prompt = PromptTemplate(
        template="""
System: Detect explicit beneficiary groups (e.g. SMEs, women, rural communities).
User: Return an array with up to 5 distinct groups; use nouns, singular.
Text: {text}
{format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:8000])
    return result.beneficiary_group_raw if result else []


def extract_country_list(text: str, llm) -> List[str]:
    parser = PydanticOutputParser(pydantic_object=CountryListOutput)
    prompt = PromptTemplate(
        template="""
System: List all countries explicitly mentioned in the document.
User: Return an array of ISO‑3 oficial codes, max 50, sorted alphabetically,no regions, no duplicates, exclude lead country.
Text: {text}
{format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:10000])
    return result.country_list_iso if result else []

# -----------------------------------------------------------------------------
# 3.5  TIMELINE, KPIs, COVERAGE (prompts mostly unchanged but stricter format)
# -----------------------------------------------------------------------------

def extract_responsible_entity(text: str, llm) -> Optional[str]:
    parser = PydanticOutputParser(pydantic_object=ResponsibleEntityOutput)
    prompt = PromptTemplate(
        template="""
System: Extract the OFFICIAL entity primarily responsible for implementation (one entity).
User: Look for phrases "will lead", "shall coordinate", "is responsible".
Text: {text}
{format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:6000])
    return result.responsible_entity if result else None


def extract_timeline(text: str, llm) -> Dict[str, str]:
    parser = PydanticOutputParser(pydantic_object=TimelineOutput)
    prompt = PromptTemplate(
        template="""
System: Extract start and end dates in ISO‑8601.
User: Return {{"start_date":"YYYY-MM-DD","end_date":"YYYY-MM-DD"}}.
If only year is present, use YYYY-01-01. If missing → null.
Text: {text}
{format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:10000])
    return {
        "start_date": (result.start_date if result else None),
        "end_date": (result.end_date if result else None),
    }


def extract_kpis(text: str, llm) -> List[KPI]:
    parser = PydanticOutputParser(pydantic_object=KPIListOutput)
    prompt = PromptTemplate(
        template="""
        System: Extract up to 5 quantifiable KPIs.
        User: For each KPI, return kpi, target_value (number) and unit.
        The output MUST be a valid JSON object containing a list of KPIs, conforming to the provided schema.
        Do NOT include any comments or explanatory text within the JSON output.
        If a target value is not found, use 0.
        Text: {text}
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:10000])
    return result.kpi_list if result else []


def extract_coverage_scope(text: str, llm) -> Optional[str]:
    parser = PydanticOutputParser(pydantic_object=CoverageScopeOutput)
    prompt = PromptTemplate(
        template="""
System: Determine the geographic coverage scope.
User: Return one of ["Bilateral","Sub-regional","Regional","Multilateral","Global","Unknown"].
Text: {text}
{format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:6000])
    return result.coverage_scope if result else None

def extract_sdg_alignment(text: str, llm) -> List[str]:
    """Detect mentions of Sustainable Development Goals (SDGs)."""
    parser = PydanticOutputParser(pydantic_object=SDGAlignmentOutput)
    prompt = PromptTemplate(
        template="""
System: Extract explicit references to UN Sustainable Development Goals (SDGs).
User: Identify all mentions of SDGs in the text. Look for:
- Direct references like "SDG 7" or "Sustainable Development Goal 13"
- Goal descriptions such as "Zero Hunger" (SDG 2) or "Climate Action" (SDG 13)
- Policy alignments that explicitly mention SDGs

Return a JSON array of SDG codes in the format ["SDG 1","SDG 2",...]. 
Only include goals 1-17 that are explicitly mentioned or clearly referenced.
If no SDGs are mentioned, return an empty array.

Text: {text}
{format_instructions}
""",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:10000])
    return result.sdg_alignment if result else []

# -----------------------------------------------------------------------------
# 4.  DERIVED METRICS – IMPLEMENTATION DEGREE & ACTIONABILITY
# -----------------------------------------------------------------------------

_STATUS_SCORE_MAP = {
    "Completed": 100,
    "Ongoing": 50,
    "Delayed": 25,
    "Not-started": 0,
    "Cancelled": 0,
}

def _compute_implementation_degree(commitments: List[CommitmentWithClass]) -> float:
    if not commitments:
        return 0.0
    scores = [_STATUS_SCORE_MAP.get(c.implementation_status or "Not-started", 0) for c in commitments]
    pct = sum(scores) / len(scores)
    return round(pct, 1)


def _compute_actionability(impl_pct: float, budget_eur: Optional[float], kpis: List[KPI]) -> float:
    has_budget = 1 if (budget_eur or 0) > 0 else 0
    has_kpi = 1 if kpis else 0
    raw = impl_pct + (has_budget * 30) + (has_kpi * 30)
    return round(min(raw / 1.6, 100), 2)

# -----------------------------------------------------------------------------
# 5.  MAIN ENTRY – full metadata extraction
# -----------------------------------------------------------------------------

def process_document_for_extra_metadata(text: str, title: str, commitments: List[str], llm) -> ExtraMetadata:
    """Extract enriched metadata from a document."""
    logger.info("Starting extraction of extra metadata")
    title_and_preamble = f"{title}\n{text[:2000]}"

    # ---- core fields
    lead_country = extract_lead_country(title_and_preamble, llm)
    agreement_type = extract_agreement_type(title_and_preamble, llm)
    legal_bindingness = extract_legal_bindingness(text, llm)
    coverage_scope = extract_coverage_scope(text, llm)

    # ---- alignment & governance
    review_schedule = extract_review_schedule(text, llm)
    eu_policy_alignment = extract_eu_policy_alignment(text, llm)
    monitoring_body = extract_monitoring_body(text, llm)

    # ---- commitments
    commitment_details = analyze_commitments(commitments, llm)

    # ---- finance
    budget_info = extract_budget_info(text, llm)
    financing_source = extract_financing_source(text, llm)

    # ---- timeline & KPIs
    timeline = extract_timeline(text, llm)
    kpis = extract_kpis(text, llm)

    # ---- beneficiaries & reach
    beneficiary_group_raw = extract_beneficiary_group(text, llm)
    beneficiary_group = normalize_beneficiary_group(beneficiary_group_raw, llm)
    country_list_iso = extract_country_list(text, llm)
    
    # ---- derived metrics
    implementation_pct = _compute_implementation_degree(commitment_details)
    actionability = _compute_actionability(implementation_pct, budget_info["budget_amount_eur"], kpis)
    
    # Set budget_detected flag based on extracted budget amount
    budget_detected = False if (budget_info["budget_amount_eur"] == 0.0) else True
    
    # ─── extract SDG alignment ────────────────────────────────
    sdg_alignment       = extract_sdg_alignment(text, llm)

    metadata = ExtraMetadata(
        lead_country_iso=lead_country,
        agreement_type=agreement_type,
        legal_bindingness=legal_bindingness,
        coverage_scope=coverage_scope,
        review_schedule=review_schedule,
        eu_policy_alignment=eu_policy_alignment,
        monitoring_body=monitoring_body,
        budget_amount_eur=budget_info["budget_amount_eur"],
        financing_instrument=budget_info["financing_instrument"],
        financing_source=financing_source,
        beneficiary_group_raw=beneficiary_group_raw,
        beneficiary_group=beneficiary_group,
        start_date=timeline["start_date"],
        end_date=timeline["end_date"],
        commitment_details=commitment_details,
        kpi_list=kpis,
        implementation_degree_pct=implementation_pct,
        actionability_score=actionability,
        country_list_iso=country_list_iso,
        budget_detected=budget_detected,
        # ─── include new SDG field ─────────────────────────────
        sdg_alignment=sdg_alignment
    )

    logger.info("Completed extraction of extra metadata")
    return metadata

# -----------------------------------------------------------------------------
# 6.  PUBLIC INTEGRATION FUNCTION
# -----------------------------------------------------------------------------

def enrich_report_with_metadata(report_json: Dict[str, Any], text: str, llm) -> Dict[str, Any]:
    """Enrich an existing report JSON (already containing title & commitments) with the extra metadata."""
    title = report_json.get("title", "")
    commitments = report_json.get("commitments", [])

    extra_metadata = process_document_for_extra_metadata(text, title, commitments, llm)
    report_json["extra_metadata"] = extra_metadata.dict()
    return report_json

def normalize_beneficiary_group(beneficiary_raw: List[str], llm) -> List[Dict[str, str]]:
    """
    Normalize the raw beneficiary groups using a fixed taxonomy.
    Returns a list of objects with keys:
      - category: the top‐level taxonomy bucket
      - label: the matched sub‐label
    """
    if not beneficiary_raw:
        return []
    prompt = PromptTemplate(
        template="""
System: Normalize beneficiary group terms.
User: You have this taxonomy (category → labels):
{taxonomy}

For each item in the JSON array `beneficiary_group_raw`:
 • find the label it matches under the taxonomy,
 • return an object {{ "category": top_level_category, "label": matched_label }}.
If it does not match any label, return {{ "category": "Other", "label": "Other" }}.

Input JSON: {beneficiary_group_raw}
Output must be JSON with key "normalized_beneficiary_group".
        """,
        input_variables=["beneficiary_group_raw"],
        partial_variables={"taxonomy": json.dumps(BENEFICIARIES_TAXONOMY, ensure_ascii=False)},
    )
    norm_parser = PydanticOutputParser(pydantic_object=NormalizedBeneficiaryGroupOutput)
    result = _invoke(prompt, llm, norm_parser, beneficiary_group_raw=json.dumps(beneficiary_raw))
    return result.normalized_beneficiary_group if result else []
