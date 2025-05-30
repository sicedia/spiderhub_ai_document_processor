import json
import logging
import re
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

from src.tags import BENEFICIARIES_TAXONOMY, SDG_TAXONOMY
from src.normalize_empty_values import normalize_empty_values  # Import the utility

logger = logging.getLogger(__name__)


# ——— Constantes de valores por defecto ————————————————————————————
DEFAULT_STRING: Optional[str] = None
DEFAULT_LIST: List[Any] = []
DEFAULT_DICT: Dict[str, Any] = {}

# Removed _or_default - replaced with normalize_empty_values

# -----------------------------------------------------------------------------
# 1.  DATA MODELS
# -----------------------------------------------------------------------------

class CommitmentWithClass(BaseModel):
    """Model for a commitment with its classification and status."""
    text: str = Field(description="The commitment text")
    commitment_class: str = Field(description="Commitment classification – one of Declarative/Programmatic/Financed/Implemented")


class KPI(BaseModel):
    """Key‑Performance Indicator with target value."""
    kpi: str = Field(description="KPI description")
    target_value: float = Field(description="Target value as number")
    unit: str = Field(description="Unit, incl. reference year if present")


class ExtraData(BaseModel):
    """Principal strategic extra_data captured for each agreement/event."""
    # --- core identification
    lead_country_iso: Optional[str] = Field(description="ISO-3 country code of leading country", default=None)
    agreement_type: List[str] = Field(
        description="Type(s) of agreement/document, one per input segment", 
        default_factory=list
    )
    legal_bindingness: Optional[str] = Field(description="Legal bindingness level", default=None)
    coverage_scope: Optional[str] = Field(description="Geographic scope of document", default=None)

    # --- alignment & governance
    eu_policy_alignment: List[str] = Field(description="EU policies referenced", default_factory=list)
    review_schedule: Optional[str] = Field(description="Periodicity of review", default=None)

    # beneficiaries
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

    # --- reach
    country_list_iso: List[str] = Field(description="All countries mentioned (ISO-3)", default_factory=list)
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

class ResponsibleEntityOutput(BaseModel):
    responsible_entity: Optional[str] = None

class TimelineOutput(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class KPIListOutput(BaseModel):
    kpi_list: List[KPI] = []

class CoverageScopeOutput(BaseModel):
    coverage_scope: Optional[str] = None

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
        result = chain.invoke(kwargs)
        return result
    except Exception as exc:
        logger.error("Extraction error for %s – %s", parser.pydantic_object.__class__.__name__, str(exc))
        return None

# -----------------------------------------------------------------------------
# 3.1  CORE FIELD EXTRACTORS (UPDATED WITH NORMALIZATION)
# -----------------------------------------------------------------------------

def extract_lead_country(text: str, llm) -> Optional[str]:
    parser = PydanticOutputParser(pydantic_object=LeadCountryOutput)
    prompt = PromptTemplate(
        template="""
System: You are an expert in multilingual legal‑political analysis. Return only an ISO‑3 oficial country code.
If no clear lead country is found, return 'null'.
User: Identify the host country or the country presiding over this agreement. If several appear, choose the one labelled *chair*, *host* or *president* in the preamble.
Text: {text}
{format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:5000])
    raw_value = result.lead_country_iso if result else None
    return normalize_empty_values(raw_value)


def extract_agreement_type(full_text: str, llm) -> List[str]:
    parser = PydanticOutputParser(pydantic_object=AgreementTypeOutput)
    # corregir backreference para capturar cada START/END
    segments = re.findall(
        r"=== START (.*?) ===\n(.*?)=== END \1 ===",
        full_text,
        re.DOTALL
    )

    labels: List[str] = []
    if segments:
        for filename, segment in segments:
            prompt = PromptTemplate(
                template="""
                System: You are an expert in legal-political analysis.
                Identify the document type for the text between START/END for "{filename}".
                Common types include: "Declaration", "MoU", "Roadmap", "Ministerial Communiqué", "Investment Programme", "Legal Treaty"
                But you're not limited to these - return the most accurate document type based on the content and structure.
                If no clear type can be determined, return 'null'.
                Text:
                {segment}

                {format_instructions}
                """,
                input_variables=["filename", "segment"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
            result = _invoke(prompt, llm, parser, filename=filename, segment=segment)
            raw_value = result.agreement_type if result else None
            normalized = normalize_empty_values(raw_value)
            if normalized:
                labels.append(normalized)
        return labels if labels else []

    # fallback único
    prompt = PromptTemplate(
        template="""
            System: Classify the type of legal instrument or agreement.
            Common examples include ["Declaration","MoU","Roadmap","Ministerial Communiqué","Investment Programme","Legal Treaty"]
            but you're not limited to these options. Return the most accurate document type based on the content and structure.
            If no clear type can be determined, return 'null'.
            Text:
            {text}

            {format_instructions}
            """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=full_text)
    if result:
        normalized = normalize_empty_values(result.agreement_type)
        return [normalized] if normalized else []
    return []


def extract_legal_bindingness(full_text: str, llm) -> Optional[str]:
    parser = PydanticOutputParser(pydantic_object=LegalBindingnessOutput)
    prompt = PromptTemplate(
        template="""
            System: You are an international‑law expert.
            User: Using modal verbs, classify bindingness as "Non-binding", "Politically-binding", or "Legally-binding".
            If bindingness cannot be determined, return 'null'.
            Text: {text}
            {format_instructions}
                    """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=full_text[:12000])
    raw_value = result.legal_bindingness if result else None
    return normalize_empty_values(raw_value)


def extract_review_schedule(text: str, llm) -> Optional[str]:
    parser = PydanticOutputParser(pydantic_object=ReviewScheduleOutput)
    prompt = PromptTemplate(
        template="""
            System: Extract the document's review frequency.
            User: Return one of ["Annual","Biennial","Triennial","Quarterly","None"].
                Map phrases like "every two years" or "mid-term review" to "Biennial".
                If no review schedule is found, return 'null'.
            Text: {text}
            {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:8000])
    raw_value = result.review_schedule if result else None
    return normalize_empty_values(raw_value)


def extract_eu_policy_alignment(text: str, llm) -> List[str]:
    """Extract references to EU policies mentioned in the text."""
    parser = PydanticOutputParser(pydantic_object=EUPolicyAlignmentOutput)
    prompt = PromptTemplate(
        template="""
            System: Extract references to EU policies.
            User: Return an array of EU policies detected from this fixed list:
            ["Global Gateway","NDICI-Global Europe","Digital Decade","Horizon Europe","EU Cyber Strategy"]

            If none are found, return empty array: []
            Text: {text}
            {format_instructions}
            """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:12000])
    raw_value = result.eu_policy_alignment if result else []
    normalized = normalize_empty_values(raw_value)
    return normalized if isinstance(normalized, list) else []

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
            Commitments JSON: {commitments}
            
            If no commitments are found, return empty array: []
            {format_instructions}
        """,
        input_variables=["commitments"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, commitments=json.dumps(commitments))
    if result and result.commitments:
        return result.commitments
    # fallback – unclassified
    return [CommitmentWithClass(text=c, commitment_class="Unclassified") for c in commitments]


# -----------------------------------------------------------------------------
# 3.4  GOVERNANCE / BENEFICIARIES / COUNTRY LIST / MONITORING
# -----------------------------------------------------------------------------

def extract_beneficiary_group(text: str, llm) -> List[str]:
    parser = PydanticOutputParser(pydantic_object=BeneficiaryGroupOutput)
    prompt = PromptTemplate(
        template="""
            System: Detect explicit beneficiary groups (e.g. SMEs, women, rural communities).
            User: Return an array with up to 5 distinct groups; use nouns, singular.
            If no beneficiary groups are found, return empty array: []
            Text: {text}
            {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:8000])
    raw_value = result.beneficiary_group_raw if result else []
    normalized = normalize_empty_values(raw_value)
    return normalized if isinstance(normalized, list) else []


def extract_country_list(text: str, llm) -> List[str]:
    parser = PydanticOutputParser(pydantic_object=CountryListOutput)
    prompt = PromptTemplate(
        template="""
            System: List all countries explicitly mentioned in the document.
            User: Return an array of ISO‑3 oficial codes, max 50, sorted alphabetically, no regions, no duplicates, exclude lead country.
            If no countries are found, return empty array: []
            Text: {text}
            {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:10000])
    raw_value = result.country_list_iso if result else []
    normalized = normalize_empty_values(raw_value)
    return normalized if isinstance(normalized, list) else []

# -----------------------------------------------------------------------------
# 3.5  TIMELINE, KPIs, COVERAGE
# -----------------------------------------------------------------------------

def extract_responsible_entity(text: str, llm) -> Optional[str]:
    parser = PydanticOutputParser(pydantic_object=ResponsibleEntityOutput)
    prompt = PromptTemplate(
        template="""
            System: Extract the OFFICIAL entity primarily responsible for implementation (one entity).
            User: Look for phrases "will lead", "shall coordinate", "is responsible".
            If no responsible entity is found, return 'null'.
            Text: {text}
            {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:6000])
    raw_value = result.responsible_entity if result else None
    return normalize_empty_values(raw_value)


def extract_timeline(text: str, llm) -> Dict[str, Optional[str]]:
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
    
    start_date = normalize_empty_values(result.start_date if result else None)
    end_date = normalize_empty_values(result.end_date if result else None)
    
    return {
        "start_date": start_date,
        "end_date": end_date,
    }


def extract_kpis(text: str, llm) -> List[KPI]:
    parser = PydanticOutputParser(pydantic_object=KPIListOutput)
    prompt = PromptTemplate(
        template="""
        System: Extract up to 5 KPIs, including both quantitative and qualitative metrics.
        User: For each KPI, return:
        - kpi: The KPI description
        - target_value: Numeric value (use 1.0 for qualitative KPIs without specific numbers)
        - unit: Unit of measurement (use "qualitative" for non-numeric KPIs)
        
        The output MUST be a valid JSON object containing a list of KPIs, conforming to the provided schema.
        Do NOT include any comments or explanatory text within the JSON output.
        If no KPIs are found, return empty array: []
        Look for both:
        - Measurable metrics with numbers
        - Qualitative goals that can serve as success indicators
        
        Text: {text}
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:10000])
    raw_value = result.kpi_list if result else []
    normalized = normalize_empty_values(raw_value)
    return normalized if isinstance(normalized, list) else []


def extract_coverage_scope(text: str, llm) -> Optional[str]:
    parser = PydanticOutputParser(pydantic_object=CoverageScopeOutput)
    prompt = PromptTemplate(
        template="""
            System: Determine the geographic coverage scope.
            User: Return one of ["Bilateral","Sub-regional","Regional","Multilateral","Global","Unknown"].
            If scope cannot be determined, return 'null'.
            Text: {text}
            {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = _invoke(prompt, llm, parser, text=text[:6000])
    raw_value = result.coverage_scope if result else None
    return normalize_empty_values(raw_value)

def extract_sdg_alignment(text: str, llm) -> List[str]:
    """Detect mentions of Sustainable Development Goals (SDGs)."""
    parser = PydanticOutputParser(pydantic_object=SDGAlignmentOutput)

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a sustainability expert. Your task is to map text to the United Nations Sustainable Development Goals (SDGs)."
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Given the following text, identify any **explicit** or **implicit** references to the SDGs.  
            Use the official SDG list below for both exact and paraphrased matches (e.g. "zero hunger" → SDG 2):

            {sdg_list_json}

            {text}

            Return ONLY valid JSON matching exactly this schema:

            {format_instructions}
                - Do NOT include any extra keys or commentary.
                - If no SDG is found, return an empty array: []
                        """
        ),
        ])

    try:
        result = _invoke(
            chat_prompt,
            llm,
            parser,
            sdg_list_json=json.dumps(SDG_TAXONOMY, ensure_ascii=False),
            text=text[:10000],
            format_instructions=parser.get_format_instructions()
        )
        raw_value = result.sdg_alignment if result else []
        normalized = normalize_empty_values(raw_value)
        return normalized if isinstance(normalized, list) else []
    except Exception as e:
        logger.error(f"Error extracting SDG alignment: {str(e)}")
        return []

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
            If it does not match any label, create a new category with format {{ "category": "<matched_category>", "label": "<matched_label>" }}, choose the better matched_category and matched_label.

            If no beneficiaries can be normalized, return empty array: []
            Input JSON: {beneficiary_group_raw}
            Output must be JSON with key "normalized_beneficiary_group".
        """,
        input_variables=["beneficiary_group_raw"],
        partial_variables={"taxonomy": json.dumps(BENEFICIARIES_TAXONOMY, ensure_ascii=False)},
    )
    norm_parser = PydanticOutputParser(pydantic_object=NormalizedBeneficiaryGroupOutput)
    result = _invoke(prompt, llm, norm_parser, beneficiary_group_raw=json.dumps(beneficiary_raw))
    
    raw_value = result.normalized_beneficiary_group if result else []
    normalized = normalize_empty_values(raw_value)
    return normalized if isinstance(normalized, list) else []

# -----------------------------------------------------------------------------
# 5.  MAIN ENTRY – full extra_data extraction
# -----------------------------------------------------------------------------

def process_document_for_extra_data(text: str, title: str, commitments: List[str], llm) -> ExtraData:
    logger.info("Starting extraction of extra extra_data")

    # ---- core fields
    logger.info("Extracting lead_country...")
    lead_country = extract_lead_country(text, llm)

    logger.info("Extracting agreement_type...")
    # Usar el text completo para capturar todos los bloques START/END
    agreement_type = extract_agreement_type(text, llm)
    logger.debug("agreement_type -> %s", agreement_type)

    logger.info("Extracting legal_bindingness...")
    legal_bindingness = extract_legal_bindingness(text, llm)
    logger.debug("legal_bindingness -> %s", legal_bindingness)

    logger.info("Extracting coverage_scope...")
    coverage_scope = extract_coverage_scope(text, llm)
    logger.debug("coverage_scope -> %s", coverage_scope)

    # ---- alignment & governance
    logger.info("Extracting review_schedule...")
    review_schedule = extract_review_schedule(text, llm)
    logger.debug("review_schedule -> %s", review_schedule)

    logger.info("Extracting eu_policy_alignment...")
    eu_policy_alignment = extract_eu_policy_alignment(text, llm)
    logger.debug("eu_policy_alignment -> %s", eu_policy_alignment)

    # ---- commitments
    logger.info("Analyzing commitments...")
    commitment_details = analyze_commitments(commitments, llm)
    logger.debug("commitment_details -> %s", commitment_details)

    # ---- timeline & KPIs
    logger.info("Extracting timeline...")
    timeline = extract_timeline(text, llm)
    logger.debug("timeline -> %s", timeline)

    logger.info("Extracting kpis...")
    kpis = extract_kpis(text, llm)
    logger.debug("kpi_list -> %s", kpis)

    # ---- beneficiaries & reach
    logger.info("Extracting beneficiary_group_raw...")
    beneficiary_group_raw = extract_beneficiary_group(text, llm)
    logger.debug("beneficiary_group_raw -> %s", beneficiary_group_raw)

    logger.info("Normalizing beneficiary_group...")
    beneficiary_group = normalize_beneficiary_group(beneficiary_group_raw, llm)
    logger.debug("beneficiary_group -> %s", beneficiary_group)

    logger.info("Extracting country_list_iso...")
    country_list_iso = extract_country_list(text, llm)
    logger.debug("country_list_iso -> %s", country_list_iso)

    # ─── extract SDG alignment ────────────────────────────────
    logger.info("Extracting sdg_alignment...")
    sdg_alignment = extract_sdg_alignment(text, llm)
    logger.debug("sdg_alignment -> %s", sdg_alignment)

    extra_data = ExtraData(
        lead_country_iso=lead_country,
        agreement_type=agreement_type,
        legal_bindingness=legal_bindingness,
        coverage_scope=coverage_scope,
        review_schedule=review_schedule,
        eu_policy_alignment=eu_policy_alignment,
        beneficiary_group_raw=beneficiary_group_raw,
        beneficiary_group=beneficiary_group,
        start_date=timeline["start_date"],
        end_date=timeline["end_date"],
        commitment_details=commitment_details,
        kpi_list=kpis,
        country_list_iso=country_list_iso,
        sdg_alignment=sdg_alignment
    )

    logger.info("Completed extraction of extra extra_data")
    logger.debug("Extracted extra_data: %s", extra_data.dict())
    return extra_data

# -----------------------------------------------------------------------------
# 6.  PUBLIC INTEGRATION FUNCTION
# -----------------------------------------------------------------------------

def enrich_report_with_extradata(report_json: Dict[str, Any], text: str, llm) -> Dict[str, Any]:
    """Enrich an existing report JSON (already containing title & commitments) with the extra extra_data."""
    title = report_json.get("title", "")
    commitments = report_json.get("commitments", [])

    extra_data = process_document_for_extra_data(text, title, commitments, llm)
    report_json["extra_data"] = extra_data.dict()
    return report_json

