import os
import asyncio
import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional
from functools import lru_cache

from dotenv import load_dotenv        # <--- new
load_dotenv()                         # <--- load .env before getenv

import dateparser
from pydantic import BaseModel, Field

# update deprecated imports → langchain_openai
# replace langchain_community import with the official package
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings   # <--- updated import
from langchain_community.vectorstores import Chroma
from langchain.schema import SystemMessage, HumanMessage

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import PydanticOutputParser
from tenacity import retry, wait_exponential, stop_after_attempt, before_sleep_log
from httpx import RequestError as APIConnectionError
from tenacity import retry, retry_if_exception_type

logger = logging.getLogger(__name__)

# ——————————————————————————————————————————————
# 0. CONSTANTS & CONFIG
# ——————————————————————————————————————————————
DEFAULT_STRING: Optional[str] = None
DEFAULT_LIST: List[Any]    = []
CHUNK_SIZE     = 6000
CHUNK_OVERLAP  = 600
WEIGHTS        = {"budget_bonus": 30, "kpi_bonus": 30}

# LangChain splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],
)

# now that .env is loaded, pick up your API key
# remove temperature (unsupported) from embeddings
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-ada-002",
    openai_api_key= os.getenv("LLMS_API_KEY"),
    base_url= os.getenv("LLMS_API_URL"),
)

# ——————————————————————————————————————————————
# 1. ENUMS
# ——————————————————————————————————————————————
class AgreementType(str, Enum):
    Declaration           = "Declaration"
    MoU                   = "MoU"
    Roadmap               = "Roadmap"
    MinisterialCommunique = "Ministerial Communiqué"
    InvestmentProgramme   = "Investment Programme"
    LegalTreaty           = "Legal Treaty"
    Unknown               = "Unknown"

class Bindingness(str, Enum):
    Non       = "Non-binding"
    Political = "Politically-binding"
    Legal     = "Legally-binding"

class Review(str, Enum):
    None_     = "None"
    Annual    = "Annual"
    Biennial  = "Biennial"
    Triennial = "Triennial"
    Quarterly = "Quarterly"

class Coverage(str, Enum):
    Bilateral    = "Bilateral"
    SubRegional  = "Sub-regional"
    Regional     = "Regional"
    Multilateral = "Multilateral"
    Global       = "Global"
    Unknown      = "Unknown"

# ——————————————————————————————————————————————
# 2. DATA MODELS
# ——————————————————————————————————————————————
class CommitmentWithClass(BaseModel):
    text                 : str
    commitment_class     : str
    implementation_status: Optional[str] = None

class KPI(BaseModel):
    kpi         : str
    target_value: float
    unit        : str

class ExtraMetadata(BaseModel):
    lead_country_iso       : Optional[str]                 = None
    agreement_type         : Optional[str]                 = None
    legal_bindingness      : Optional[str]                 = None
    coverage_scope         : Optional[str]                 = None
    eu_policy_alignment    : List[str]                     = Field(default_factory=list)
    review_schedule        : Optional[str]                 = None
    monitoring_body        : Optional[str]                 = None
    budget_amount_eur      : Optional[float]               = None
    financing_instrument   : Optional[str]                 = None
    financing_source       : Optional[str]                 = None
    beneficiary_group_raw  : List[str]                     = Field(default_factory=list)
    beneficiary_group      : List[Dict[str, str]]          = Field(default_factory=list)
    start_date             : Optional[str]                 = None
    end_date               : Optional[str]                 = None
    commitment_details     : List[CommitmentWithClass]     = Field(default_factory=list)
    kpi_list               : List[KPI]                     = Field(default_factory=list)
    implementation_degree_pct: Optional[float]              = None
    actionability_score    : Optional[float]               = None
    country_list_iso       : List[str]                     = Field(default_factory=list)
    sdg_alignment          : List[str]                     = Field(default_factory=list)

# ——————————————————————————————————————————————
# 3. OUTPUT MODELS
# ——————————————————————————————————————————————
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

class SDGAlignmentOutput(BaseModel):
    sdg_alignment: List[str] = []

class CommitmentsOutput(BaseModel):
    commitments: List[CommitmentWithClass] = []

class BudgetInfoOutput(BaseModel):
    budget_amount_eur    : Optional[float] = None
    financing_instrument : Optional[str]   = None

class TimelineOutput(BaseModel):
    start_date: Optional[str] = None
    end_date  : Optional[str] = None

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

# ——————————————————————————————————————————————
# 4. SEMANTIC CHUNK INDEXING & SELECTION
# ——————————————————————————————————————————————
@lru_cache(maxsize=64)
def _get_vectorstore(text: str) -> Chroma:
    chunks = splitter.split_text(text)
    return Chroma.from_texts(chunks, embeddings)

async def _semantic_select(text: str, query: str, k: int = 3) -> str:
    vs   = _get_vectorstore(text)
    docs = vs.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)

# ——————————————————————————————————————————————
# 5. RETRY & INVOKE HELPERS
# ——————————————————————————————————————————————
def _retry():
    return retry(
        reraise=True,
        wait=wait_exponential(multiplier=1.5, min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

@_retry()
def _invoke_sync(prompt, llm, parser, **kwargs):
    chain = prompt | llm | parser
    return chain.invoke(kwargs)

def _invoke(prompt, llm, parser, **kwargs):
    try:
        return _invoke_sync(prompt, llm, parser, **kwargs)
    except Exception as e:
        logger.error("Extraction error %s → %s", parser.pydantic_object.__class__.__name__, e)
        return None

# ——————————————————————————————————————————————
# 6. GENERIC ASYNC EXTRACTOR with retry on connection errors
# ——————————————————————————————————————————————
# retry up to 3 times on APIConnectionError with exponential backoff
@retry(
    reraise=True,
    retry=retry_if_exception_type(APIConnectionError),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
)
async def _agenerate_with_retry(llm, messages):
    return await llm.agenerate([messages])

async def _generic_extract(
    llm,                    # BaseChatModel, e.g. ChatOpenAI
    parser_model: BaseModel,
    system_msg: str,
    user_msg: str,
    text: str,
    query: Optional[str] = None,
):
    parser = PydanticOutputParser(pydantic_object=parser_model)
    chunk  = await _semantic_select(text, query or user_msg)

    # Construct explicit messages
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"{user_msg}\n{parser.get_format_instructions()}\nText: {chunk}")
    ]

    try:
        # call agenerate with built-in retry for connection errors
        ai_result = await _agenerate_with_retry(llm, messages)
        reply     = ai_result.generations[0][0].message.content
        return parser.parse(reply)
    except Exception as e:
        logger.error(f"Error in _generic_extract for {parser_model.__name__}: {e}")
        return None

# ——————————————————————————————————————————————
# 7. REFACTORED EXTRACTORS
# ——————————————————————————————————————————————
async def extract_lead_country(text: str, llm) -> Optional[str]:
    sys  = "You are a geopolitical analyst. Return only ISO-3 country code."
    usr  = "Identify the host or presiding country. Output ISO-3.\nText: {text}"
    res  = await _generic_extract(llm, LeadCountryOutput, sys, usr, text, "host country")
    return res.lead_country_iso if res else None

async def extract_agreement_type(text: str, llm) -> Optional[str]:
    valid = str([e.value for e in AgreementType])
    sys   = "Classify the document's legal instrument."
    usr   = "Return ONE label from " + valid + ". If unsure → 'Unknown'.\nText: {text}"
    res   = await _generic_extract(llm, AgreementTypeOutput, sys, usr, text, "type of agreement")
    return res.agreement_type if res else None

async def extract_legal_bindingness(text: str, llm) -> Optional[str]:
    valid = str([e.value for e in Bindingness])
    sys   = "You are an international-law expert."
    usr   = "Classify bindingness as one of " + valid + ".\nText: {text}"
    res   = await _generic_extract(llm, LegalBindingnessOutput, sys, usr, text, "bindingness")
    return res.legal_bindingness if res else None

async def extract_review_schedule(text: str, llm) -> Optional[str]:
    valid = str([e.value for e in Review])
    sys   = "Extract document review frequency."
    usr   = "Return one of " + valid + ".\nText: {text}"
    res   = await _generic_extract(llm, ReviewScheduleOutput, sys, usr, text, "review schedule")
    return res.review_schedule if res else None

async def extract_coverage_scope(text: str, llm) -> Optional[str]:
    valid = str([e.value for e in Coverage])
    sys   = "Determine geographic coverage scope."
    usr   = "Return one of " + valid + ".\nText: {text}"
    res   = await _generic_extract(llm, CoverageScopeOutput, sys, usr, text, "coverage scope")
    return res.coverage_scope if res else None

async def extract_eu_policy_alignment(text: str, llm) -> List[str]:
    sys = "Extract references to EU policies."
    usr = """Return an array from:
["Global Gateway","NDICI-Global Europe","Digital Decade","Horizon Europe","EU Cyber Strategy"].
If none → [] c.
Text: {text}"""
    res = await _generic_extract(llm, EUPolicyAlignmentOutput, sys, usr, text, "EU policies")
    return res.eu_policy_alignment if res else []

async def extract_sdg_alignment(text: str, llm) -> List[str]:
    sys = "Find explicit SDG mentions in text."
    usr = """Look for patterns like "SDG 9", "ODS 17", "Goal 3". Return JSON array.
Text: {text}"""
    res = await _generic_extract(llm, SDGAlignmentOutput, sys, usr, text, "SDG mentions")
    return res.sdg_alignment if res else []

async def extract_timeline(text: str, llm) -> Dict[str, str]:
    sys = "Extract start/end dates in ISO-8601."
    usr = """Return {{"start_date":"YYYY-MM-DD","end_date":"YYYY-MM-DD"}}. If only year → YYYY-01-01. If missing → null.
Text: {text}"""
    res = await _generic_extract(llm, TimelineOutput, sys, usr, text, "timeline")
    sd  = dateparser.parse(res.start_date) if res and res.start_date else None
    ed  = dateparser.parse(res.end_date)   if res and res.end_date   else None
    return {
        "start_date": sd.date().isoformat() if sd else None,
        "end_date"  : ed.date().isoformat() if ed else None,
    }

async def extract_kpis(text: str, llm) -> List[KPI]:
    sys = "Extract up to 5 qualitative indicators."
    usr = """Provide kpi and unit="qualitative". Return JSON.
Text: {text}"""
    res = await _generic_extract(llm, KPIListOutput, sys, usr, text, "qualitative indicators")
    return res.kpi_list if res else []

async def extract_budget_info(text: str, llm) -> Dict[str, Any]:
    sys = "Extract total budget and financing instrument."
    usr = """Return JSON {{budget_amount_eur, financing_instrument}}. Convert to EUR. Sum multiple. If none → 0.0, "Unspecified".
Text: {text}"""
    res = await _generic_extract(llm, BudgetInfoOutput, sys, usr, text, "budget")
    amt = res.budget_amount_eur if res and res.budget_amount_eur else 0.0
    inst = res.financing_instrument if res and res.financing_instrument else "Unspecified"
    return {"budget_amount_eur": amt, "financing_instrument": inst}

async def extract_country_list(text: str, llm) -> List[str]:
    sys = "List ISO-3 codes of all countries mentioned, exclude lead country."
    usr = "Return sorted unique array max 50.\nText: {text}"
    res = await _generic_extract(llm, CountryListOutput, sys, usr, text, "country list")
    return res.country_list_iso if res else []

# ——————————————————————————————————————————————
# 8. MAIN PARALLEL ENTRY
# ——————————————————————————————————————————————
async def _process_document_async(text: str, title: str, commitments: List[str], llm) -> ExtraMetadata:
    preamble = f"{title}\n{text[:2000]}"
    (
        lead_country,
        agr_type,
        binding,
        coverage,
        review,
        eu_align,
        sdg_align,
        timeline,
        kpis,
        budget_info,
        country_list,
    ) = await asyncio.gather(
        extract_lead_country(preamble, llm),
        extract_agreement_type(preamble, llm),
        extract_legal_bindingness(text, llm),
        extract_coverage_scope(text, llm),
        extract_review_schedule(text, llm),
        extract_eu_policy_alignment(text, llm),
        extract_sdg_alignment(text, llm),
        extract_timeline(text, llm),
        extract_kpis(text, llm),
        extract_budget_info(text, llm),
        extract_country_list(text, llm),
    )
    # commitments analysis & metrics
    # (you can add analyze_commitments & normalization here similarly)
    implementation_pct = 0.0
    actionability     = 0.0

    return ExtraMetadata(
        lead_country_iso        = lead_country,
        agreement_type          = agr_type,
        legal_bindingness       = binding,
        coverage_scope          = coverage,
        review_schedule         = review,
        eu_policy_alignment     = eu_align,
        sdg_alignment           = sdg_align,
        start_date              = timeline["start_date"],
        end_date                = timeline["end_date"],
        kpi_list                = kpis,
        budget_amount_eur       = budget_info["budget_amount_eur"],
        financing_instrument    = budget_info["financing_instrument"],
        country_list_iso        = country_list,
        implementation_degree_pct = implementation_pct,
        actionability_score     = actionability,
    )

def process_document_for_extra_metadata(text: str, title: str, commitments: List[str], llm) -> ExtraMetadata:
    try:
        return asyncio.run(_process_document_async(text, title, commitments, llm))
    except Exception as e:
        logger.error("Failed processing document: %s", e)
        return ExtraMetadata()

def enrich_report_with_metadata(report_json: Dict[str, Any], text: str, llm) -> Dict[str, Any]:
    extra = process_document_for_extra_metadata(text, report_json.get("title", ""), report_json.get("commitments", []), llm)
    report_json["extra_metadata"] = extra.dict()
    return report_json