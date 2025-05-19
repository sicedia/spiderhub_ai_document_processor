import json
import logging
import re
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

logger = logging.getLogger(__name__)

class CommitmentWithClass(BaseModel):
    """Model for a commitment with its classification."""
    text: str = Field(description="The commitment text")
    commitment_class: str = Field(description="The commitment classification")
    implementation_status: Optional[str] = Field(description="Implementation status", default=None)

class KPI(BaseModel):
    """Key Performance Indicator with target value."""
    kpi: str = Field(description="KPI description")
    target: str = Field(description="Target value with units")

class ExtraMetadata(BaseModel):
    """Additional metadata for strategic document analysis."""
    lead_country_iso: Optional[str] = Field(description="ISO-3 country code of leading country", default=None)
    agreement_type: Optional[str] = Field(description="Type of agreement/document", default=None)
    legal_bindingness: Optional[str] = Field(description="Legal bindingness level", default=None)
    review_schedule: Optional[str] = Field(description="Periodicity of review", default=None)
    eu_policy_alignment: List[str] = Field(description="EU policies referenced", default_factory=list)
    commitment_details: List[CommitmentWithClass] = Field(description="Detailed commitment analysis", default_factory=list)
    budget_amount_eur: Optional[float] = Field(description="Total budget in EUR", default=None)
    financing_instrument: Optional[str] = Field(description="Type of financing mechanism", default=None)
    responsible_entity: Optional[str] = Field(description="Entity responsible for implementation", default=None)
    start_date: Optional[str] = Field(description="Start date (ISO-8601)", default=None)
    end_date: Optional[str] = Field(description="End date (ISO-8601)", default=None)
    kpi_list: List[KPI] = Field(description="List of quantifiable KPIs", default_factory=list)
    coverage_scope: Optional[str] = Field(description="Geographic scope of document", default=None)

# Output models for each extraction function
class LeadCountryOutput(BaseModel):
    lead_country_iso: Optional[str] = Field(description="ISO-3 country code of leading country", default=None)

class AgreementTypeOutput(BaseModel):
    agreement_type: Optional[str] = Field(description="Type of agreement/document", default=None)

class LegalBindingnessOutput(BaseModel):
    legal_bindingness: Optional[str] = Field(description="Legal bindingness level", default=None)

class ReviewScheduleOutput(BaseModel):
    review_schedule: Optional[str] = Field(description="Periodicity of review", default=None)

class EUPolicyAlignmentOutput(BaseModel):
    eu_policy_alignment: List[str] = Field(description="EU policies referenced", default_factory=list)

class CommitmentsOutput(BaseModel):
    commitments: List[CommitmentWithClass] = Field(description="List of commitments with classifications", default_factory=list)

class BudgetInfoOutput(BaseModel):
    budget_amount_eur: Optional[float] = Field(description="Total budget in EUR", default=None)
    financing_instrument: Optional[str] = Field(description="Type of financing mechanism", default=None)

class ResponsibleEntityOutput(BaseModel):
    responsible_entity: Optional[str] = Field(description="Entity responsible for implementation", default=None)

class TimelineOutput(BaseModel):
    start_date: Optional[str] = Field(description="Start date (ISO-8601)", default=None)
    end_date: Optional[str] = Field(description="End date (ISO-8601)", default=None)

class KPIListOutput(BaseModel):
    kpi_list: List[KPI] = Field(description="List of quantifiable KPIs", default_factory=list)

class CoverageScopeOutput(BaseModel):
    coverage_scope: Optional[str] = Field(description="Geographic scope of document", default=None)

def extract_lead_country(text: str, llm) -> Optional[str]:
    """Extract the lead country ISO code from document text."""
    parser = PydanticOutputParser(pydantic_object=LeadCountryOutput)
    
    prompt = PromptTemplate(
        template="""
        System: Eres un experto en análisis jurídico-político multilingüe. Devuelve solo un ISO-3 country code.
        
        User: Identifica el país anfitrión o el país que preside este acuerdo. Si aparece más de uno, 
        elige el que figure como chair, host o president en el preámbulo: {text}
        
        Piensa paso a paso pero no muestres tu razonamiento.
        
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({"text": text[:5000]})  # Limit text length
        return result.lead_country_iso
    except Exception as e:
        logger.error(f"Error extracting lead country: {str(e)}")
        return None

def extract_agreement_type(title_and_preamble: str, llm) -> Optional[str]:
    """Extract the type of agreement from the document title and preamble."""
    parser = PydanticOutputParser(pydantic_object=AgreementTypeOutput)
    
    prompt = PromptTemplate(
        template="""
        System: Clasifica el tipo de instrumento.
        
        User: Devuelve UNA etiqueta de esta lista: ["Declaration","MoU","Roadmap","Ministerial Communiqué",
        "Investment Programme","Legal Treaty"]. 
        Si no se puede inferir, responde "Unknown". Texto: {text}
        
        Piensa paso a paso pero no muestres tu razonamiento.
        
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({"text": title_and_preamble})
        return result.agreement_type
    except Exception as e:
        logger.error(f"Error extracting agreement type: {str(e)}")
        return None

def extract_legal_bindingness(full_text: str, llm) -> Optional[str]:
    """Assess the legal bindingness of the document."""
    parser = PydanticOutputParser(pydantic_object=LegalBindingnessOutput)
    
    prompt = PromptTemplate(
        template="""
        System: Eres abogado internacional.
        
        User: Devuelve "Non-binding", "Politically-binding" o "Legally-binding" evaluando verbos 
        ("shall", "commit" vs. "encourage"). Texto: {text}
        
        Piensa paso a paso pero no muestres tu razonamiento.
        
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({"text": full_text[:10000]})  # Use first 10000 chars
        return result.legal_bindingness
    except Exception as e:
        logger.error(f"Error extracting legal bindingness: {str(e)}")
        return None

def extract_review_schedule(text: str, llm) -> Optional[str]:
    """Extract the review schedule periodicity."""
    parser = PydanticOutputParser(pydantic_object=ReviewScheduleOutput)
    
    prompt = PromptTemplate(
        template="""
        System: Extrae periodicidad de revisión.
        
        User: Si se menciona "every two years" → "Biennial", "annual" → "Annual", etc. 
        Si no se menciona → "None". Texto: {text}
        
        Piensa paso a paso pero no muestres tu razonamiento.
        
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({"text": text[:8000]})
        return result.review_schedule or "None"
    except Exception as e:
        logger.error(f"Error extracting review schedule: {str(e)}")
        return "None"

def extract_eu_policy_alignment(text: str, llm) -> List[str]:
    """Extract references to EU policies."""
    parser = PydanticOutputParser(pydantic_object=EUPolicyAlignmentOutput)
    
    prompt = PromptTemplate(
        template="""
        System: Extrae referencias a políticas UE.
        
        User: Devuelve array con políticas detectadas de: ["Global Gateway","NDICI-Global Europe",
        "Digital Decade","Horizon Europe","EU Cyber Strategy"]. Texto: {text}
        
        Piensa paso a paso pero no muestres tu razonamiento.
        
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({"text": text[:10000]})
        return result.eu_policy_alignment
    except Exception as e:
        logger.error(f"Error extracting EU policy alignment: {str(e)}")
        return []

def analyze_commitments(commitments: List[str], llm) -> List[CommitmentWithClass]:
    """Analyze and classify commitments."""
    if not commitments:
        return []
    
    parser = PydanticOutputParser(pydantic_object=CommitmentsOutput)
    commitment_json = json.dumps(commitments)
    
    prompt = PromptTemplate(
        template="""
        System: Clasifica compromisos.
        
        User: Para cada compromiso en {commitments} devuelve una de: "Declarative", "Programmatic", 
        "Financed", "Implemented". Usa verbos guía (e.g., "launches" → Implemented).
        
        También clasifica el estado de implementación como: "Not-started","Ongoing","Completed","Delayed","Cancelled"
        
        Piensa paso a paso pero no muestres tu razonamiento.
        
        {format_instructions}
        """,
        input_variables=["commitments"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({"commitments": commitment_json})
        return result.commitments
    except Exception as e:
        logger.error(f"Error analyzing commitments: {str(e)}")
        return [CommitmentWithClass(text=text, commitment_class="Unclassified") 
                for text in commitments]

def extract_budget_info(financial_section: str, llm) -> Dict[str, Any]:
    """Extract budget amount and financing instrument."""
    parser = PydanticOutputParser(pydantic_object=BudgetInfoOutput)
    
    prompt = PromptTemplate(
        template="""
        System: Extrae información financiera.
        
        User: Busca expresiones monetarias "€", "million euros/euro". Convierte a valor absoluto en EUR. 
        Si varias cifras, suma. También identifica el instrumento financiero de la lista: 
        ["Grant","Loan","Blended Finance","Guarantee","Equity"] o "Unspecified".
        
        Texto: {text}
        
        Piensa paso a paso pero no muestres tu razonamiento.
        
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({"text": financial_section})
        return {
            "budget_amount_eur": result.budget_amount_eur,
            "financing_instrument": result.financing_instrument
        }
    except Exception as e:
        logger.error(f"Error extracting budget info: {str(e)}")
        return {"budget_amount_eur": None, "financing_instrument": "Unspecified"}

def extract_responsible_entity(implementation_paragraph: str, llm) -> Optional[str]:
    """Extract the entity responsible for implementation."""
    parser = PydanticOutputParser(pydantic_object=ResponsibleEntityOutput)
    
    prompt = PromptTemplate(
        template="""
        System: Devuelve el nombre oficial del organismo que implementará el compromiso.
        
        User: Busca frases "will lead", "is responsible", "shall coordinate". Texto: {text}
        
        Piensa paso a paso pero no muestres tu razonamiento.
        
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({"text": implementation_paragraph})
        return result.responsible_entity
    except Exception as e:
        logger.error(f"Error extracting responsible entity: {str(e)}")
        return None

def extract_timeline(timeline_section: str, llm) -> Dict[str, str]:
    """Extract start and end dates."""
    parser = PydanticOutputParser(pydantic_object=TimelineOutput)
    
    prompt = PromptTemplate(
        template="""
        System: Extrae fechas de inicio y fin.
        
        User: Devuelve fechas en formato ISO-8601 (YYYY-MM-DD). Si solo hay año, usa YYYY-01-01. Texto: {text}
        
        Piensa paso a paso pero no muestres tu razonamiento.
        
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({"text": timeline_section})
        return {
            "start_date": result.start_date,
            "end_date": result.end_date
        }
    except Exception as e:
        logger.error(f"Error extracting timeline: {str(e)}")
        return {"start_date": None, "end_date": None}

def extract_kpis(text: str, llm) -> List[KPI]:
    """Extract quantifiable KPIs."""
    parser = PydanticOutputParser(pydantic_object=KPIListOutput)
    
    prompt = PromptTemplate(
        template="""
        System: Extrae KPIs cuantificables.
        
        User: Extrae una lista máx. 5 KPIs cuantificables con su valor meta y unidad.
        
        Texto: {text}
        
        Piensa paso a paso pero no muestres tu razonamiento.
        
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({"text": text[:10000]})
        return result.kpi_list
    except Exception as e:
        logger.error(f"Error extracting KPIs: {str(e)}")
        return []

def extract_coverage_scope(text: str, llm) -> Optional[str]:
    """Extract the geographic scope of the document."""
    parser = PydanticOutputParser(pydantic_object=CoverageScopeOutput)
    
    prompt = PromptTemplate(
        template="""
        System: Determina el alcance geográfico.
        
        User: Devuelve "Bilateral","Sub-regional","Regional","Multilateral","Global". 
        Basado en países mencionados.
        
        Ejemplos: 
        - "EU and Colombia cooperation" -> "Bilateral"
        - "Central American integration" -> "Sub-regional"
        - "Latin American initiative" -> "Regional"
        - "UN framework agreement" -> "Global"
        
        Texto: {text}
        
        Piensa paso a paso pero no muestres tu razonamiento.
        
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({"text": text[:5000]})
        return result.coverage_scope
    except Exception as e:
        logger.error(f"Error extracting coverage scope: {str(e)}")
        return None

def process_document_for_extra_metadata(text: str, title: str, commitments: List[str], llm) -> ExtraMetadata:
    """Process a document to extract all extra metadata."""
    logger.info("Starting extraction of extra metadata")
    
    # Prepare text sections
    title_and_preamble = title + "\n" + text[:2000]  # First 2000 chars likely contain preamble
    
    # Extract metadata fields
    lead_country = extract_lead_country(title_and_preamble, llm)
    agreement_type = extract_agreement_type(title_and_preamble, llm)
    legal_bindingness = extract_legal_bindingness(text, llm)
    review_schedule = extract_review_schedule(text, llm)
    eu_policy_alignment = extract_eu_policy_alignment(text, llm)
    
    # Process commitments
    commitment_details = analyze_commitments(commitments, llm)
    
    # Find potential financial section
    financial_section = text  # Ideally we would use semantic search to find financial paragraphs
    budget_info = extract_budget_info(financial_section, llm)
    
    # Extract implementation details
    implementation_paragraph = text  # Ideally we would use semantic search
    responsible_entity = extract_responsible_entity(implementation_paragraph, llm)
    
    # Extract timeline
    timeline = extract_timeline(text, llm)
    
    # Extract KPIs and scope
    kpis = extract_kpis(text, llm)
    coverage_scope = extract_coverage_scope(text, llm)
    
    # Create and return ExtraMetadata object
    metadata = ExtraMetadata(
        lead_country_iso=lead_country,
        agreement_type=agreement_type,
        legal_bindingness=legal_bindingness,
        review_schedule=review_schedule,
        eu_policy_alignment=eu_policy_alignment,
        commitment_details=commitment_details,
        budget_amount_eur=budget_info.get("budget_amount_eur"),
        financing_instrument=budget_info.get("financing_instrument"),
        responsible_entity=responsible_entity,
        start_date=timeline.get("start_date"),
        end_date=timeline.get("end_date"),
        kpi_list=kpis,
        coverage_scope=coverage_scope
    )
    
    logger.info(f"Completed extraction of extra metadata: {json.dumps(metadata.dict(), indent=2)}")
    return metadata

# Integration function to be called from report_generator.py
def enrich_report_with_metadata(report_json: Dict[str, Any], text: str, llm) -> Dict[str, Any]:
    """Enrich an existing report JSON with extra metadata."""
    title = report_json.get("title", "")
    commitments = report_json.get("commitments", [])
    
    # Process and extract metadata
    extra_metadata = process_document_for_extra_metadata(text, title, commitments, llm)
    
    # Add metadata to report
    report_json["extra_metadata"] = extra_metadata.dict()
    
    return report_json