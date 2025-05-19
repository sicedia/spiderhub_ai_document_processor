import logging
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

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

def extract_lead_country(text: str, llm) -> Optional[str]:
    """Extract the lead country ISO code from document text."""
    prompt = """
    System: Eres un experto en análisis jurídico-político multilingüe. Devuelve solo un ISO-3 country code.
    User: Identifica el país anfitrión o el país que preside este acuerdo. Si aparece más de uno, elige el que figure como chair, host o president en el preámbulo: {text}
    
    Piensa paso a paso pero no muestres tu razonamiento.
    
    Format: {{"lead_country_iso":"XXX"}}
    """
    
    try:
        response = llm.invoke(prompt.format(text=text[:5000]))  # Limit text length to avoid token limits
        # Parse response to extract ISO code
        import re
        import json
        match = re.search(r'{"lead_country_iso":"(\w{3})"}', response)
        if match:
            return match.group(1)
        
        # Try JSON parsing if regex fails
        try:
            parsed = json.loads(response)
            return parsed.get("lead_country_iso")
        except:
            logger.warning("Failed to parse lead country ISO response")
            return None
    except Exception as e:
        logger.error(f"Error extracting lead country: {str(e)}")
        return None

def extract_agreement_type(title_and_preamble: str, llm) -> Optional[str]:
    """Extract the type of agreement from the document title and preamble."""
    prompt = """
    System: Clasifica el tipo de instrumento.
    User: Devuelve UNA etiqueta de esta lista: ["Declaration","MoU","Roadmap","Ministerial Communiqué","Investment Programme","Legal Treaty"]. 
    Si no se puede inferir, responde "Unknown". Texto: {text}
    
    Piensa paso a paso pero no muestres tu razonamiento.
    
    Format: {{"agreement_type":"Declaration"}}
    """
    
    try:
        response = llm.invoke(prompt.format(text=title_and_preamble))
        import json
        try:
            parsed = json.loads(response)
            return parsed.get("agreement_type")
        except:
            # Fallback to regex
            import re
            match = re.search(r'"agreement_type"\s*:\s*"([^"]+)"', response)
            if match:
                return match.group(1)
        return None
    except Exception as e:
        logger.error(f"Error extracting agreement type: {str(e)}")
        return None

def extract_legal_bindingness(full_text: str, llm) -> Optional[str]:
    """Assess the legal bindingness of the document."""
    prompt = """
    System: Eres abogado internacional.
    User: Devuelve "Non-binding", "Politically-binding" o "Legally-binding" evaluando verbos ("shall", "commit" vs. "encourage"). Texto: {text}
    
    Piensa paso a paso pero no muestres tu razonamiento.
    
    Format: {{"legal_bindingness":"Non-binding"}}
    """
    
    try:
        # Use first 10000 chars to stay within token limits
        response = llm.invoke(prompt.format(text=full_text[:10000]))
        import re
        import json
        match = re.search(r'"legal_bindingness"\s*:\s*"([^"]+)"', response)
        if match:
            return match.group(1)
        
        try:
            parsed = json.loads(response)
            return parsed.get("legal_bindingness")
        except:
            pass
            
        return None
    except Exception as e:
        logger.error(f"Error extracting legal bindingness: {str(e)}")
        return None

# Add similar functions for each metadata field

def extract_review_schedule(text: str, llm) -> Optional[str]:
    """Extract the review schedule periodicity."""
    prompt = """
    System: Extrae periodicidad de revisión.
    User: Si se menciona "every two years" → "Biennial", "annual" → "Annual", etc. Si no se menciona → "None".
    Texto: {text}
    
    Piensa paso a paso pero no muestres tu razonamiento.
    
    Format: {{"review_schedule":"Annual"}}
    """
    
    try:
        response = llm.invoke(prompt.format(text=text[:8000]))
        import json
        try:
            parsed = json.loads(response)
            return parsed.get("review_schedule")
        except:
            import re
            match = re.search(r'"review_schedule"\s*:\s*"([^"]+)"', response)
            if match:
                return match.group(1)
        return "None"
    except Exception as e:
        logger.error(f"Error extracting review schedule: {str(e)}")
        return "None"

def extract_eu_policy_alignment(text: str, llm) -> List[str]:
    """Extract references to EU policies."""
    prompt = """
    System: Extrae referencias a políticas UE.
    User: Devuelve array con políticas detectadas de: ["Global Gateway","NDICI-Global Europe",
    "Digital Decade","Horizon Europe","EU Cyber Strategy"]. Texto: {text}
    
    Piensa paso a paso pero no muestres tu razonamiento.
    
    Format: {{"eu_policy_alignment":["Global Gateway","Digital Decade"]}}
    """
    
    try:
        response = llm.invoke(prompt.format(text=text[:10000]))
        import json
        try:
            parsed = json.loads(response)
            return parsed.get("eu_policy_alignment", [])
        except:
            # Fallback parsing
            import re
            match = re.search(r'"eu_policy_alignment"\s*:\s*(\[.*?\])', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
        return []
    except Exception as e:
        logger.error(f"Error extracting EU policy alignment: {str(e)}")
        return []

def analyze_commitments(commitments: List[str], llm) -> List[CommitmentWithClass]:
    """Analyze and classify commitments."""
    if not commitments:
        return []
    
    commitment_json = json.dumps(commitments)
    prompt = """
    System: Clasifica compromisos.
    User: Para cada compromiso en {commitments} devuelve una de: "Declarative", "Programmatic", 
    "Financed", "Implemented". Usa verbos guía (e.g., "launches" → Implemented).
    
    También clasifica el estado de implementación como: "Not-started","Ongoing","Completed","Delayed","Cancelled"
    
    Piensa paso a paso pero no muestres tu razonamiento.
    
    Format: [
      {{"text": "Commitment 1", "commitment_class": "Declarative", "implementation_status": "Not-started"}},
      {{"text": "Commitment 2", "commitment_class": "Implemented", "implementation_status": "Completed"}}
    ]
    """
    
    try:
        response = llm.invoke(prompt.format(commitments=commitment_json))
        import json
        try:
            parsed = json.loads(response)
            return [CommitmentWithClass(**item) for item in parsed]
        except:
            logger.warning("Failed to parse commitment classifications")
            return [CommitmentWithClass(text=text, commitment_class="Unclassified") 
                    for text in commitments]
    except Exception as e:
        logger.error(f"Error analyzing commitments: {str(e)}")
        return [CommitmentWithClass(text=text, commitment_class="Unclassified") 
                for text in commitments]

def extract_budget_info(financial_section: str, llm) -> Dict[str, Any]:
    """Extract budget amount and financing instrument."""
    prompt = """
    System: Extrae información financiera.
    User: Busca expresiones monetarias "€", "million euros/euro". Convierte a valor absoluto en EUR. 
    Si varias cifras, suma. También identifica el instrumento financiero de la lista: 
    ["Grant","Loan","Blended Finance","Guarantee","Equity"] o "Unspecified".
    
    Texto: {text}
    
    Piensa paso a paso pero no muestres tu razonamiento.
    
    Format: {{"budget_amount_eur": 1500000, "financing_instrument": "Grant"}}
    """
    
    try:
        response = llm.invoke(prompt.format(text=financial_section))
        import json
        try:
            return json.loads(response)
        except:
            logger.warning("Failed to parse budget information")
            return {"budget_amount_eur": None, "financing_instrument": "Unspecified"}
    except Exception as e:
        logger.error(f"Error extracting budget info: {str(e)}")
        return {"budget_amount_eur": None, "financing_instrument": "Unspecified"}

def extract_responsible_entity(implementation_paragraph: str, llm) -> Optional[str]:
    """Extract the entity responsible for implementation."""
    prompt = """
    System: Devuelve el nombre oficial del organismo que implementará el compromiso.
    User: Busca frases "will lead", "is responsible", "shall coordinate". Texto: {text}
    
    Piensa paso a paso pero no muestres tu razonamiento.
    
    Format: {{"responsible_entity":"European Commission"}}
    """
    
    try:
        response = llm.invoke(prompt.format(text=implementation_paragraph))
        import json
        try:
            parsed = json.loads(response)
            return parsed.get("responsible_entity")
        except:
            import re
            match = re.search(r'"responsible_entity"\s*:\s*"([^"]+)"', response)
            if match:
                return match.group(1)
        return None
    except Exception as e:
        logger.error(f"Error extracting responsible entity: {str(e)}")
        return None

def extract_timeline(timeline_section: str, llm) -> Dict[str, str]:
    """Extract start and end dates."""
    prompt = """
    System: Extrae fechas de inicio y fin.
    User: Devuelve fechas en formato ISO-8601 (YYYY-MM-DD). Si solo hay año, usa YYYY-01-01. Texto: {text}
    
    Piensa paso a paso pero no muestres tu razonamiento.
    
    Format: {{"start_date": "2023-01-01", "end_date": "2025-12-31"}}
    """
    
    try:
        response = llm.invoke(prompt.format(text=timeline_section))
        import json
        try:
            return json.loads(response)
        except:
            logger.warning("Failed to parse timeline information")
            return {"start_date": None, "end_date": None}
    except Exception as e:
        logger.error(f"Error extracting timeline: {str(e)}")
        return {"start_date": None, "end_date": None}

def extract_kpis(text: str, llm) -> List[KPI]:
    """Extract quantifiable KPIs."""
    prompt = """
    System: Extrae KPIs cuantificables.
    User: Extrae una lista máx. 5 KPIs cuantificables con su valor meta y unidad.
    
    Texto: {text}
    
    Piensa paso a paso pero no muestres tu razonamiento.
    
    Format: {{"kpi_list":[{{"kpi":"Internet access","target":"95% by 2025"}},{{"kpi":"Digital skills training","target":"10,000 people"}}]}}
    """
    
    try:
        response = llm.invoke(prompt.format(text=text[:10000]))
        import json
        try:
            parsed = json.loads(response)
            kpis = parsed.get("kpi_list", [])
            return [KPI(**item) for item in kpis]
        except:
            logger.warning("Failed to parse KPI information")
            return []
    except Exception as e:
        logger.error(f"Error extracting KPIs: {str(e)}")
        return []

def extract_coverage_scope(text: str, llm) -> Optional[str]:
    """Extract the geographic scope of the document."""
    prompt = """
    System: Determina el alcance geográfico.
    User: Devuelve "Bilateral","Sub-regional","Regional","Multilateral","Global". Basado en países mencionados.
    
    Ejemplos: 
    - "EU and Colombia cooperation" -> "Bilateral"
    - "Central American integration" -> "Sub-regional"
    - "Latin American initiative" -> "Regional"
    - "UN framework agreement" -> "Global"
    
    Texto: {text}
    
    Piensa paso a paso pero no muestres tu razonamiento.
    
    Format: {{"coverage_scope":"Bilateral"}}
    """
    
    try:
        response = llm.invoke(prompt.format(text=text[:5000]))
        import json
        try:
            parsed = json.loads(response)
            return parsed.get("coverage_scope")
        except:
            import re
            match = re.search(r'"coverage_scope"\s*:\s*"([^"]+)"', response)
            if match:
                return match.group(1)
        return None
    except Exception as e:
        logger.error(f"Error extracting coverage scope: {str(e)}")
        return None

def process_document_for_extra_metadata(text: str, title: str, commitments: List[str], llm) -> ExtraMetadata:
    """Process a document to extract all extra metadata."""
    import json
    
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