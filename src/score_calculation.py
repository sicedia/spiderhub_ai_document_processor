import json
import logging
from typing import Optional, Dict, Any, List
import json

from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class QualityScore(BaseModel):
    """Model for quality assessment scores."""
    faithfulness: int = Field(description="Faithfulness score 0-100", ge=0, le=100)
    consistency: int = Field(description="Internal consistency score 0-100", ge=0, le=100)
    completeness: int = Field(description="Information completeness score 0-100", ge=0, le=100)
    accuracy: int = Field(description="Factual accuracy score 0-100", ge=0, le=100)
    overall_score: int = Field(description="Overall quality score 0-100", ge=0, le=100)
    issues: List[str] = Field(description="List of identified issues", default_factory=list)

class SingleNumericScore(BaseModel):
    score: int = Field(description="The numeric score", ge=0, le=100)

def calculate_score(
    source_text: str, 
    generated_content: Dict[str, Any],
    llm: BaseLanguageModel
) -> Optional[int]:
    """
    Calculate comprehensive quality score for generated content using multiple LangChain evaluators.
    
    Args:
        source_text: Original document text
        generated_content: AI-generated structured content (including extra_data)
        llm: Language model instance
        
    Returns:
        Overall quality score (0-100) or None if evaluation fails
    """
    try:
        logger.info("Starting comprehensive quality evaluation...")
        
        # Convert generated content to readable format (including extra_data)
        generated_summary = _format_generated_content(generated_content)
        
        # 1. Faithfulness evaluation using LangChain
        faithfulness_score = _evaluate_faithfulness(source_text, generated_summary, llm)
        
        # 2. Consistency evaluation
        consistency_score = _evaluate_consistency(generated_content, llm)
        
        # 3. Completeness evaluation
        completeness_score = _evaluate_completeness(source_text, generated_content, llm)
        
        # 4. Accuracy evaluation
        accuracy_score = _evaluate_accuracy(source_text, generated_content, llm)
        
        # 5. Calculate weighted overall score
        overall_score = _calculate_weighted_score(
            faithfulness_score, consistency_score, 
            completeness_score, accuracy_score
        )
        
        logger.info(f"Quality scores - Faithfulness: {faithfulness_score}, "
                   f"Consistency: {consistency_score}, Completeness: {completeness_score}, "
                   f"Accuracy: {accuracy_score}, Overall: {overall_score}")
        
        return overall_score
        
    except Exception as e:
        logger.error(f"Error calculating comprehensive score: {e}")
        return 75  # Fallback score

def _format_generated_content(generated_content: Dict[str, Any]) -> str:
    """Convert generated content to readable summary text including extra_data."""
    summary_parts = []
    
    if title := generated_content.get("title"):
        summary_parts.append(f"Title: {title}")
    
    if exec_summary := generated_content.get("executive_summary"):
        summary_parts.append(f"Summary: {exec_summary}")
    
    if characteristics := generated_content.get("characteristics"):
        if isinstance(characteristics, list):
            summary_parts.append(f"Key characteristics: {'; '.join(characteristics)}")
    
    if themes := generated_content.get("themes"):
        theme_summary = []
        for theme, subthemes in themes.items():
            if isinstance(subthemes, list):
                theme_summary.append(f"{theme}: {', '.join(subthemes)}")
        if theme_summary:
            summary_parts.append(f"Themes: {'; '.join(theme_summary)}")
    
    if actors := generated_content.get("actors"):
        actor_summary = []
        for actor_type, actor_list in actors.items():
            if isinstance(actor_list, list):
                actor_summary.append(f"{actor_type}: {', '.join(actor_list)}")
        if actor_summary:
            summary_parts.append(f"Actors: {'; '.join(actor_summary)}")
    
    if commitments := generated_content.get("commitments"):
        if isinstance(commitments, list) and commitments:
            summary_parts.append(f"Commitments: {'; '.join(commitments[:3])}")  # First 3 commitments
    
    # Include extra_data in the summary
    if extra_data := generated_content.get("extra_data"):
        extra_summary = _format_extra_data(extra_data)
        if extra_summary:
            summary_parts.append(f"Strategic Data: {extra_summary}")
    
    return "\n".join(summary_parts)

def _format_extra_data(extra_data: Dict[str, Any]) -> str:
    """Format extra_data for evaluation."""
    extra_parts = []
    
    if lead_country := extra_data.get("lead_country_iso"):
        extra_parts.append(f"Lead Country: {lead_country}")
    
    if agreement_types := extra_data.get("agreement_type"):
        if isinstance(agreement_types, list) and agreement_types:
            extra_parts.append(f"Agreement Types: {', '.join(agreement_types)}")
    
    if legal_binding := extra_data.get("legal_bindingness"):
        extra_parts.append(f"Legal Bindingness: {legal_binding}")
    
    if coverage := extra_data.get("coverage_scope"):
        extra_parts.append(f"Coverage: {coverage}")
    
    if start_date := extra_data.get("start_date"):
        extra_parts.append(f"Start Date: {start_date}")
    
    if end_date := extra_data.get("end_date"):
        extra_parts.append(f"End Date: {end_date}")
    
    if eu_policies := extra_data.get("eu_policy_alignment"):
        if isinstance(eu_policies, list) and eu_policies:
            extra_parts.append(f"EU Policies: {', '.join(eu_policies)}")
    
    if sdg_alignment := extra_data.get("sdg_alignment"):
        if isinstance(sdg_alignment, list) and sdg_alignment:
            extra_parts.append(f"SDG Alignment: {', '.join(sdg_alignment)}")
    
    if countries := extra_data.get("country_list_iso"):
        if isinstance(countries, list) and countries:
            extra_parts.append(f"Countries: {', '.join(countries[:5])}")  # First 5 countries
    
    if kpis := extra_data.get("kpi_list"):
        if isinstance(kpis, list) and kpis:
            kpi_descriptions = [kpi.get("kpi", "") for kpi in kpis[:3]]  # First 3 KPIs
            if kpi_descriptions:
                extra_parts.append(f"KPIs: {'; '.join(kpi_descriptions)}")
    
    return "; ".join(extra_parts)

def _evaluate_faithfulness(source_text: str, generated_summary: str, llm: BaseLanguageModel) -> int:
    """Evaluate faithfulness using LangChain's CRITERIA evaluator with a custom criterion."""
    try:
        # define a custom "faithfulness" criterion
        custom_criteria = {
            "faithfulness": (
                "Does the generated summary accurately reflect the source text, "
                "without adding unsupported claims, contradicting, or omitting key info?"
            )
        }
        evaluator = load_evaluator(
            EvaluatorType.CRITERIA,
            criteria=custom_criteria,
            llm=llm
        )
        result = evaluator.evaluate_strings(
            input=source_text[:8000],
            prediction=generated_summary
        )
        # result may have a float "score" 0–1 or {"faithfulness": "Y"/"N"/"..."
        raw = result.get("score", None)
        if raw is None and isinstance(result, dict):
            v = str(result.get("faithfulness", "")).strip().upper()
            if v in {"Y", "YES"}:
                raw = 1.0
            elif v in {"N", "NO"}:
                raw = 0.0
            else:
                # try to parse embedded float
                import re
                m = re.search(r"([0](?:\.\d+)?|1(?:\.0+)?)", v)
                raw = float(m.group(1)) if m else 0.75
        raw = float(raw) if isinstance(raw, (int, float, str)) else 0.75
        return max(0, min(100, int(raw * 100)))
    except Exception as e:
        logger.error(f"Error in faithfulness evaluation: {e}")
        return 75

def _evaluate_consistency(generated_content: Dict[str, Any], llm: BaseLanguageModel) -> int:
    """Evaluate internal consistency of generated content including extra_data."""
    try:
        class SingleNumericScore(BaseModel):
            score: int = Field(description="The numeric score", ge=0, le=100)

        parser = PydanticOutputParser(pydantic_object=SingleNumericScore)
        
        prompt = PromptTemplate(
            template="""
            Evaluate the internal consistency of this generated content including all extracted data. 
            Check for contradictions, logical coherence, and alignment between different sections.
            Pay special attention to consistency between:
            - Main content and strategic extra data
            - Dates and timelines across sections
            - Actor information and country lists
            - Themes and commitment classifications
            - Agreement types and legal bindingness
            
            Generated Content:
            {content}
            
            Rate consistency from 0-100 where:
            - 90-100: Perfectly consistent, no contradictions across all data
            - 70-89: Mostly consistent with minor inconsistencies
            - 50-69: Some contradictions or unclear relationships
            - 30-49: Multiple inconsistencies across sections
            - 0-29: Major contradictions throughout
            
            {format_instructions}
            """,
            input_variables=["content"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt | llm | parser
        response_obj = chain.invoke({"content": json.dumps(generated_content, indent=2)})
        return response_obj.score
        
    except Exception as e:
        logger.error(f"Error in consistency evaluation: {e}")
        return 75

def _evaluate_completeness(source_text: str, generated_content: Dict[str, Any], llm: BaseLanguageModel) -> int:
    """Evaluate completeness of information extraction including extra_data."""
    try:
        parser = PydanticOutputParser(pydantic_object=SingleNumericScore)
        
        prompt = PromptTemplate(
            template="""
            You are an information completeness expert. Your ONLY task is to evaluate completeness and return a JSON score.

            IMPORTANT: You MUST respond with ONLY valid JSON in this exact format:
            {{"score": <number>}}

            Do NOT include any explanatory text, comments, or additional information outside the JSON.

            Task: Compare the source document with the extracted information to evaluate completeness.
            Consider both basic extraction (title, summary, themes, actors, commitments) 
            and strategic data (agreement types, legal bindingness, timelines, KPIs, country lists, SDG alignment).
            
            Source Document (first 6000 chars):
            {source_text}
            
            Extracted Information:
            {extracted_content}
            
            Rate completeness from 0-100 where:
            - 90-100: All key information captured across all categories
            - 70-89: Most important information captured in both basic and strategic data
            - 50-69: Adequate information captured but missing some strategic details
            - 30-49: Missing important details in multiple categories
            - 0-29: Significant information gaps in both basic and strategic data
            
            Focus on whether the extraction captures:
            - Main themes, actors, commitments, and key details
            - Agreement classification and legal framework
            - Timeline and implementation details
            - Strategic alignment (EU policies, SDGs)
            - Geographic and beneficiary scope

            {format_instructions}

            REMEMBER: Return ONLY the JSON object with the score. No additional text.
            """,
            input_variables=["source_text", "extracted_content"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt | llm | parser
        response_obj = chain.invoke({
            "source_text": source_text[:6000],
            "extracted_content": json.dumps(generated_content, indent=2)
        })
        
        return max(0, min(100, response_obj.score))
            
    except Exception as e:
        logger.error(f"Error in completeness evaluation: {e}")
        return 75

def _evaluate_accuracy(source_text: str, generated_content: Dict[str, Any], llm: BaseLanguageModel) -> int:
    """Evaluate factual accuracy of extracted information including extra_data."""
    # 1. Construyo factual_claims como antes…
    factual_claims = []
    if date := generated_content.get("date"):
        if not date.lower().startswith("no "):
            factual_claims.append(f"Date: {date}")
    if location := generated_content.get("principal_location"):
        if not location.lower().startswith("no "):
            factual_claims.append(f"Location: {location}")
    commits = generated_content.get("commitments", [])
    valid_commits = [c for c in commits if c and not c.lower().startswith("no specific")]
    factual_claims += [f"Commitment: {c}" for c in valid_commits[:3]]
    if extra := generated_content.get("extra_data", {}):
        if lead := extra.get("lead_country_iso"):
            factual_claims.append(f"Lead Country ISO: {lead}")
        if start := extra.get("start_date"):
            factual_claims.append(f"Start Date: {start}")
        if end := extra.get("end_date"):
            factual_claims.append(f"End Date: {end}")
        if types := extra.get("agreement_type", []):
            factual_claims.append(f"Agreement Types: {', '.join(types)}")
        if countries := extra.get("country_list_iso", []):
            factual_claims.append(f"Country List: {', '.join(countries[:5])}")
        if cov := extra.get("coverage_scope"):
            factual_claims.append(f"Coverage: {cov}")

    # 2. Si no hay claims, retorno fallback
    if not factual_claims:
        return 85

    # 3. Verificación claim-by-claim
    correct = 0
    total = len(factual_claims)
    for claim in factual_claims:
        # Definimos un pequeño Pydantic para parsear { "valid": true/false }
        class ClaimResult(BaseModel):
            valid: bool

        parser = PydanticOutputParser(pydantic_object=ClaimResult)
        prompt = PromptTemplate(
            template="""
    You are a fact-verification expert. Return ONLY a valid JSON with the key "valid": true if the claim is correct, or false if not.
            
    Source Document:
    {text}

    Claim to verify:
    {claim}

    {format_instructions}
    """,
            input_variables=["text", "claim"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | llm | parser
        try:
            resp = chain.invoke({
                "text": source_text,
                "claim": claim
            })
            if resp.valid:
                correct += 1
        except Exception:
            # en caso de fallo en este claim lo damos por incorrecto
            continue

    return int(100 * correct / total)

def _calculate_weighted_score(faithfulness: int, consistency: int, 
                            completeness: int, accuracy: int) -> int:
    """Calculate weighted overall score."""
    # Weights: faithfulness and accuracy are most important
    weights = {
        'faithfulness': 0.35,
        'accuracy': 0.30,
        'completeness': 0.20,
        'consistency': 0.15
    }
    
    weighted_score = (
        faithfulness * weights['faithfulness'] +
        accuracy * weights['accuracy'] +
        completeness * weights['completeness'] +
        consistency * weights['consistency']
    )
    
    return int(weighted_score)

def get_quality_assessment(source_text: str, generated_content: Dict[str, Any], 
                         llm: BaseLanguageModel) -> QualityScore:
    """
    Get detailed quality assessment for manual review including extra_data evaluation.
    
    Returns:
        QualityScore object with detailed breakdown and identified issues
    """
    try:
        generated_summary = _format_generated_content(generated_content)
        
        faithfulness = _evaluate_faithfulness(source_text, generated_summary, llm)
        consistency = _evaluate_consistency(generated_content, llm)
        completeness = _evaluate_completeness(source_text, generated_content, llm)
        accuracy = _evaluate_accuracy(source_text, generated_content, llm)
        overall = _calculate_weighted_score(faithfulness, consistency, completeness, accuracy)
        
        # Identify issues for manual review including extra_data issues
        issues = []
        if faithfulness < 70:
            issues.append("Low faithfulness - check for hallucinations in basic or strategic extra data")
        if consistency < 70:
            issues.append("Internal inconsistencies detected across data sections")
        if completeness < 70:
            issues.append("Missing important information in extraction or strategic analysis")
        if accuracy < 70:
            issues.append("Potential factual errors in basic content or strategic extra data")
        
        # Check for specific extra_data issues
        extra_data = generated_content.get("extra_data", {})
        if not extra_data:
            issues.append("No strategic extra data extracted")
        elif len(extra_data) < 5:  # Expecting multiple fields
            issues.append("Limited strategic data extraction")
        
        return QualityScore(
            faithfulness=faithfulness,
            consistency=consistency,
            completeness=completeness,
            accuracy=accuracy,
            overall_score=overall,
            issues=issues
        )
        
    except Exception as e:
        logger.error(f"Error in quality assessment: {e}")
        return QualityScore(
            faithfulness=75, consistency=75, completeness=75, 
            accuracy=75, overall_score=75, issues=["Evaluation error"]
        )

def get_low_scoring_fields(
    source_text: str,
    generated_content: Dict[str, Any],
    llm: BaseLanguageModel,
    threshold: int = 70
) -> List[str]:
    """
    Returns the names of any score fields whose value is below `threshold`.
    """
    qs = get_quality_assessment(source_text, generated_content, llm)
    low = []
    for field_name, value in qs.dict().items():
        if field_name == "issues":
            continue
        if isinstance(value, int) and value < threshold:
            low.append(field_name)
    return low