import json
import logging
import os
import subprocess
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from src.score_calculation import calculate_faithfulness_score
from src.prompts import build_prompts
from src.themes_processor import process_text_with_themes  # Used to extract themes
from src.actor_processor import process_text_with_actors  # Used to extract actors
from src.extra_metadata import ExtraMetadata, enrich_report_with_metadata  # Import the new module

logger = logging.getLogger(__name__)

# Modificar el modelo DocumentReport para incluir extra_metadata
class DocumentReport(BaseModel):
    """Structure for the document report."""
    title: str = Field(description="Document title")
    date: str = Field(description="Document date in YYYY-MM-DD or YYYY-MM format")
    location: str = Field(description="Principal location")
    executive_summary: str = Field(description="Executive summary of the document")
    characteristics: List[str] = Field(description="Key characteristics as bullet points")
    themes: Dict[str, List[str]] = Field(description="Main themes categorized")
    actors: Dict[str, List[str]] = Field(description="Key actors and stakeholders categorized")
    practical_applications: List[str] = Field(description="Existing practical applications")
    commitments: List[str] = Field(description="Future quantifiable commitments")
    faithfulness_score: Optional[int] = Field(description="Faithfulness score (0-100)", default=None)
    extra_metadata: Optional[Dict[str, Any]] = Field(description="Additional strategic metadata extracted for analysis", default=None)
    # Nuevo campo para almacenar las descripciones de actores
    

def process_text_with_prompts(text: str, llm) -> DocumentReport:
    """
    Process document text with various prompts to extract structured information.
    
    Args:
        text: Text content from PDF document
        llm: Language model instance
        
    Returns:
        DocumentReport object with extracted information
    """
    prompts = build_prompts()
    results = {}
    
    logger.info("Applying prompts to document text...")
    
    # Process each prompt (except themes and actors, which are processed separately)
    for field, prompt in prompts.items():
            
        try:
            logger.info(f"Extracting {field}...")
            chain = prompt | llm
            response = chain.invoke({"text": text})
            
            if field in ["characteristics", "practical_applications", "commitments"]:
                bullet_points = []
                for line in response.content.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("- "):
                        bullet_points.append(line[2:])
                    elif line and not any(s in line for s in [":", "bullet", "point"]):
                        bullet_points.append(line)
                
                results[field] = bullet_points
            else:
                results[field] = response.content.strip()
                
        except Exception as e:
            logger.error(f"Error processing {field}: {e}")
            if field in ["characteristics", "practical_applications", "commitments"]:
                results[field] = ["Processing failed"]
            else:
                results[field] = "Processing failed"
    
    # Process themes separately using themes_processor
    logger.info("Extracting themes via themes_processor...")
    try:
        results["themes"] = process_text_with_themes(text, llm)
    except Exception as e:
        logger.error(f"Error processing themes: {e}")
        results["themes"] = {}
        
    # Process actors separately using actor_processor
    logger.info("Extracting actors via actor_processor...")
    try:
        results["actors"] = process_text_with_actors(text, llm)
    except Exception as e:
        logger.error(f"Error processing actors: {e}")
        results["actors"] = {}
    
    try:
        results["faithfulness_score"] = calculate_faithfulness_score(
            source_text=text,
            generated_content=results,
            llm=llm
        )
    except Exception as e:
        logger.error(f"Error calculating faithfulness score: {e}")
        results["faithfulness_score"] = None

    # Enriquecer con metadatos adicionales estratégicos
    try:
        logger.info("Enriching report with extra metadata...")
        # pass the interim results dict, the full text, and llm
        enriched = enrich_report_with_metadata(results, text, llm)
        # the helper returns the full report dict with an "extra_metadata" key
        results["extra_metadata"] = enriched.get("extra_metadata", {})
    except Exception as e:
        logger.error(f"Error enriching report with metadata: {e}")
        results["extra_metadata"] = {}
    
    return DocumentReport(
        title=results.get("title", "Untitled Document"),
        date=results.get("date", "No date available"),
        location=results.get("principal_location", "Unknown location"),
        executive_summary=results.get("executive_summary", "No summary available"),
        characteristics=results.get("characteristics", []),
        themes=results.get("themes", {}),
        actors=results.get("actors", {}),
        practical_applications=results.get("practical_applications", []),
        commitments=results.get("commitments", []),
        faithfulness_score=results.get("faithfulness_score"),
        extra_metadata=results.get("extra_metadata", {})
    )

# Modificar la función generate_markdown_report para que utilice el campo actor_descriptions del reporte
def generate_markdown_report(report: DocumentReport) -> str:
    """
    Generate a markdown report from the extracted information.
    
    Args:
        report: DocumentReport object with extracted information
        
    Returns:
        Markdown formatted report
    """
    md_lines = []
    
    # Add faithfulness score and title
    if report.faithfulness_score is not None:
        score = report.faithfulness_score
        rating = "Excellent" if score >= 80 else "Regular" if score >= 60 else "Poor"
        md_lines.append(f"**Faithfulness Score**: {score}/100 - {rating}")
        md_lines.append("")
    
    md_lines.append(f"# {report.title}")
    md_lines.append("")
    md_lines.append(f"**Date**: {report.date}")
    md_lines.append("")
    md_lines.append(f"**Location**: {report.location}")
    md_lines.append("")
    
    # Add executive summary
    md_lines.append("## Executive Summary")
    md_lines.append("")
    md_lines.append(report.executive_summary)
    md_lines.append("")
    
    # Add characteristics
    md_lines.append("## Characteristics")
    md_lines.append("")
    for char in report.characteristics:
        md_lines.append(f"- {char}")
    md_lines.append("")
    
    # Add actors and stakeholders table
    md_lines.append("## Actors")
    md_lines.append("")
    if report.actors:
        md_lines.append("| Category | Actor |")
        md_lines.append("| --- | --- |")
        for actor_category, actors in report.actors.items():
            actors_str = "; ".join(actors)
            md_lines.append(f"| {actor_category} | {actors_str} |")
        md_lines.append("")
    else:
        md_lines.append("No actors identified.")
        md_lines.append("")
    
    # Add themes as table (agrupados)
    md_lines.append("## Main Themes")
    md_lines.append("")
    if report.themes:
        md_lines.append("| Category | Subcategory |")
        md_lines.append("| --- | --- |")
        for main_theme, subs in report.themes.items():
            subs_str = "; ".join(subs)
            md_lines.append(f"| {main_theme} | {subs_str} |")
        md_lines.append("")
    else:
        md_lines.append("No themes identified.")
        md_lines.append("")
    
    # Add practical applications
    md_lines.append("## Practical Applications")
    md_lines.append("")
    if report.practical_applications:
        for app in report.practical_applications:
            md_lines.append(f"- {app}")
    else:
        md_lines.append("No practical applications identified.")
    
    # Add commitments
    md_lines.append("")
    md_lines.append("## Commitments")
    md_lines.append("")
    if report.commitments:
        for commit in report.commitments:
            md_lines.append(f"- {commit}")
    else:
        md_lines.append("No specific commitments identified.")
    
    # Add extra metadata
    if report.extra_metadata:
        md_lines.append("")
        md_lines.append("## Strategic Metadata")
        md_lines.append("")

        # Document Classification
        md_lines.append("### Document Classification")
        md_lines.append("")
        md_lines.append("| Attribute | Value |")
        md_lines.append("| --- | --- |")
        if report.extra_metadata.get("agreement_type"):
            md_lines.append(f"| Agreement Type | {report.extra_metadata['agreement_type']} |")
        if report.extra_metadata.get("legal_bindingness"):
            md_lines.append(f"| Legal Status | {report.extra_metadata['legal_bindingness']} |")
        if report.extra_metadata.get("lead_country_iso"):
            md_lines.append(f"| Leading Country | {report.extra_metadata['lead_country_iso']} |")
        if report.extra_metadata.get("coverage_scope"):
            md_lines.append(f"| Geographic Scope | {report.extra_metadata['coverage_scope']} |")
        if report.extra_metadata.get("review_schedule"):
            md_lines.append(f"| Review Schedule | {report.extra_metadata['review_schedule']} |")
        md_lines.append("")
        
        # Implementation Timeline
        if report.extra_metadata.get("start_date") or report.extra_metadata.get("end_date"):
            md_lines.append("### Implementation Timeline")
            md_lines.append("")
            md_lines.append("| Start Date | End Date |")
            md_lines.append("| --- | --- |")
            start_date = report.extra_metadata.get("start_date", "Not specified")
            end_date = report.extra_metadata.get("end_date", "Not specified")
            md_lines.append(f"| {start_date} | {end_date} |")
            md_lines.append("")
        
        # Financial Information
        if report.extra_metadata.get("budget_amount_eur") is not None or report.extra_metadata.get("financing_instrument"):
            md_lines.append("### Financial Information")
            md_lines.append("")
            md_lines.append("| Budget (EUR) | Financing Instrument |")
            md_lines.append("| --- | --- |")
            budget = report.extra_metadata.get("budget_amount_eur", "Not specified")
            instrument = report.extra_metadata.get("financing_instrument", "Not specified")
            md_lines.append(f"| {budget} | {instrument} |")
            md_lines.append("")
        
        # Commitment Analysis
        if report.extra_metadata.get("commitment_details"):
            md_lines.append("### Commitment Analysis")
            md_lines.append("")
            md_lines.append("| Commitment | Classification | Implementation Status |")
            md_lines.append("| --- | --- | --- |")
            for commitment in report.extra_metadata.get("commitment_details", []):
                commitment_text = commitment.get("text", "")
                classification = commitment.get("commitment_class", "Unclassified")
                status = commitment.get("implementation_status", "Unknown")
                md_lines.append(f"| {commitment_text[:50]}... | {classification} | {status} |")
            md_lines.append("")
        
        # EU Policy Alignment
        if report.extra_metadata.get("eu_policy_alignment"):
            policies = ", ".join(report.extra_metadata.get("eu_policy_alignment", []))
            md_lines.append("### EU Policy Alignment")
            md_lines.append("")
            md_lines.append(f"Aligned with: {policies}")
            md_lines.append("")
        
        # SDG Alignment
        if report.extra_metadata.get("sdg_alignment"):
            sdgs = ", ".join(report.extra_metadata.get("sdg_alignment", []))
            if sdgs:
                md_lines.append("### SDG Alignment")
                md_lines.append("")
                md_lines.append(f"Aligned with: {sdgs}")
                md_lines.append("")
        
        # Synthesis & Governance Data
        md_lines.append("### Synthesis & Governance")
        md_lines.append("")
        if report.extra_metadata.get("implementation_degree_pct") is not None:
            md_lines.append(f"- Implementation Degree: {report.extra_metadata['implementation_degree_pct']}%")
        if report.extra_metadata.get("actionability_score") is not None:
            md_lines.append(f"- Actionability Score: {report.extra_metadata['actionability_score']}/100")
        if report.extra_metadata.get("monitoring_body"):
            md_lines.append(f"- Monitoring Body: {report.extra_metadata['monitoring_body']}")
        if report.extra_metadata.get("financing_source"):
            md_lines.append(f"- Financing Source: {report.extra_metadata['financing_source']}")
        md_lines.append("")

        # Beneficiary Groups (normalizados)
        if report.extra_metadata.get("beneficiary_group"):
            md_lines.append("### Beneficiary Groups")
            md_lines.append("")
            md_lines.append("| Category           | Label               |")
            md_lines.append("| ------------------ | ------------------- |")
            for bg in report.extra_metadata["beneficiary_group"]:
                md_lines.append(f"| {bg['category']} | {bg['label']} |")
            md_lines.append("")

        # Beneficiary Groups (raw)
        if report.extra_metadata.get("beneficiary_group_raw"):
            raw = ", ".join(report.extra_metadata["beneficiary_group_raw"])
            md_lines.append(f"- Raw Beneficiary Groups: {raw}")
            md_lines.append("")

        # Additional Quantifiable KPIs (new section)
        if report.extra_metadata.get("kpi_list"):
            md_lines.append("### Quantifiable KPIs")
            md_lines.append("")
            md_lines.append("| KPI Description | Target Value | Unit |")
            md_lines.append("| --- | --- | --- |")
            for kpi in report.extra_metadata.get("kpi_list", []):
                desc = kpi.get("kpi", "")
                target = kpi.get("target_value", "")
                unit = kpi.get("unit", "")
                md_lines.append(f"| {desc} | {target} | {unit} |")
            md_lines.append("")
        
        # Countries Mentioned (new section)
        if report.extra_metadata.get("country_list_iso"):
            md_lines.append("### Countries Mentioned")
            md_lines.append("")
            countries = ", ".join(report.extra_metadata.get("country_list_iso", []))
            md_lines.append(countries)
            md_lines.append("")
    
    return "\n".join(md_lines)

def save_report(markdown_content: str, report: DocumentReport, output_dir: str, filename_base: str) -> Dict[str, str]:
    """
    Save report in markdown, convert to Word, and save structured data as JSON.
    
    Args:
        markdown_content: Report content in markdown format
        report: DocumentReport object with structured data
        output_dir: Directory to save the report
        filename_base: Base filename without extension
        
    Returns:
        Dictionary with paths to created files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths
    md_path = os.path.join(output_dir, f"{filename_base}.md")
    docx_path = os.path.join(output_dir, f"{filename_base}.docx")
    json_path = os.path.join(output_dir, f"{filename_base}.json")
    
    # Save markdown
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    # Save JSON with structured data
    with open(json_path, 'w', encoding='utf-8') as f:
        # Convert report to dict and save as JSON
        json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)
    
    # Convert to Word using Pandoc
    try:
        subprocess.run(['pandoc', md_path, '-o', docx_path], check=True)
        logger.info(f"Successfully converted to Word: {docx_path}")
    except Exception as e:
        logger.error(f"Error converting to Word: {e}")
    
    return {
        "markdown": md_path,
        "docx": docx_path,
        "json": json_path
    }

def generate_report(text: str, llm, output_dir: str, folder_name: str) -> Dict[str, str]:
    """
    Process text, generate report and save to files.
    
    Args:
        text: Text content extracted from PDF
        llm: Language model instance
        output_dir: Output directory path
        folder_name: Name of the folder/document
        
    Returns:
        Dictionary with paths to created files
    """
    logger.info(f"Generating report for {folder_name}...")
    report = process_text_with_prompts(text, llm)
    
    # Enrich the report with extra metadata
    logger.info("Enriching report with additional strategic metadata...")
    report_dict = report.model_dump()
    enriched_report_dict = enrich_report_with_metadata(report_dict, text, llm)
    
    # Update the report with the enriched data
    report.extra_metadata = enriched_report_dict.get("extra_metadata")
    
    # Generate markdown from the report
    markdown = generate_markdown_report(report)
    
    # Pass both markdown content and report object to save_report
    return save_report(markdown, report, output_dir, folder_name)