import json
import logging
import os
import subprocess
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from src.score_calculation import calculate_score
from src.prompts import build_prompts
from src.themes_processor import process_text_with_themes  # Used to extract themes
from src.actor_processor import process_text_with_actors  # Used to extract actors
from src.extra_data import ExtraData, enrich_report_with_extradata  # Import the new module
from src.score_calculation import get_quality_assessment
from src.template_generator import generate_word_from_template
from src.documentReport import DocumentReport
logger = logging.getLogger(__name__)


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
    
    # Valores por defecto seguros
    default_values = {
        "title": None,
        "date": None,
        "principal_location": None,
        "executive_summary": None,
        "characteristics": [],
        "practical_applications": [],
        "commitments": []
    }
    
    # Process each prompt
    for field, prompt in prompts.items():
        try:
            logger.info(f"Extracting {field}...")
            chain = prompt | llm
            response = chain.invoke({"text": text})
            
            # Procesar respuesta y normalizar valores vacíos
            content = response.content.strip()
            
            # Normalizar valores "null" o vacíos
            if content.lower() in ['null', 'none', 'no information available', '']:
                results[field] = default_values.get(field, None)
            elif field in ["characteristics", "practical_applications", "commitments"]:
                if content == '[]':
                    results[field] = []
                else:
                    bullet_points = []
                    for line in content.split("\n"):
                        line = line.strip()
                        if line.startswith("- "):
                            bullet_points.append(line[2:])
                        elif line and not any(s in line for s in [":", "bullet", "point"]):
                            bullet_points.append(line)
                    results[field] = bullet_points if bullet_points else []
            else:
                results[field] = content if content else default_values.get(field, None)
                
        except Exception as e:
            logger.error(f"Error processing {field}: {e}")
            results[field] = default_values.get(field, None)
    
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
    
    # Enriquecer con datos adicionales estratégicos
    try:
        logger.info("Enriching report with extra data...")
        # pass the interim results dict, the full text, and llm
        enriched = enrich_report_with_extradata(results, text, llm)
        # the helper returns the full report dict with an "extra_data" key
        results["extra_data"] = enriched.get("extra_data", {})
    except Exception as e:
        logger.error(f"Error enriching report with data: {e}")
        results["extra_data"] = {}
        
    try:
        results["score"] = calculate_score(
            source_text=text,
            generated_content=results,
            llm=llm
        )
        # Get detailed quality assessment for manual review

        quality_assessment = get_quality_assessment(text, results, llm)
        
        # Log quality issues for manual review
        if quality_assessment.issues:
            logger.warning(f"Quality issues detected: {quality_assessment.issues}")
            
        # Store detailed scores in extra_data for review
        results["quality_breakdown"] = {
            "faithfulness": quality_assessment.faithfulness,
            "consistency": quality_assessment.consistency,
            "completeness": quality_assessment.completeness,
            "accuracy": quality_assessment.accuracy,
            "issues": quality_assessment.issues
        }
        
    except Exception as e:
        logger.error(f"Error calculating score: {e}")
        results["score"] = None
    
    return DocumentReport(
        title=results.get("title"),  # Permitir None
        date=results.get("date"),    # Permitir None
        location=results.get("principal_location"),  # Permitir None
        executive_summary=results.get("executive_summary"),  # Permitir None
        characteristics=results.get("characteristics", []),
        themes=results.get("themes", {}),
        actors=results.get("actors", {}),
        practical_applications=results.get("practical_applications", []),
        commitments=results.get("commitments", []),
        extra_data=results.get("extra_data", {}),
        score=results.get("score"),
        quality_breakdown=results.get("quality_breakdown", {})
    )

def generate_markdown_report(report: DocumentReport) -> str:
    """
    Generate a markdown report from the extracted information.
    
    Args:
        report: DocumentReport object with extracted information
        
    Returns:
        Markdown formatted report
    """
    md_lines = []
    
    # Handle title with fallback message
    title = report.title or "Document Title Not Available"
    md_lines.append(f"# {title}")
    md_lines.append("")
    
    # Handle date with fallback message  
    date = report.date or "Date not specified"
    md_lines.append(f"**Date**: {date}")
    md_lines.append("")
    
    # Handle location with fallback message
    location = report.location or "Location not specified"
    md_lines.append(f"**Location**: {location}")
    md_lines.append("")
    
    # Add executive summary with fallback
    md_lines.append("## Executive Summary")
    md_lines.append("")
    summary = report.executive_summary or "Executive summary not available in the source document."
    md_lines.append(summary)
    md_lines.append("")
    
    # Add characteristics with proper empty handling
    md_lines.append("## Characteristics")
    md_lines.append("")
    if report.characteristics:
        for char in report.characteristics:
            md_lines.append(f"- {char}")
    else:
        md_lines.append("No key characteristics identified in the document.")
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
    
    # Add practical applications with better messaging
    md_lines.append("## Practical Applications")
    md_lines.append("")
    if report.practical_applications:
        for app in report.practical_applications:
            md_lines.append(f"- {app}")
    else:
        md_lines.append("No existing practical applications or implementations identified.")
    
    # Add commitments with better messaging
    md_lines.append("")
    md_lines.append("## Commitments")
    md_lines.append("")
    if report.commitments:
        for commit in report.commitments:
            md_lines.append(f"- {commit}")
    else:
        md_lines.append("No specific quantifiable commitments or targets identified.")
   
    return "\n".join(md_lines)

def save_report(markdown_content: str, report: DocumentReport, output_dir: str, filename_base: str, 
               template_path: str = None) -> Dict[str, str]:
    """
    Save report in markdown, convert to Word using template, and save structured data as JSON.
    
    Args:
        markdown_content: Report content in markdown format
        report: DocumentReport object with structured data
        output_dir: Directory to save the report
        filename_base: Base filename without extension
        template_path: Optional path to Word template
        
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
        report_dict = report.model_dump()
        if report.quality_breakdown:
            logger.info(f"Quality breakdown included in JSON: {report.quality_breakdown}")
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
    
    # Generate Word document using template
    try:
        generated_docx = generate_word_from_template(
            report=report,
            template_path=template_path,
            output_path=output_dir,
            filename_base=filename_base
        )
        logger.info(f"Successfully generated Word document: {generated_docx}")
    except Exception as e:
        logger.error(f"Error generating Word document with template: {e}")
        # Fallback to pandoc conversion
        try:
            subprocess.run(['pandoc', md_path, '-o', docx_path], check=True)
            logger.info(f"Fallback: Successfully converted to Word using Pandoc: {docx_path}")
        except Exception as pandoc_error:
            logger.error(f"Error with Pandoc fallback: {pandoc_error}")
    
    return {
        "markdown": md_path,
        "docx": docx_path,
        "json": json_path
    }

def generate_report(text: str, llm, output_dir: str, folder_name: str, 
                   template_path: str = None) -> Dict[str, str]:
    """
    Process text, generate report and save to files.
    
    Args:
        text: Text content extracted from PDF
        llm: Language model instance
        output_dir: Output directory path
        folder_name: Name of the folder/document
        template_path: Optional path to Word template
        
    Returns:
        Dictionary with paths to created files
    """
    logger.info(f"Generating report for {folder_name}...")
    report = process_text_with_prompts(text, llm)

    # Enrich the report with extra data
    logger.info("Enriching report with additional strategic extra data...")
    report_dict = report.model_dump()
    enriched_report_dict = enrich_report_with_extradata(report_dict, text, llm)
    
    # Update the report with the enriched data
    report.extra_data = enriched_report_dict.get("extra_data")
    
    # Generate markdown from the report
    markdown = generate_markdown_report(report)
    
    # Pass template_path to save_report
    return save_report(markdown, report, output_dir, folder_name, template_path)