import json
import logging
import os
import re
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
from src.ranking_processor import get_top_actors, get_top_themes
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
        # Ejecutar UNA sola vez todas las evaluaciones de calidad
        quality_assessment = get_quality_assessment(text, results, llm)
        results["score"] = quality_assessment.overall_score
        if quality_assessment.issues:
            logger.warning(f"Quality issues detected: {quality_assessment.issues}")
        results["quality_breakdown"] = quality_assessment.dict(
            include={"faithfulness","consistency","completeness","accuracy","issues"}
        )
    except Exception as e:
        logger.error(f"Error calculating score: {e}")
        results["score"] = None
    
    # After processing themes and actors, get top rankings
    logger.info("Ranking top actors and themes...")
    try:
        top_actors = get_top_actors(text, results.get("actors", {}), llm, top_n=3)
        top_themes = get_top_themes(text, results.get("themes", {}), llm, top_n=3)
        
        results["top_actors"] = top_actors
        results["top_themes"] = top_themes
        
    except Exception as e:
        logger.error(f"Error ranking actors/themes: {e}")
        results["top_actors"] = []
        results["top_themes"] = []
    
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
        quality_breakdown=results.get("quality_breakdown", {}),
        top_actors=results.get("top_actors", []),
        top_themes=results.get("top_themes", [])
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

def sanitize_filename(name: str, max_length: int = 150) -> str:
    """
    Reemplaza barras, caracteres inválidos y limita longitud del nombre de archivo.
    
    Args:
        name: Nombre original
        max_length: Longitud máxima del nombre (sin extensión)
        
    Returns:
        Nombre sanitizado y truncado
    """
    # Remover caracteres inválidos para Windows
    sanitized = re.sub(r'[\\/:"*?<>|]+', '_', name)
    
    # Remover caracteres de control y espacios extra
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # Remover puntos al final (problemático en Windows)
    sanitized = sanitized.rstrip('.')
    
    # Truncar si es muy largo, preservando palabras completas
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rsplit(' ', 1)[0]
        # Si queda muy corto, usar los primeros caracteres
        if len(sanitized) < 10:
            sanitized = name[:max_length]
    
    # Fallback si queda vacío
    if not sanitized:
        sanitized = "unnamed_document"
        
    return sanitized

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
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating output directory {output_dir}: {e}")
        raise
    
    # Sanear el filename_base para evitar problemas
    safe_base = sanitize_filename(filename_base, max_length=150)
    
    # Verificar que el path completo no sea demasiado largo
    max_path_length = 260  # Límite de Windows
    sample_path = os.path.join(output_dir, f"{safe_base}.docx")
    
    if len(sample_path) > max_path_length:
        # Reducir más el nombre si el path completo es muy largo
        available_length = max_path_length - len(output_dir) - 10  # margen de seguridad
        safe_base = sanitize_filename(filename_base, max_length=available_length)
        logger.warning(f"Filename truncated due to path length: {safe_base}")
    
    # Paths finales
    md_path = os.path.join(output_dir, f"{safe_base}.md")
    docx_path = os.path.join(output_dir, f"{safe_base}.docx")
    json_path = os.path.join(output_dir, f"{safe_base}.json")
    
    try:
        # Save markdown
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        logger.info(f"Markdown saved: {md_path}")
        
    except Exception as e:
        logger.error(f"Error saving markdown to {md_path}: {e}")
        raise
    
    try:
        # Save JSON with structured data
        with open(json_path, 'w', encoding='utf-8') as f:
            report_dict = report.model_dump()
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON saved: {json_path}")
        
    except Exception as e:
        logger.error(f"Error saving JSON to {json_path}: {e}")
        # No lanzar error aquí, continuar con Word
    
    try:
        # Generate Word document using pandoc
        logger.info("Generating Word document using pandoc...")
        result = subprocess.run(
            ["pandoc", md_path, "-o", docx_path],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Word document saved: {docx_path}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating Word document: {e}")
        logger.error(f"Pandoc stderr: {e.stderr}")
        # Crear un archivo de error en lugar del docx
        error_path = docx_path.replace('.docx', '_error.txt')
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(f"Error generating Word document:\n{e}\n\nStderr:\n{e.stderr}")
        docx_path = error_path
        
    except FileNotFoundError:
        logger.error("Pandoc not found. Please install pandoc to generate Word documents.")
        docx_path = None
   
    return {
        "markdown": md_path,
        "docx": docx_path,
        "json": json_path,
        "safe_filename": safe_base
    }

def generate_report(text: str, llm, output_dir: str, folder_name: str, 
                    template_path: str = None) -> Dict[str, str]:
     logger.info(f"Generating report for {folder_name}...")
     report = process_text_with_prompts(text, llm)

     # Generate markdown from the report
     markdown = generate_markdown_report(report)
     
     # Pass template_path to save_report
     return save_report(markdown, report, output_dir, folder_name, template_path)