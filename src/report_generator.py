import json
import logging
import os
import subprocess
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from src.score_calculation import calculate_faithfulness_score
from src.prompts import build_prompts
from src.themes_processor import process_text_with_themes  # Usado para extraer themes

logger = logging.getLogger(__name__)

class DocumentReport(BaseModel):
    """Structure for the document report."""
    title: str = Field(description="Document title")
    date: str = Field(description="Document date in YYYY-MM-DD or YYYY-MM format")
    location: str = Field(description="Principal location")
    executive_summary: str = Field(description="Executive summary of the document")
    characteristics: List[str] = Field(description="Key characteristics as bullet points")
    themes: Dict[str, List[str]] = Field(description="Main themes categorized")
    themes_description: Optional[str] = Field(
        description="Bullet-point list describing the most important main themes",
        default=None
    )
    actors_stakeholders: Dict[str, List[str]] = Field(description="Key actors and stakeholders categorized")
    practical_applications: List[str] = Field(description="Existing practical applications")
    commitments: List[str] = Field(description="Future quantifiable commitments")
    faithfulness_score: Optional[int] = Field(description="Faithfulness score (0-100)", default=None)

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
    
    # Process each prompt (except themes, which se procesa por separado)
    for field, prompt in prompts.items():
        # Saltamos "themes_description", será procesado de forma separada
        if field == "themes_description":
            continue

        try:
            logger.info(f"Extracting {field}...")
            chain = prompt | llm
            response = chain.invoke({"text": text})
            
            if field == "actors_stakeholders":
                # Parse actors into the expected dictionary structure
                actors_dict = {}
                try:
                    lines = response.content.strip().split("\n")
                    current_category = None
                    
                    for line in lines:
                        if line.strip().endswith(":"):
                            current_category = line.strip().rstrip(":")
                            actors_dict[current_category] = []
                        elif current_category and line.strip().startswith("- "):
                            actors_dict[current_category].append(line.strip()[2:])
                        # Handle flat list format with "Actor:" and "Category:" pattern
                        elif "Actor:" in line and "Category:" in line:
                            parts = line.split("Category:")
                            if len(parts) > 1:
                                actor_part = parts[0].replace("Actor:", "").strip()
                                category_part = parts[1].split(",")[0].strip()
                                
                                if category_part not in actors_dict:
                                    actors_dict[category_part] = []
                                
                                actors_dict[category_part].append(actor_part)
                    
                    # Fallback si no se encontró ninguna categoría
                    if not actors_dict:
                        actors_dict["Unclassified"] = [
                            line.strip() for line in lines 
                            if line.strip() and not line.strip().endswith(":") and "No actors found" not in line
                        ]
                    
                    results[field] = actors_dict
                except Exception as e:
                    logger.error(f"Error parsing actors: {e}")
                    results[field] = {"Unclassified": ["Actor parsing error"]}
            
            elif field in ["characteristics", "practical_applications", "commitments"]:
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
            if field in ["actors_stakeholders"]:
                results[field] = {"Error": ["Processing failed"]}
            elif field in ["characteristics", "practical_applications", "commitments"]:
                results[field] = ["Processing failed"]
            else:
                results[field] = "Processing failed"
    
    # Procesar themes de manera separada usando themes_processor (ya que ya no está en build_prompts)
    try:
        logger.info("Extracting themes via themes_processor...")
        theme_matches = process_text_with_themes(text, llm)
        theme_dict = {}
        for tm in theme_matches:
            if tm.subthemes:
                for sub in tm.subthemes:
                    key = sub.label
                    if key not in theme_dict:
                        theme_dict[key] = []
                    theme_dict[key].append(tm.theme)
            else:
                key = "Unclassified"
                if key not in theme_dict:
                    theme_dict[key] = []
                theme_dict[key].append(tm.theme)
        results["themes"] = theme_dict
    except Exception as e:
        logger.error(f"Error processing themes: {e}")
        results["themes"] = {"Error": ["Processing failed"]}
    
    # Procesar el prompt themes_description por separado
    try:
        logger.info("Extracting main themes description via themes_description prompt...")
        themes_desc_prompt = prompts["themes_description"]
        chain = themes_desc_prompt | llm
        response = chain.invoke({"text": text})
        results["themes_description"] = response.content.strip()
    except Exception as e:
        logger.error(f"Error processing themes_description: {e}")
        results["themes_description"] = "No themes description generated."
    
    try:
        results["faithfulness_score"] = calculate_faithfulness_score(
            source_text=text,
            generated_content=results,
            llm=llm
        )
    except Exception as e:
        logger.error(f"Error calculating faithfulness score: {e}")
        results["faithfulness_score"] = None
    
    return DocumentReport(
        title=results.get("title", "Untitled Document"),
        date=results.get("date", "No date available"),
        location=results.get("principal_location", "Unknown location"),
        executive_summary=results.get("executive_summary", "No summary available"),
        characteristics=results.get("characteristics", []),
        themes=results.get("themes", {}),
        themes_description=results.get("themes_description"),
        actors_stakeholders=results.get("actors_stakeholders", {}),
        practical_applications=results.get("practical_applications", []),
        commitments=results.get("commitments", []),
        faithfulness_score=results.get("faithfulness_score")
    )

def generate_markdown_report(report: DocumentReport, entity_data: Dict[str, List]) -> str:
    """
    Generate a markdown report from the extracted information.
    
    Args:
        report: DocumentReport object with extracted information
        entity_data: Dictionary containing entities extracted from the document
        
    Returns:
        Markdown formatted report
    """
    md_lines = []
    
    # Add faithfulness score
    if report.faithfulness_score is not None:
        score = report.faithfulness_score
        rating = "Excellent" if score >= 80 else "Regular" if score >= 60 else "Poor"
        md_lines.append(f"**Faithfulness Score**: {score}/100 - {rating}")
        md_lines.append("")
    
    # Add title
    md_lines.append(f"# {report.title}")
    md_lines.append("")
    
    # Add date and location
    md_lines.append(f"**Date**: {report.date}")
    md_lines.append("")
    md_lines.append(f"**Location**: {report.location}")
    md_lines.append("")
    
    # Add executive summary
    md_lines.append("")
    md_lines.append("## Executive Summary")
    md_lines.append("")
    md_lines.append(report.executive_summary)
    md_lines.append("")
    
    # Add characteristics
    md_lines.append("")
    md_lines.append("## Characteristics")
    md_lines.append("")
    for char in report.characteristics:
        md_lines.append(f"- {char}")
    md_lines.append("")
    
    # Add actors and stakeholders
    md_lines.append("## Actors")
    md_lines.append("")
    if report.actors_stakeholders:
        md_lines.append("| Category | Actors |")
        md_lines.append("| --- | --- |")
        for category, actors in report.actors_stakeholders.items():
            cleaned = [actor.rstrip(",").strip() for actor in actors] 
            actors_str = "; ".join(cleaned)
            md_lines.append(f"| {category} | {actors_str} |")
        md_lines.append("")
    else:
        md_lines.append("No actors identified.")
        md_lines.append("")
    
    # Add themes as table (agrupados)
    md_lines.append("## Main Themes")
    md_lines.append("")
    if report.themes:
        # La primera columna muestra los themes y la segunda la sub-categoría
        md_lines.append("| Themes | Sub-category |")
        md_lines.append("| --- | --- |")
        for subcat, themes_list in report.themes.items():
            themes_str = "; ".join(themes_list)
            md_lines.append(f"| {themes_str} | {subcat} |")
        md_lines.append("")
    else:
        md_lines.append("No themes identified.")
        md_lines.append("")
    
    # Add main themes description (nuevo)
    md_lines.append("### Main Themes Description")
    md_lines.append("")
    md_lines.append(report.themes_description or "No themes description provided.")
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
    
    return "\n".join(md_lines)

def save_report(markdown_content: str, output_dir: str, filename_base: str) -> Dict[str, str]:
    """
    Save report in markdown and convert to Word.
    
    Args:
        markdown_content: Report content in markdown format
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

def generate_report(text: str, entities: Dict[str, Any], llm, output_dir: str, folder_name: str) -> Dict[str, str]:
    """
    Process text, generate report and save to files.
    
    Args:
        text: Text content extracted from PDF
        entities: Dictionary of extracted entities
        llm: Language model instance
        output_dir: Output directory path
        folder_name: Name of the folder/document
        
    Returns:
        Dictionary with paths to created files
    """
    logger.info(f"Generating report for {folder_name}...")
    report = process_text_with_prompts(text, llm)
    markdown = generate_markdown_report(report, entities)
    return save_report(markdown, output_dir, folder_name)