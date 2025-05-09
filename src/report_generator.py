import json
import logging
import os
import subprocess
from typing import Dict, List, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.tags import MAIN_THEMES_TAXONOMY as TAG_THEMES
from src.tags import ACTORS_TAXONOMY
from src.score_calculation import calculate_faithfulness_score

logger = logging.getLogger(__name__)

def build_prompts() -> Dict[str, ChatPromptTemplate]:
    """Separate prompt per field."""
    # instrucción común para todos los prompts
    common_instruction = (
        "When multiple source texts are provided, always prioritize information "
        "from the most recent document, but also consider the context of previous documents.\n"
    )

    # Format theme taxonomy for prompt
    theme_json = json.dumps(TAG_THEMES).replace("{", "{{").replace("}", "}}").replace("\n", " ")
    # Format actors taxonomy for prompt
    actors_json = json.dumps(ACTORS_TAXONOMY).replace("{", "{{").replace("}", "}}").replace("\n", " ")

    prompts = {}

    prompts["title"] = ChatPromptTemplate.from_messages([
        ("system", common_instruction +
                   "You are an expert summariser. Provide a concise, descriptive title."),
        ("human", "Source text:\n\n{text}\n\nTitle:")
    ])

    prompts["date"] = ChatPromptTemplate.from_messages([
        ("system", common_instruction +
                   "Extract the exact date (YYYY-MM-DD), output 'YYYY-MM-DD'. If only month/year present, output 'YYYY-MM'. If none: 'No information available.'"),
        ("human", "Source text:\n\n{text}\n\nDate:")
    ])

    prompts["principal_location"] = ChatPromptTemplate.from_messages([
        ("system", "Identify the principal location (country or city) where the event/document originates."),
        ("human", "Source text:\n\n{text}\n\nPrincipal location:")
    ])

    prompts["executive_summary"] = ChatPromptTemplate.from_messages([
        ("system", common_instruction +
                   "Create a concise executive summary of no more than 150 words."),
        ("human", "Source text:\n\n{text}\n\nExecutive summary:")
    ])

    prompts["characteristics"] = ChatPromptTemplate.from_messages([
        ("system",  common_instruction +
                "Summarise the main characteristics in 3‑6 bullet points (≤30 words each)."),
        ("human", "Source text:\n\n{text}\n\nCharacteristics:")
    ])

    prompts["themes"] = ChatPromptTemplate.from_messages([
        ("system", common_instruction +
                f"Identify the main themes mentioned in the text. Assign each to a category from MAIN_THEMES_TAXONOMY: {theme_json}, Use the format: 'Theme: [theme], Sub-category: [sub-category]'.  If no themes are found, output 'No themes found.'"),
        ("human", "Source text:\n\n{text}\n\nMain themes:")
    ])
    
    prompts["actors_stakeholders"] = ChatPromptTemplate.from_messages([
        ("system", common_instruction +
                f"Identify the key actors and stakeholders mentioned in the text. Assign each to a category from ACTORS_TAXONOMY: {actors_json}. Use the format: 'Actor: [name/organization], Category: [category from taxonomy]'. If no actors are found, output 'No actors found.'"),
        ("human", "Source text:\n\n{text}\n\nActors and Stakeholders:")
    ])

    prompts["practical_applications"] = ChatPromptTemplate.from_messages([
        ("system", common_instruction +
                """Extract only concrete, actionable initiatives that are already being implemented or have been established.
            Focus on:
            - Existing programs and initiatives
            - Established MoUs and partnerships
            - Current funding mechanisms
            - Active pilots and projects
            - Implemented policy frameworks
            Format each application as a bullet point. If no practical applications are found, output 'No practical applications identified.'"""),
        ("human", "Source text:\n\n{text}\n\nExisting practical applications:")
    ])

    prompts["commitments"] = ChatPromptTemplate.from_messages([
        ("system", common_instruction +
                """Extract only specific, quantifiable, future commitments mentioned in the document.
            Focus on:
            - Numerical targets with deadlines
            - Pledged funding amounts
            - Concrete deadlines for implementation
            - Specific percentage increases or reductions
            - Measurable goals with clear metrics
            Format each commitment as a bullet point. If no specific commitments are found, output 'No specific commitments identified.'"""),
        ("human", "Source text:\n\n{text}\n\nSpecific quantifiable commitments:")
    ])

    return prompts

class DocumentReport(BaseModel):
    """Structure for the document report."""
    title: str = Field(description="Document title")
    date: str = Field(description="Document date in YYYY-MM-DD or YYYY-MM format")
    location: str = Field(description="Principal location")
    executive_summary: str = Field(description="Executive summary of the document")
    characteristics: List[str] = Field(description="Key characteristics as bullet points")
    themes: Dict[str, List[str]] = Field(description="Main themes categorized")
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
    
    # Process each prompt
    for field, prompt in prompts.items():
        try:
            logger.info(f"Extracting {field}...")
            chain = prompt | llm
            response = chain.invoke({"text": text})
            
            if field == "themes":
                theme_dict = {}
                lines = response.content.strip().split("\n")
                current_category = None

                for raw in lines:
                    line = raw.strip()
                    # 1) Inline format: "Theme: X, Sub-category: Y"
                    if "Theme:" in line and "Sub-category:" in line:
                        parts = line.split("Sub-category:")
                        theme_name = parts[0].replace("Theme:", "").replace(",", "").strip()
                        subcat = parts[1].strip()
                        theme_dict.setdefault(subcat, []).append(theme_name)

                    # 2) Bloque con cabecera y viñetas
                    elif line.endswith(":") and not line.startswith("-"):
                        current_category = line.rstrip(":")
                        theme_dict[current_category] = []
                    elif current_category and line.startswith("-"):
                        theme_dict[current_category].append(line[2:].strip())

                # Fallback si sigue vacío
                if not theme_dict:
                    logger.warning("No themes parsed, raw response:\n" + response.content)
                    theme_dict["Unclassified"] = ["No themes found or parsing failed"]

                results[field] = theme_dict
            
            elif field == "actors_stakeholders":
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
                        # Handle flat list format with "Actor: X, Category: Y" pattern
                        elif "Actor:" in line and "Category:" in line:
                            parts = line.split("Category:")
                            if len(parts) > 1:
                                actor_part = parts[0].replace("Actor:", "").strip()
                                category_part = parts[1].split(",")[0].strip()
                                
                                if category_part not in actors_dict:
                                    actors_dict[category_part] = []
                                
                                actors_dict[category_part].append(actor_part)
                    
                    # If parsing failed to find categories, create "Unclassified" category
                    if not actors_dict:
                        actors_dict["Unclassified"] = [
                            line.strip() for line in lines 
                            if line.strip() and not line.strip().endswith(":") and "No actors found" not in line
                        ]
                    
                    results[field] = actors_dict
                except Exception as e:
                    logger.error(f"Error parsing actors: {e}")
                    # Fallback
                    results[field] = {"Unclassified": ["Actor parsing error"]}
            
            elif field in ["characteristics", "practical_applications", "commitments"]:
                # Convert to list of bullet points
                bullet_points = []
                for line in response.content.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("- "):
                        bullet_points.append(line[2:])
                    elif line and not any(s in line for s in [":", "bullet", "point"]):
                        bullet_points.append(line)
                
                results[field] = bullet_points
            
            else:
                # For other fields, just take the text response
                results[field] = response.content.strip()
                
        except Exception as e:
            logger.error(f"Error processing {field}: {e}")
            if field in ["themes", "actors_stakeholders"]:
                results[field] = {"Error": ["Processing failed"]}
            elif field in ["characteristics", "practical_applications", "commitments"]:
                results[field] = ["Processing failed"]
            else:
                results[field] = "Processing failed"
    
    # Calculate faithfulness score using proper evaluation
    try:
        results["faithfulness_score"] = calculate_faithfulness_score(
            source_text=text,
            generated_content=results,
            llm=llm
        )
    except Exception as e:
        logger.error(f"Error calculating faithfulness score: {e}")
        results["faithfulness_score"] = None
    
    # Create and return the report model
    return DocumentReport(
        title=results.get("title", "Untitled Document"),
        date=results.get("date", "No date available"),
        location=results.get("principal_location", "Unknown location"),
        executive_summary=results.get("executive_summary", "No summary available"),
        characteristics=results.get("characteristics", []),
        themes=results.get("themes", {}),
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
    
    # Add actors and stakeholders (agrupados)
    md_lines.append("## Actors")
    md_lines.append("")
    if report.actors_stakeholders:
        md_lines.append("| Category | Actors |")
        md_lines.append("| --- | --- |")
        for category, actors in report.actors_stakeholders.items():
            # limpiamos coma/trailing spaces de cada actor
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
        md_lines.append("| Sub-category | Themes |")
        md_lines.append("| --- | --- |")
        for subcat, themes_list in report.themes.items():
            themes_str = "; ".join(themes_list)
            md_lines.append(f"| {subcat} | {themes_str} |")
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
    md_lines.append("## Future Commitments")
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
    # Process with prompts
    logger.info(f"Generating report for {folder_name}...")
    report = process_text_with_prompts(text, llm)
    
    # Generate markdown
    markdown = generate_markdown_report(report, entities)
    
    # Save files
    return save_report(markdown, output_dir, folder_name)