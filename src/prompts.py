from langchain_core.prompts import ChatPromptTemplate
from src.tags import MAIN_THEMES_TAXONOMY as TAG_THEMES
from src.tags import ACTORS_TAXONOMY
import json
from typing import Dict

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
    
    # Nuevo prompt para generar la descripción de los temas principales en bullet points
    prompts["themes_description"] = ChatPromptTemplate.from_messages([
        ("system", common_instruction +
            "Generate a bullet-point list that succinctly describes the most important main themes discussed in the text. Focus on key topics and insights for each main theme."),
        ("human", "Source text:\n\n{text}\n\nMain Themes Description (bullet points):")
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
