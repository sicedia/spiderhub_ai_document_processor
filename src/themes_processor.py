import json
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import PydanticOutputParser

from src.tags import MAIN_THEMES_TAXONOMY

logger = logging.getLogger(__name__)

# --- add this, remove the self-import below ---
def _invoke(prompt, llm, parser, **kwargs):
    """Utility to run the chain and trap errors centrally."""
    try:
        chain = prompt | llm | parser
        return chain.invoke(kwargs)
    except Exception as exc:
        logger.error("Error invoking themes chain: %s", exc)
        return None

# from src.themes_processor import _invoke  # <- remove this line

class SubthemeMatch(BaseModel):
    label: str = Field(description="Subtema label")

class ThemeMatch(BaseModel):
    theme: str = Field(description="Main theme name from taxonomy")
    subthemes: List[SubthemeMatch] = Field(
        description="List of matched subthemes for the main theme"
    )

class ThemesOutput(BaseModel):
    items: List[ThemeMatch] = Field(..., description="Lista de ThemeMatch")

# Few-shot examples to guide the LLM
_FEW_SHOT_EXAMPLES = [
    {
        "input": "El gobierno lanzó un plan de infraestructura digital y creó plataformas de e-gobierno.",
        "output": """
{"items":[
  {
    "theme":"Digital Transformation & Strategy",
    "subthemes":[{"label":"Digital Infrastructure"},{"label":"Digital Platforms"}]
  },
  {
    "theme":"Data & Governance",
    "subthemes":[{"label":"E-Governance"}]
  }
]}"""
    },
    {
        "input": "Se firmó un memorando de entendimiento para acelerar la adopción de IA y promover la ética en su uso.",
        "output": """
{"items":[
  {
    "theme":"Technology & Innovation",
    "subthemes":[{"label":"Artificial Intelligence"},{"label":"AI Ethics"}]
  }
]}"""
    }
]

def process_text_with_themes(
    text: str,
    llm,
    taxonomy: Optional[Dict[str, List[Dict[str, str]]]] = None,
    threshold: float = 0.7
) -> Dict[str, List[str]]:
    """
    Analiza el texto y devuelve { main_theme: [subtheme_label,…] },
    usando taxonomy como JSON estructurado, few-shot con LL(temperature=0).
    """
    if taxonomy is None:
        taxonomy = MAIN_THEMES_TAXONOMY

    # 1) Serializar la taxonomía como JSON
    tax_list = [
        {"theme": theme, "subthemes": subthemes}
        for theme, subthemes in taxonomy.items()
    ]
    taxonomy_json = json.dumps(tax_list, ensure_ascii=False, indent=2)

    # 2) Preparar few-shot, escapando llaves para que no sean variables de template
    few_shot_msgs = []
    for ex in _FEW_SHOT_EXAMPLES:
        escaped_output = ex["output"].strip()\
            .replace("{", "{{").replace("}", "}}")
        few_shot_msgs.append(
            HumanMessagePromptTemplate.from_template(f"Text: {ex['input']}")
        )
        few_shot_msgs.append(
            SystemMessagePromptTemplate.from_template(escaped_output)
        )

    # 3) Crear parser y prompt
    parser = PydanticOutputParser(pydantic_object=ThemesOutput)
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a JSON-output text classifier. "
            "You receive a taxonomy (JSON) and raw text; return only valid JSON."
        ),
        *few_shot_msgs,
        HumanMessagePromptTemplate.from_template(
            """
Taxonomy (JSON):
{taxonomy_json}

Text to classify:
{text}

Instructions:
- Only include matches with confidence ≥ {threshold}.
- A text may mention multiple main themes and several subthemes under each.
- Return exactly this schema, nothing else:

{format_instructions}
            """
        ),
    ])

    # 4) Invocar con temperature=0 para resultados deterministas
    try:
        result = _invoke(
            prompt,
            llm,
            parser,
            taxonomy_json=taxonomy_json,
            text=text,
            threshold=threshold,
            format_instructions=format_instructions,
        )
        matches = result.items if result else []
    except Exception as e:
        logger.error(f"Error processing text with themes: {e}")
        return {}

    # 5) Construir dict final
    theme_dict: Dict[str, List[str]] = {}
    for tm in matches:
        labels = [sub.label for sub in tm.subthemes]
        if labels:
            theme_dict[tm.theme] = labels

    return theme_dict
