import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, RootModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.tags import MAIN_THEMES_TAXONOMY

logger = logging.getLogger(__name__)

class SubthemeMatch(BaseModel):
    """Matched subtheme label"""
    label: str = Field(description="Subtema label")

class ThemeMatch(BaseModel):
    """Matched main theme with its subthemes"""
    theme: str = Field(description="Main theme name from taxonomy")
    subthemes: List[SubthemeMatch] = Field(
        description="List of matched subthemes for the main theme"
    )

# Usar RootModel para definir un modelo raíz (RootModel) en lugar de __root__
class ThemesOutput(RootModel[List[ThemeMatch]]):
    pass

def process_text_with_themes(
    text: str,
    llm,
    taxonomy: Optional[Dict[str, List[Dict[str, str]]]] = None,
    threshold: float = 0.7  # parámetro agregado con default 0.7
) -> List[ThemeMatch]:
    """
    Analyze the given text and determine which main themes and subthemes are mentioned,
    based on the MAIN_THEMES_TAXONOMY.
    
    Args:
        text: The text content from a folder to be analyzed.
        llm: Language model instance.
        taxonomy: Optional taxonomy to override MAIN_THEMES_TAXONOMY.
        threshold: Confidence threshold to consider a match valid.
        
    Returns:
        A list of ThemeMatch objects indicating the main themes and subthemes mentioned.
    """
    if taxonomy is None:
        taxonomy = MAIN_THEMES_TAXONOMY

    # Formatear la taxonomía para el prompt.
    taxonomy_lines = []
    for theme, subthemes in taxonomy.items():
        taxonomy_lines.append(f"Theme: {theme}")
        for sub in subthemes:
            taxonomy_lines.append(f"- {sub['label']}: {sub['description']}")
    taxonomy_formatted = "\n".join(taxonomy_lines)

    # Crear output parser usando el modelo ThemesOutput
    parser = PydanticOutputParser(pydantic_object=ThemesOutput)
    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate(
        template="""
        Analyze the following text and determine which main themes and subthemes it discusses based on the taxonomy provided.
        
        TEXT:
        {text}
        
        TAXONOMY:
        {taxonomy}
        
        Only include matches that meet or exceed a confidence threshold of {threshold} (a value between 0 and 1).
        
        A text may mention several main themes and multiple subthemes in each.
        
        Return a JSON list of objects with the following structure:
        {{
            "theme": "Main theme name",
            "subthemes": [
                {{"label": "Subtheme label"}},
                ...
            ]
        }}
        
        {format_instructions}
        """,
        input_variables=["text", "taxonomy", "threshold"],
        partial_variables={"format_instructions": format_instructions},
    )

    try:
        chain = prompt | llm | parser
        result = chain.invoke({
            "text": text,
            "taxonomy": taxonomy_formatted,
            "threshold": threshold
        })
        logger.debug(f"Result from chain.invoke: {result} | Type: {type(result)}")
        # En Pydantic v2, para modelos raíz se utiliza el atributo 'root'
        return result.root  
    except Exception as e:
        logger.error(f"Error processing text with themes: {e}")
        return []