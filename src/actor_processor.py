import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.tags import ACTORS_TAXONOMY

logger = logging.getLogger(__name__)

class ActorMatch(BaseModel):
    """Matched actor label"""
    label: str = Field(description="Actor label")

class ActorCategoryMatch(BaseModel):
    """Matched actor category with its actors"""
    category: str = Field(description="Actor category name from taxonomy")
    actors: List[ActorMatch] = Field(
        description="List of matched actors for the category"
    )

class ActorsOutput(BaseModel):
    """Wrapper that accepts {'items': [ActorCategoryMatch, ...]}"""
    items: List[ActorCategoryMatch] = Field(..., description="List of ActorCategoryMatch")

def process_text_with_actors(
    text: str,
    llm,
    taxonomy: Optional[Dict[str, List[Dict[str, str]]]] = None,
    threshold: float = 0.7
) -> Dict[str, List[str]]:
    """
    Analyzes the text and returns a dict { actor_category: [actor_label, ...], ... }
    
    Args:
        text: The text content from a folder to be analyzed.
        llm: Language model instance.
        taxonomy: Optional taxonomy to override ACTORS_TAXONOMY.
        threshold: Confidence threshold to consider a match valid.
        
    Returns:
        A dictionary where each key is an actor category and the value is a list of matched actor labels.
    """
    if taxonomy is None:
        taxonomy = ACTORS_TAXONOMY

    # Format the taxonomy for the prompt.
    taxonomy_lines = []
    for category, actors in taxonomy.items():
        taxonomy_lines.append(f"Category: {category}")
        for actor in actors:
            taxonomy_lines.append(f"- {actor['label']}: {actor['description']}")
    taxonomy_formatted = "\n".join(taxonomy_lines)

    # Create output parser using the ActorsOutput model
    parser = PydanticOutputParser(pydantic_object=ActorsOutput)
    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate(
        template="""
        Analyze the following text and determine which actor categories and specific actors it mentions based on the taxonomy provided.
        
        TEXT:
        {text}
        
        TAXONOMY:
        {taxonomy}
        
        Only include matches that meet or exceed a confidence threshold of {threshold} (a value between 0 and 1).
        
        A text may mention several actor categories and multiple actors in each.
        
        Return a JSON list of objects with the following structure:
        {{
            "category": "Actor category name",
            "actors": [
                {{"label": "Actor label"}},
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
        # result is an ActorsOutput, so we extract .items
        matches = result.items  # List[ActorCategoryMatch]
    except Exception as e:
        logger.error(f"Error processing text with actors: {e}")
        return {}

    # Build the final dict
    actor_dict: Dict[str, List[str]] = {}
    for am in matches:
        labels = [actor.label for actor in am.actors]
        if labels:
            actor_dict[am.category] = labels

    return actor_dict