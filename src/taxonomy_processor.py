import logging
from typing import Dict, List, Tuple, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Import the tags and modules you need
from src.tags import ACTORS_TAXONOMY

logger = logging.getLogger(__name__)

class TaxonomyMatch(BaseModel):
    """Matched entity with taxonomy classification"""
    entity: str = Field(description="Normalized entity name")
    category: str = Field(description="Category from taxonomy or 'No clasificado'")
    confidence: float = Field(description="Match confidence score (0-1)")

def process_entities_with_taxonomy(
    entities_by_folder: Dict[str, Dict[str, List[str]]],
    llm,
    threshold: float = 0.7
) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], Dict[str, Dict[str, List[str]]]]:
    """
    Filter and classify entities based on taxonomy using LLM.
    
    Args:
        entities_by_folder: Dictionary with folders and their extracted entities
        llm: Language model instance
        threshold: Minimum confidence threshold for taxonomy matches
        
    Returns:
        Tuple containing:
        - Dictionary of filtered entities with taxonomy classification
        - Dictionary of rejected entities
    """
    filtered_entities = {}
    rejected_entities = {}
    
    for folder, entities in entities_by_folder.items():
        logger.info(f"Processing entities for folder: {folder}")
        
        # Process organizations
        orgs = entities.get("organizations", [])
        if orgs:
            org_matches = match_entities_to_taxonomy(
                orgs, 
                ACTORS_TAXONOMY, 
                llm, 
                "Match organizations to actor categories",
                threshold
            )
            
            accepted_orgs = [match.model_dump() for match in org_matches if match.confidence >= threshold]
            rejected_orgs = [org for org in orgs if org not in [match.entity for match in org_matches if match.confidence >= threshold]]
            
            if not folder in filtered_entities:
                filtered_entities[folder] = {}
            filtered_entities[folder]["organizations"] = accepted_orgs
            
            if rejected_orgs:
                if not folder in rejected_entities:
                    rejected_entities[folder] = {}
                rejected_entities[folder]["organizations"] = rejected_orgs
        
        # Process geopolitical entities (these don't need taxonomy matching)
        gpes = entities.get("geopolitical_entities", [])
        if gpes:
            if not folder in filtered_entities:
                filtered_entities[folder] = {}
            filtered_entities[folder]["geopolitical_entities"] = gpes
    
    return filtered_entities, rejected_entities

def match_entities_to_taxonomy(
    entities: List[str],
    taxonomy: Dict[str, List[Dict[str, str]]],
    llm,
    instruction: str,
    threshold: float
) -> List[TaxonomyMatch]:
    """
    Match a list of entities against a taxonomy using LLM.
    
    Args:
        entities: List of entity strings to match
        taxonomy: Taxonomy dictionary
        llm: Language model instance
        instruction: Instructions for matching
        threshold: Minimum confidence threshold
        
    Returns:
        List of TaxonomyMatch objects
    """
    # Format taxonomy for prompt
    taxonomy_text = []
    for category, items in taxonomy.items():
        taxonomy_text.append(f"Category: {category}")
        for item in items:
            taxonomy_text.append(f"- {item['label']}: {item['description']}")
    
    taxonomy_formatted = "\n".join(taxonomy_text)
    entities_formatted = "\n".join([f"- {entity}" for entity in entities])
    
    # Create output parser
    parser = PydanticOutputParser(pydantic_object=List[TaxonomyMatch])
    format_instructions = parser.get_format_instructions()
    
    # Create prompt
    prompt = PromptTemplate(
        template="""
        You need to match these entities against a taxonomy.
        
        ENTITIES:
        {entities}
        
        TAXONOMY:
        {taxonomy}
        
        INSTRUCTIONS:
        {instruction}
        
        For each entity:
        1. Normalize spelling, capitalization, and aliases
        2. Find the best matching category in the taxonomy
        3. If no good match exists (below {threshold} confidence), use "No clasificado"
        4. Assign a confidence score between 0 and 1
        
        Return a list of matches.
        
        {format_instructions}
        """,
        input_variables=["entities", "taxonomy", "instruction", "threshold"],
        partial_variables={"format_instructions": format_instructions},
    )
    
    # Run the prompt through the LLM
    # This is a simplified example - you might want to add error handling
    try:
        chain = prompt | llm | parser
        result = chain.invoke({
            "entities": entities_formatted,
            "taxonomy": taxonomy_formatted,
            "instruction": instruction,
            "threshold": threshold
        })
        return result
    except Exception as e:
        logger.error(f"Error matching entities to taxonomy: {e}")
        return []