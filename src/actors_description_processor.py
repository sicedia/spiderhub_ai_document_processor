import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field, RootModel

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

logger = logging.getLogger(__name__)

# Model for actor classification (only primary actors)
class ActorsClassification(BaseModel):
    """Classification of actors by primary actors only."""
    primary_actors: List[str] = Field(..., description="List of primary actors")

# Models for actor description
class ActorDescription(BaseModel):
    """Description of an actor or stakeholder."""
    actor: str = Field(..., description="Name of the actor or stakeholder")
    description: str = Field(..., description="Description and justification of its importance")
    
# Root model for a list of actor descriptions
class ActorsDescriptions(RootModel[List[ActorDescription]]):
    pass

def classify_actors(text: str, entities_by_folder: Dict[str, Dict[str, List[str]]], llm) -> ActorsClassification:
    """
    Classifies actors (only primary actors) using the given text and extracted entities.
    
    Args:
        text: The complete text of the folder.
        entities_by_folder: Dictionary of extracted entities by folder.
        llm: Language model instance for classification.
    
    Returns:
        An ActorsClassification object containing a list of primary actors.
    """
    # Combine entity information into a single string (assumes these are actors)
    entities_info = []
    for folder, entities in entities_by_folder.items():
        # It is assumed that the relevant entities for actors are under the key "organizations"
        orgs = entities.get("organizations", [])
        for org in orgs:
            entities_info.append(f"- {org}")
    entities_formatted = "\n".join(entities_info)
    
    # Create an output parser using the ActorsClassification model
    parser = PydanticOutputParser(pydantic_object=ActorsClassification)
    format_instructions = parser.get_format_instructions()
    
    prompt = PromptTemplate(
        template="""
        Analyze the following text and the provided entity information to determine which actors are primary.
        
        TEXT:
        {text}
        
        ENTITIES:
        {entities}
        
        Return a JSON with the following structure:
        {{
            "primary_actors": ["actor1", "actor2", ...]
        }}
        
        {format_instructions}
        """,
        input_variables=["text", "entities"],
        partial_variables={"format_instructions": format_instructions},
    )
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({
            "text": text,
            "entities": entities_formatted
        })
        return result
    except Exception as e:
        logger.error(f"Error classifying actors: {e}")
        return ActorsClassification(primary_actors=[])

def describe_actors(text: str, classification: ActorsClassification, llm) -> List[ActorDescription]:
    """
    Generates a description for each primary actor with an explanation of its importance based on the text.
    
    Args:
        text: Complete text of the folder.
        classification: Result of the actor classification.
        llm: Language model instance for generating descriptions.
    
    Returns:
        A list of ActorDescription objects with the description and justification for each actor.
    """
    # Format the list of actors for the prompt
    actors_list = classification.primary_actors
    actors_formatted = "\n".join([f"- {actor}" for actor in actors_list])
    
    # Create output parser using the ActorsDescriptions model
    parser = PydanticOutputParser(pydantic_object=ActorsDescriptions)
    format_instructions = parser.get_format_instructions()
    
    prompt = PromptTemplate(
        template="""
        Given the following text and the list of identified primary actors, provide a brief description for each actor or stakeholder,
        explaining why they are important within the context of the text.
        
        TEXT:
        {text}
        
        ACTORS:
        {actors}
        
        Return a JSON that is a list of objects with the following structure:
        {{
            "actor": "Name of the actor",
            "description": "Description and importance of the actor"
        }}
        
        {format_instructions}
        """,
        input_variables=["text", "actors"],
        partial_variables={"format_instructions": format_instructions},
    )
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({
            "text": text,
            "actors": actors_formatted,
        })
        logger.debug(f"Result from describe_actors: {result}")
        return result.root
    except Exception as e:
        logger.error(f"Error describing actors: {e}")
        return []

def process_actors_description(text: str, entities_by_folder: Dict[str, Dict[str, List[str]]],
                               llm_classification, llm_description) -> Dict[str, Any]:
    """
    Processes the text and entity information to:
    1. Classify actors as primary.
    2. Generate a description for each actor explaining their relevance.
    
    Args:
        text: Complete text of the folder.
        entities_by_folder: Dictionary of extracted entities.
        llm_classification: LLM instance for actor classification.
        llm_description: LLM instance for generating actor descriptions.
    
    Returns:
        Dictionary with keys "classification" and "descriptions" containing
        the actor classification and the generated descriptions respectively.
    """
    classification = classify_actors(text, entities_by_folder, llm_classification)
    descriptions = describe_actors(text, classification, llm_description)
    return {
        "classification": classification.model_dump(),
        "descriptions": [desc.model_dump() for desc in descriptions]
    }