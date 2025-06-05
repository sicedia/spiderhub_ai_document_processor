import logging
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

logger = logging.getLogger(__name__)

class RankedItem(BaseModel):
    """Item with ranking information"""
    name: str = Field(description="Name of the item")
    category: str = Field(description="Category the item belongs to")
    relevance_score: float = Field(description="Relevance score between 0 and 1", ge=0, le=1)
    justification: str = Field(description="Brief explanation of why this item is important")

class TopRanking(BaseModel):
    """Top ranked items"""
    items: List[RankedItem] = Field(description="List of top ranked items")

def get_top_actors(
    text: str,
    actors_dict: Dict[str, List[str]],
    llm,
    top_n: int = 3
) -> List[Dict[str, Any]]:
    """
    Get top N most relevant actors from the document.
    
    Args:
        text: Document text
        actors_dict: Actors dictionary from actor_processor
        llm: Language model instance
        top_n: Number of top actors to return
        
    Returns:
        List of top actors with scores and justifications
    """
    if not actors_dict:
        return []
        
    # Flatten actors list
    all_actors = []
    for category, actors in actors_dict.items():
        for actor in actors:
            all_actors.append(f"{actor} ({category})")
    
    if not all_actors:
        return []
    
    actors_text = "\n".join([f"- {actor}" for actor in all_actors])
    
    parser = PydanticOutputParser(pydantic_object=TopRanking)
    format_instructions = parser.get_format_instructions()
    
    prompt = PromptTemplate(
        template="""
        Analyze the following text and rank the detected actors by their relevance and importance in the document.
        
        TEXT:
        {text}
        
        DETECTED ACTORS:
        {actors}
        
        Rank the top {top_n} most relevant actors based on:
        1. Prominence in the text (how much they are mentioned)
        2. Their role and importance in the context
        3. Their influence on the main topics discussed
        4. Their participation in key initiatives or decisions
        
        For each actor, provide:
        - name: The actor name (without category)
        - category: The actor category
        - relevance_score: Score between 0 and 1
        - justification: Brief explanation of their importance
        
        {format_instructions}
        """,
        input_variables=["text", "actors", "top_n"],
        partial_variables={"format_instructions": format_instructions},
    )
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({
            "text": text,
            "actors": actors_text,
            "top_n": top_n
        })
        
        if result and result.items:
            return [item.model_dump() for item in result.items[:top_n]]
        return []
        
    except Exception as e:
        logger.error(f"Error ranking actors: {e}")
        return []

def get_top_themes(
    text: str,
    themes_dict: Dict[str, List[str]],
    llm,
    top_n: int = 3
) -> List[Dict[str, Any]]:
    """
    Get top N most important themes from the document.
    
    Args:
        text: Document text
        themes_dict: Themes dictionary from themes_processor
        llm: Language model instance
        top_n: Number of top themes to return
        
    Returns:
        List of top themes with scores and justifications
    """
    if not themes_dict:
        return []
        
    # Flatten themes list
    all_themes = []
    for main_theme, subthemes in themes_dict.items():
        for subtheme in subthemes:
            all_themes.append(f"{subtheme} ({main_theme})")
    
    if not all_themes:
        return []
    
    themes_text = "\n".join([f"- {theme}" for theme in all_themes])
    
    parser = PydanticOutputParser(pydantic_object=TopRanking)
    format_instructions = parser.get_format_instructions()
    
    prompt = PromptTemplate(
        template="""
        Analyze the following text and rank the detected themes by their importance and centrality in the document.
        
        TEXT:
        {text}
        
        DETECTED THEMES:
        {themes}
        
        Rank the top {top_n} most important themes based on:
        1. How central they are to the document's main message
        2. Amount of content dedicated to each theme
        3. Their strategic importance in the context
        4. How they relate to key objectives or commitments
        
        For each theme, provide:
        - name: The theme name (subtheme)
        - category: The main theme category
        - relevance_score: Score between 0 and 1
        - justification: Brief explanation of their importance
        
        {format_instructions}
        """,
        input_variables=["text", "themes", "top_n"],
        partial_variables={"format_instructions": format_instructions},
    )
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({
            "text": text,
            "themes": themes_text,
            "top_n": top_n
        })
        
        if result and result.items:
            return [item.model_dump() for item in result.items[:top_n]]
        return []
        
    except Exception as e:
        logger.error(f"Error ranking themes: {e}")
        return []