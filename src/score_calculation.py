import logging
from typing import Optional, Dict, Any

from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)

def calculate_faithfulness_score(
    source_text: str, 
    generated_content: Dict[str, Any],
    llm: BaseLanguageModel
) -> Optional[int]:
    """
    Calculate faithfulness score by evaluating how accurately the generated content
    represents the source text using LangChain's built-in evaluator.
    
    Args:
        source_text: Original source text from the document
        generated_content: Dictionary of generated content from various prompts
        llm: Language model instance for evaluation
        
    Returns:
        Faithfulness score between 0-100 or None if evaluation fails
    """
    try:
        # Ensure source_text is a string
        if not isinstance(source_text, str) or not source_text.strip():
            logger.error("Source text is empty or not a string")
            return None

        # Prepare representative content for evaluation with safe string handling
        title = str(generated_content.get('title', '')) if generated_content.get('title') is not None else ''
        date = str(generated_content.get('date', '')) if generated_content.get('date') is not None else ''
        location = str(generated_content.get('location', '')) if generated_content.get('location', '') is not None else ''
        executive_summary = str(generated_content.get('executive_summary', '')) if generated_content.get('executive_summary') is not None else ''
        
        # Start building the content to evaluate
        content_to_evaluate = f"Title: {title}\n"
        content_to_evaluate += f"Date: {date}\n"
        content_to_evaluate += f"Location: {location}\n"
        content_to_evaluate += f"Executive summary: {executive_summary}\n\n"
        
        # Add characteristics if available (with safe handling)
        characteristics = generated_content.get("characteristics", [])
        if characteristics and isinstance(characteristics, list):
            content_to_evaluate += "Key characteristics:\n"
            for char in characteristics:
                if char is not None:
                    content_to_evaluate += f"- {str(char)}\n"
        
        # Add themes
        themes = generated_content.get("themes", {})
        if themes and isinstance(themes, dict):
            content_to_evaluate += "\nMain themes:\n"
            for category, theme_list in themes.items():
                for theme in theme_list:
                    content_to_evaluate += f"- {category}: {theme}\n"
        
        # Add actors and stakeholders
        actors = generated_content.get("actors_stakeholders", {})
        if actors and isinstance(actors, dict):
            content_to_evaluate += "\nActors and stakeholders:\n"
            for category, actor_list in actors.items():
                for actor in actor_list:
                    content_to_evaluate += f"- {category}: {actor}\n"
        
        # Add practical applications
        applications = generated_content.get("practical_applications", [])
        if applications and isinstance(applications, list):
            content_to_evaluate += "\nPractical applications:\n"
            for app in applications:
                if app is not None:
                    content_to_evaluate += f"- {str(app)}\n"
        
        # Add commitments
        commitments = generated_content.get("commitments", [])
        if commitments and isinstance(commitments, list):
            content_to_evaluate += "\nCommitments:\n"
            for commit in commitments:
                if commit is not None:
                    content_to_evaluate += f"- {str(commit)}\n"

        # Ensure we have something to evaluate
        if not content_to_evaluate.strip():
            logger.error("Generated content is empty after processing")
            return None

        # Labeled criteria evaluator for correctness
        evaluator = load_evaluator(
            evaluator= EvaluatorType.LABELED_CRITERIA,
            criteria="correctness",
            llm=llm
        )
        # call evaluate_strings (sync) with only the args the chain needs
        evaluation_result = evaluator.evaluate_strings(
            input=source_text,           # here you could also use "" if you prefer
            prediction=content_to_evaluate,
            reference=source_text
        )

        # Convert score (0–1) to 0–100
        raw_score  = evaluation_result.get("score", 0)
        final_score = int(raw_score * 100)
        logger.info(f"Faithfulness score calculated: {final_score}/100")
        return final_score

    except Exception as e:
        logger.error(f"Error calculating faithfulness score: {e}")
        return None