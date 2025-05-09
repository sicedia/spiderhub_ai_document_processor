import logging
from typing import List, Dict, Any, Optional
import os

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import LLM
from langchain.chains import LLMChain
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel, Field

from src.tags import MAIN_THEMES_TAXONOMY, ACTORS_TAXONOMY

# Setup logging
logger = logging.getLogger(__name__)

# Text splitter for chunking documents
text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=200)

class ClassificationResult(BaseModel):
    """Results of classification for a single taxonomy."""
    labels: List[Dict[str, Any]] = Field(
        description="List of detected labels with their score (label and score)."
    )

def classify_taxonomy(
    text: str,
    llm: LLM,
    taxonomy: Dict[str, List[Dict[str, str]]],
    instruction: str,
    threshold: float = 0.3
) -> ClassificationResult:
    """
    Classify text with a single taxonomy.

    Args:
      text: Text content to classify
      llm: instance of LangChain LLM
      taxonomy: dictionary {"Category": [{{label, description}}, ...]}
      instruction: specific instructions for this taxonomy
      threshold: minimum score threshold to include the label

    Returns:
      ClassificationResult with list of {label, score}
    """
    # Convert taxonomy to readable text
    labels_text = []
    for cat, items in taxonomy.items():
        labels_text.append(f"**{cat}**:")
        for item in items:
            labels_text.append(f"- {item['label']}: {item['description']}")
    labels_block = "\n".join(labels_text)

    # Pydantic parser and format instructions
    parser = PydanticOutputParser(pydantic_schema=ClassificationResult)
    fmt_instructions = parser.get_format_instructions()

    prompt = PromptTemplate(
        input_variables=["text", "labels", "instruction", "fmt_instructions"],
        template="""
            Analyze the following text and classify it according to this taxonomy:

            {labels}

            Instructions: {instruction}

            TEXT:
            {text}

            {fmt_instructions}
            """
        )

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_parser=parser
    )

    # Split text into chunks and accumulate scores
    chunks = text_splitter.split_text(text)
    agg: Dict[str, float] = {}
    
    for i, chunk in enumerate(chunks):
        try:
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            result: ClassificationResult = chain.predict_and_parse(
                text=chunk,
                labels=labels_block,
                instruction=instruction,
                fmt_instructions=fmt_instructions
            )
            for item in result.labels:
                if item['score'] >= threshold:
                    agg[item['label']] = max(agg.get(item['label'], 0), item['score'])
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {e}")
    
    # Build final result
    final = ClassificationResult(
        labels=[{"label": lbl, "score": scr} for lbl, scr in agg.items()]
    )
    return final

def classify_document(
    document_text: str,
    llm: LLM,
) -> Dict[str, ClassificationResult]:
    """
    Classify a document using both themes and actors taxonomies.
    
    Args:
        document_text: The text content of the document
        llm: LangChain LLM instance
        
    Returns:
        Dictionary with theme and actor classification results
    """
    logger.info("Classifying document themes")
    themes_result = classify_taxonomy(
        text=document_text,
        llm=llm,
        taxonomy=MAIN_THEMES_TAXONOMY,
        instruction="Select the main relevant themes from the text. Score each theme based on its importance and prevalence in the text.",
        threshold=0.3
    )
    
    logger.info("Classifying document actors")
    actors_result = classify_taxonomy(
        text=document_text,
        llm=llm,
        taxonomy=ACTORS_TAXONOMY,
        instruction="Identify the political and social actors mentioned in the text. Score each actor based on its importance and prevalence in the text.",
        threshold=0.3
    )
    
    return {
        "themes": themes_result,
        "actors": actors_result
    }

def classify_documents(
    documents_dict: Dict[str, str],
    llm: LLM,
) -> Dict[str, Dict[str, ClassificationResult]]:
    """
    Classify multiple documents using the taxonomies.
    
    Args:
        documents_dict: Dictionary mapping document names to their text content
        llm: LangChain LLM instance
        
    Returns:
        Dictionary mapping document names to their classification results
    """
    results = {}
    
    for doc_name, doc_text in documents_dict.items():
        logger.info(f"Classifying document: {doc_name}")
        try:
            results[doc_name] = classify_document(doc_text, llm)
            logger.info(f"Classification completed for {doc_name}")
        except Exception as e:
            logger.error(f"Error classifying {doc_name}: {e}")
    
    return results