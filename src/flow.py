import logging
from typing import Dict, Any

from src.pdf_loader import load_pdfs_from_documents
from src.nlp import extract_entities_from_folder
from src.entity_processor import process_all_entities

logger = logging.getLogger(__name__)

def run_analysis_pipeline(
    documents_path: str,
    llm,
    use_parallel: bool = False,
    max_workers: int | None = None
) -> Dict[str, Any]:
    """
    Run the complete analysis pipeline including entity extraction and classification.
    
    Args:
        documents_path: Path to the documents directory
        llm: Language model instance for classification
        use_parallel: Whether to use parallel processing for PDF loading
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary containing entity extraction and classification results
    """
    logger.info(f"Starting analysis pipeline for: {documents_path}")
    
    # Step 1: Load PDFs
    logger.info("Loading PDF documents")
    docs = load_pdfs_from_documents(documents_path, use_parallel, max_workers)
    
    if not docs:
        logger.warning("No documents found.")
        return {"entities": {}, "classifications": {}}
    
    entities_by_folder: dict[str, dict[str, list[str]]] = {}
    for folder, texts in docs.items():
        logger.info(f"→ Extracting entities in {folder}")
         # Step 2: Extract entities from loaded documents
        # This function is expected to return a dictionary with keys "organizations" and "geopolitical_entities" by default
        raw = extract_entities_from_folder(texts)
        entities_by_folder[folder] = process_all_entities(raw, llm)
        #Filter entities and keep only the entities inside the taxonomy
        #For organizations, we keep only the ones that are in the taxonomy
        
        logger.info(f"   → {folder}: {entities_by_folder[folder]}")

    return {"entities": entities_by_folder}







