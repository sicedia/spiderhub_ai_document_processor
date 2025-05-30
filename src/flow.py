import logging
import os
from typing import Dict, Any

from src.pdf_loader import load_pdfs_from_documents
from src.report_generator import generate_report

logger = logging.getLogger(__name__)

def run_analysis_pipeline(
    documents_path: str,
    llm,
    use_parallel: bool = False,
    max_workers: int | None = None,
    taxonomy_threshold: float = 0.7,
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Run the complete analysis pipeline including entity extraction and classification.
    
    Args:
        documents_path: Path to the documents directory
        llm: Language model instance for classification
        use_parallel: Whether to use parallel processing for PDF loading
        max_workers: Maximum number of workers for parallel processing
        taxonomy_threshold: Minimum confidence threshold for taxonomy matches
        output_dir: Output directory for reports
        
    Returns:
        Dictionary containing entity extraction and classification results
    """
    logger.info(f"Starting analysis pipeline for: {documents_path}")
    
    # Step 1: Load PDFs
    logger.info("Loading PDF documents")
    docs = load_pdfs_from_documents(documents_path, use_parallel, max_workers)
    
    if not docs:
        logger.warning("No documents found.")
        return {"entities": {}, "reports": {}}
    
    entities_by_folder: dict[str, dict[str, list[str]]] = {}
    reports_paths: dict[str, dict[str, str]] = {}
    
    for folder, texts in docs.items():
        try:
            logger.info(f"→ Processing folder: {folder}")

            # Listado de archivos
            file_names = list(texts.keys())
            files_info = f"Files in folder '{folder}': " + ", ".join(file_names)

            # Combinar texto con delimitadores por archivo
            combined_text = files_info + "\n\n"
            for filename, content in texts.items():
                combined_text += (
                    f"=== START {filename} ===\n"
                    f"{content}\n"
                    f"=== END {filename} ===\n\n"
                )     
            
            # Step 5: Generate report from extracted text and entities
            logger.info(f"  → Generating report for {folder}")
            report_paths = generate_report(
                text=combined_text,
                llm=llm,
                output_dir=output_dir,
                folder_name=folder
            )
            
            reports_paths[folder] = report_paths
            logger.info(f"  → Report generated for {folder}: {report_paths}")
            
        except Exception as e:
            logger.error(f"Error processing folder {folder}: {str(e)}")
            # Opcionalmente, puedes agregar el error al resultado
            reports_paths[folder] = {"error": str(e)}
            # El flujo continúa con la siguiente carpeta
            continue

    return {
        "reports": reports_paths
    }







