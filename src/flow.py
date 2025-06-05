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
    output_dir: str = "reports",
    template_path: str | None = None
) -> Dict[str, Any]:
    """Run the complete analysis pipeline."""
    logger.info(f"Starting analysis pipeline for: {documents_path}")
    
    # Validar que el directorio de documentos existe
    if not os.path.exists(documents_path):
        logger.error(f"Documents path does not exist: {documents_path}")
        return {"entities": {}, "reports": {}, "errors": ["Documents path not found"]}
    
    # Validar que el directorio de salida se puede crear
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Cannot create output directory {output_dir}: {e}")
        return {"entities": {}, "reports": {}, "errors": [f"Cannot create output directory: {e}"]}
    
    # Step 1: Load PDFs
    logger.info("Loading PDF documents")
    try:
        docs = load_pdfs_from_documents(documents_path, use_parallel, max_workers)
    except Exception as e:
        logger.error(f"Error loading PDFs: {e}")
        return {"entities": {}, "reports": {}, "errors": [f"Error loading PDFs: {e}"]}
    
    if not docs:
        logger.warning("No documents found.")
        return {"entities": {}, "reports": {}, "errors": ["No documents found"]}
    
    entities_by_folder: dict[str, dict[str, list[str]]] = {}
    reports_paths: dict[str, dict[str, str]] = {}
    processing_errors = []
    
    for folder, texts in docs.items():
        try:
            logger.info(f"→ Processing folder: {folder}")
            
            # Validar que hay contenido
            if not texts:
                logger.warning(f"No text content in folder: {folder}")
                processing_errors.append(f"No content in {folder}")
                continue

            # Listado de archivos
            file_names = list(texts.keys())
            files_info = f"Files in folder '{folder}': " + ", ".join(file_names)

            # Combinar texto con delimitadores por archivo
            combined_text = files_info + "\n\n"
            for filename, content in texts.items():
                if content.strip():  # Solo incluir archivos con contenido
                    combined_text += (
                        f"=== START {filename} ===\n"
                        f"{content}\n"
                        f"=== END {filename} ===\n\n"
                    )
            
            # Validar que el texto combinado no esté vacío
            if not combined_text.strip():
                logger.warning(f"Empty combined text for folder: {folder}")
                processing_errors.append(f"Empty content in {folder}")
                continue
            
            # Step 5: Generate report from extracted text and entities
            logger.info(f"  → Generating report for {folder}")
            report_paths = generate_report(
                text=combined_text,
                llm=llm,
                output_dir=output_dir,
                folder_name=folder,
                template_path=template_path
            )
            
            reports_paths[folder] = report_paths
            logger.info(f"  → Report generated for {folder}")
            
        except Exception as e:
            error_msg = f"Error processing folder {folder}: {str(e)}"
            logger.error(error_msg)
            processing_errors.append(error_msg)
            reports_paths[folder] = {"error": str(e)}
            continue

    return {
        "reports": reports_paths,
        "errors": processing_errors,
        "summary": {
            "total_folders": len(docs),
            "successful": len(reports_paths) - len([r for r in reports_paths.values() if "error" in r]),
            "failed": len(processing_errors)
        }
    }







