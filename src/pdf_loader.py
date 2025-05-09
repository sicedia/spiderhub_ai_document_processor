import os
import logging
from typing import Dict, List, Optional
import concurrent.futures
from pypdf import PdfReader

def load_pdfs_from_documents(
    documents_path: str,
    use_parallel: bool = False,
    max_workers: Optional[int] = None
) -> Dict[str, str]:
    """
    Load PDFs from immediate subfolders in the documents folder.
    
    Args:
        documents_path: Path to the documents folder
        use_parallel: Whether to use parallel processing
        max_workers: Maximum number of workers for parallel processing
        
    Returns:
        Dictionary mapping 
         { carpeta1: { archivo1.pdf: texto1, archivo2.pdf: texto2, ... },
        carpeta2: {...}, ... }
    """
    # Setup logging
    logger = logging.getLogger(__name__)
    subfolders = [f for f in os.listdir(documents_path)
                  if os.path.isdir(os.path.join(documents_path, f))]
    results: Dict[str, Dict[str, str]] = {}

    for subfolder in subfolders:
        subfolder_path = os.path.join(documents_path, subfolder)
        pdf_files = [f for f in os.listdir(subfolder_path)
                     if f.lower().endswith(".pdf")]
        results[subfolder] = {}

        if use_parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_single_pdf, os.path.join(subfolder_path, pdf)): pdf
                    for pdf in pdf_files
                }
                for fut in concurrent.futures.as_completed(futures):
                    pdf = futures[fut]
                    results[subfolder][pdf] = fut.result()
        else:
            for pdf in pdf_files:
                path = os.path.join(subfolder_path, pdf)
                results[subfolder][pdf] = process_single_pdf(path)

    return results

def process_pdfs_sequential(subfolder_path: str, pdf_files: List[str]) -> str:
    """Process PDF files sequentially."""
    logger = logging.getLogger(__name__)
    combined_text = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(subfolder_path, pdf_file)
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                combined_text.append(text)
                logger.info(f"Successfully processed {pdf_path}")
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
    
    return "\n\n".join(combined_text)

def process_single_pdf(pdf_path: str) -> str:
    """Process a single PDF file and return its text."""
    logger = logging.getLogger(__name__)
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            logger.info(f"Successfully processed {pdf_path}")
            return text
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return ""

def process_pdfs_parallel(
    subfolder_path: str, 
    pdf_files: List[str], 
    max_workers: Optional[int] = None
) -> str:
    """Process PDF files in parallel."""
    pdf_paths = [os.path.join(subfolder_path, pdf) for pdf in pdf_files]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_pdf, pdf_paths))
    
    return "\n\n".join([r for r in results if r])  # Filter out empty results