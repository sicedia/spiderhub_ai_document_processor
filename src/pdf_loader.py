import os
import logging
from typing import Dict, List, Optional
import concurrent.futures
from pypdf import PdfReader

def load_pdfs_from_documents(
    documents_path: str,
    use_parallel: bool = False,
    max_workers: Optional[int] = None
) -> Dict[str, Dict[str, str]]:
    """
    Load PDFs from dos niveles de subcarpetas en la carpeta `documents`.
    Los PDFs están dentro de las carpetas del segundo nivel.
    
    Args:
        documents_path: Path to documents folder
        use_parallel: Whether to use parallel processing
        max_workers: Max workers for parallel
    
    Returns:
        { "Nivel1/Nivel2": { "archivo1.pdf": texto1, ... }, ... }
    """
    logger = logging.getLogger(__name__)
    results: Dict[str, Dict[str, str]] = {}
    empty_folders = []

    # Primer nivel
    level1 = [
        d for d in os.listdir(documents_path)
        if os.path.isdir(os.path.join(documents_path, d))
    ]
    
    for l1 in level1:
        l1_path = os.path.join(documents_path, l1)
        try:
            # Segundo nivel
            level2 = [
                d for d in os.listdir(l1_path)
                if os.path.isdir(os.path.join(l1_path, d))
            ]
        except Exception as e:
            logger.error(f"Error accessing level 1 folder {l1_path}: {e}")
            continue
            
        for l2 in level2:
            l2_path = os.path.join(l1_path, l2)
            key = f"{l1}/{l2}"
            
            # Buscar PDFs DENTRO de la carpeta del segundo nivel
            pdf_files = []
            if os.path.exists(l2_path) and os.path.isdir(l2_path):
                try:
                    all_files = os.listdir(l2_path)
                    pdf_files = [
                        f for f in all_files
                        if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(l2_path, f))
                    ]
                    
                    # Log información sobre el contenido de la carpeta
                    if not all_files:
                        logger.warning(f"Empty folder: {l2_path}")
                        empty_folders.append(key)
                    elif not pdf_files:
                        non_pdf_files = [f for f in all_files if not f.lower().endswith(".pdf")]
                        logger.warning(f"No PDF files in {l2_path} (found {len(non_pdf_files)} non-PDF files)")
                        empty_folders.append(key)
                    else:
                        logger.info(f"Found {len(pdf_files)} PDF files in {l2_path}")
                        
                except Exception as e:
                    logger.error(f"Error listing files in {l2_path}: {e}")
                    empty_folders.append(key)
                    continue
            else:
                logger.error(f"Path does not exist or is not a directory: {l2_path}")
                empty_folders.append(key)
                continue

            # Solo procesar si hay PDFs
            if pdf_files:
                results[key] = {}

                if use_parallel:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(process_single_pdf, os.path.join(l2_path, pdf)): pdf
                            for pdf in pdf_files
                        }
                        for fut in concurrent.futures.as_completed(futures):
                            pdf = futures[fut]
                            try:
                                results[key][pdf] = fut.result()
                            except Exception as e:
                                logger.error(f"Error processing {pdf}: {e}")
                                results[key][pdf] = ""
                else:
                    for pdf in pdf_files:
                        path = os.path.join(l2_path, pdf)
                        results[key][pdf] = process_single_pdf(path)

    # Resumen final
    if empty_folders:
        logger.warning(f"Found {len(empty_folders)} empty folders: {', '.join(empty_folders[:5])}{'...' if len(empty_folders) > 5 else ''}")
    
    logger.info(f"Successfully processed {len(results)} folders with PDF files")
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