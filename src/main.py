import logging
import os
import json
import argparse
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from .flow import run_analysis_pipeline
from src.report_generator import DocumentReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_llm(provider: str, model: str):
    """
    Initialize and return the appropriate LLM based on provider and model.
    
    Args:
        provider: LLM provider ("openai" or "gemini")
        model: Model name
        
    Returns:
        Initialized LLM instance
    """
    api_key = os.getenv("LLMS_API_KEY")
    base_url = os.getenv("LLMS_API_URL")
    if provider == "openai":
        from langchain_openai import OpenAI, ChatOpenAI
        # Extract model name from format like "openai/gpt-4o-mini"
        model_name = model.split("/")[-1] if "/" in model else model
        
        return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)

    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze documents with entity extraction and classification")
    parser.add_argument("--documents", default="documents", help="Path to documents folder")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--output", default="reports", help="Output directory for reports")
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai", help="LLM provider")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="Model name (e.g., openai/gpt-4o-mini, gemini/gemini-2.5-flash-preview-04-17)")
    parser.add_argument("--output_dir", default="reports", help="Output directory for reports")
    parser.add_argument("--template", type=str, default="templates/SPIDER Deliverable Template.docx", help="Path to custom report template (optional)")
    args = parser.parse_args()
    
    
    # Initialize language model
    logger.info(f"Provider={args.provider} · Model={args.model}")
    try:
        llm = get_llm(args.provider, args.model)
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Run analysis pipeline
    logger.info(f"Starting document analysis from: {args.documents}")
    analysis_results = run_analysis_pipeline(
        documents_path=args.documents,
        llm=llm,
        use_parallel=args.parallel,
        max_workers=args.workers,
        output_dir=args.output,
        template_path=args.template
    )
    
    
    logger.info("Analysis completed")
if __name__ == "__main__":
    main()