import os
import pytest
from unittest import mock
import tempfile
import shutil
from src.pdf_loader import (
    load_pdfs_from_documents,
    process_pdfs_sequential,
    process_pdfs_parallel,
    process_single_pdf
)

# Paths to actual documents
DOCUMENTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents")

class TestPdfLoader:
    
    def test_documents_folder_exists(self):
        """Verify the documents folder exists and contains subdirectories."""
        assert os.path.exists(DOCUMENTS_PATH), "Documents folder doesn't exist"
        
        # Get subfolders
        subfolders = [f for f in os.listdir(DOCUMENTS_PATH) 
                     if os.path.isdir(os.path.join(DOCUMENTS_PATH, f))]
        
        # Verify there are subfolders
        assert len(subfolders) > 0, "No subfolders found in documents directory"
        
        # Verify each subfolder contains at least one PDF
        for subfolder in subfolders:
            subfolder_path = os.path.join(DOCUMENTS_PATH, subfolder)
            pdf_files = [f for f in os.listdir(subfolder_path) 
                        if f.lower().endswith('.pdf')]
            assert len(pdf_files) > 0, f"No PDF files found in {subfolder}"
    
    def test_process_single_pdf_real(self):
        """Test processing a real PDF file from the first available subfolder."""
        # Get first subfolder
        subfolders = [f for f in os.listdir(DOCUMENTS_PATH) 
                     if os.path.isdir(os.path.join(DOCUMENTS_PATH, f))]
        assert len(subfolders) > 0, "No subfolders found"
        
        # Get first PDF in first subfolder
        first_subfolder = subfolders[0]
        subfolder_path = os.path.join(DOCUMENTS_PATH, first_subfolder)
        pdf_files = [f for f in os.listdir(subfolder_path) 
                    if f.lower().endswith('.pdf')]
        assert len(pdf_files) > 0, "No PDF files found"
        
        # Process the PDF
        pdf_path = os.path.join(subfolder_path, pdf_files[0])
        result = process_single_pdf(pdf_path)
        
        # Check that we got some text content back
        assert isinstance(result, str)
        assert len(result) > 100, "Expected significant text content from real PDF"
    
    @mock.patch("src.pdf_loader.PdfReader")
    def test_process_single_pdf_with_mock(self, mock_pdf_reader):
        """Test process_single_pdf with mocked PdfReader."""
        # Setup mock
        mock_page = mock.MagicMock()
        mock_page.extract_text.return_value = "Test content"
        mock_instance = mock_pdf_reader.return_value
        mock_instance.pages = [mock_page, mock_page]
        
        # Get any valid PDF path for testing
        subfolders = [f for f in os.listdir(DOCUMENTS_PATH) 
                     if os.path.isdir(os.path.join(DOCUMENTS_PATH, f))]
        first_subfolder = subfolders[0]
        subfolder_path = os.path.join(DOCUMENTS_PATH, first_subfolder)
        pdf_files = [f for f in os.listdir(subfolder_path) 
                    if f.lower().endswith('.pdf')]
        pdf_path = os.path.join(subfolder_path, pdf_files[0])
        
        result = process_single_pdf(pdf_path)
        
        # Assertions
        assert result == "Test content\nTest content\n"
        assert mock_page.extract_text.call_count == 2
    
    def test_process_pdfs_sequential_real(self):
        """Test sequential processing with real PDFs from a real subfolder."""
        # Get first subfolder with PDFs
        subfolders = [f for f in os.listdir(DOCUMENTS_PATH) 
                     if os.path.isdir(os.path.join(DOCUMENTS_PATH, f))]
        subfolder_path = os.path.join(DOCUMENTS_PATH, subfolders[0])
        pdf_files = [f for f in os.listdir(subfolder_path) 
                    if f.lower().endswith('.pdf')]
        
        result = process_pdfs_sequential(subfolder_path, pdf_files)
        
        # Check we got content
        assert isinstance(result, str)
        assert len(result) > 100, "Expected significant text content from PDFs"
    
    def test_process_pdfs_parallel_real(self):
        """Test parallel processing with real PDFs."""
        # Get first subfolder with PDFs
        subfolders = [f for f in os.listdir(DOCUMENTS_PATH) 
                     if os.path.isdir(os.path.join(DOCUMENTS_PATH, f))]
        subfolder_path = os.path.join(DOCUMENTS_PATH, subfolders[0])
        pdf_files = [f for f in os.listdir(subfolder_path) 
                    if f.lower().endswith('.pdf')]
        
        result = process_pdfs_parallel(subfolder_path, pdf_files, max_workers=2)
        
        # Check we got content
        assert isinstance(result, str)
        assert len(result) > 100, "Expected significant text content from PDFs"
    
    def test_process_single_pdf_error(self):
        """Test error handling when processing a non-existent PDF file."""
        result = process_single_pdf("nonexistent.pdf")
        # Should handle the error and return empty string
        assert result == ""
    
    def test_load_pdfs_from_documents_sequential(self):
        """Test loading all PDFs from documents folder sequentially."""
        result = load_pdfs_from_documents(DOCUMENTS_PATH, use_parallel=False)
        
        # Get actual subfolders
        subfolders = [f for f in os.listdir(DOCUMENTS_PATH) 
                     if os.path.isdir(os.path.join(DOCUMENTS_PATH, f))]
        
        # Check all subfolders were processed
        for subfolder in subfolders:
            assert subfolder in result, f"Subfolder {subfolder} missing from results"
            assert len(result[subfolder]) > 100, f"Insufficient content for {subfolder}"
    
    def test_load_pdfs_from_documents_parallel(self):
        """Test loading all PDFs from documents folder in parallel."""
        result = load_pdfs_from_documents(DOCUMENTS_PATH, use_parallel=True, max_workers=2)
        
        # Get actual subfolders
        subfolders = [f for f in os.listdir(DOCUMENTS_PATH) 
                     if os.path.isdir(os.path.join(DOCUMENTS_PATH, f))]
        
        # Check all subfolders were processed
        for subfolder in subfolders:
            assert subfolder in result, f"Subfolder {subfolder} missing from results"
            assert len(result[subfolder]) > 100, f"Insufficient content for {subfolder}"
    
    def test_compare_sequential_vs_parallel(self):
        """Compare results from sequential and parallel processing."""
        sequential_result = load_pdfs_from_documents(DOCUMENTS_PATH, use_parallel=False)
        parallel_result = load_pdfs_from_documents(DOCUMENTS_PATH, use_parallel=True)
        
        # Both methods should produce the same content
        assert sequential_result.keys() == parallel_result.keys()
        for key in sequential_result:
            # Compare content length (should be roughly the same)
            seq_len = len(sequential_result[key])
            par_len = len(parallel_result[key])
            # Allow small differences due to whitespace handling
            assert abs(seq_len - par_len) < seq_len * 0.05, f"Content length differs significantly for {key}"
            
    @mock.patch("src.pdf_loader.process_pdfs_sequential")
    @mock.patch("src.pdf_loader.process_pdfs_parallel")
    def test_empty_folder_handling(self, mock_parallel, mock_sequential):
        """Test handling of empty folders."""
        # Create temporary empty folder
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_folder = os.path.join(temp_dir, "empty_folder")
            os.makedirs(empty_folder)
            
            # Setup mocks
            mock_sequential.return_value = ""
            mock_parallel.return_value = ""
            
            # Test with sequential processing
            result_seq = load_pdfs_from_documents(temp_dir, use_parallel=False)
            assert "empty_folder" in result_seq
            assert result_seq["empty_folder"] == ""
            
            # Test with parallel processing
            result_par = load_pdfs_from_documents(temp_dir, use_parallel=True)
            assert "empty_folder" in result_par
            assert result_par["empty_folder"] == ""