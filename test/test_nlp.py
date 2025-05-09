import os
import pytest
from unittest import mock
from src.pdf_loader import load_pdfs_from_documents
from src.nlp import extract_entities, extract_entities_from_documents

# Paths to actual documents
DOCUMENTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents")

class TestNLP:
    
    def test_extract_entities_sample_text(self):
        """Test entity extraction on a sample text with known entities."""
        sample_text = "The European Union and Latin America discussed partnerships in Colombia."
        
        result = extract_entities(sample_text)
        
        assert len(result) == 1  # One "page" of text
        assert "organizations" in result[0]
        assert "geopolitical_entities" in result[0]
        
        # Check organizations and geopolitical entities were found
        orgs = result[0]["organizations"]
        gpes = result[0]["geopolitical_entities"]
        assert len(orgs) > 0
        assert len(gpes) > 0
    
    def test_extract_entities_multipage(self):
        """Test entity extraction on text with multiple pages."""
        multi_page_text = "The European Union met in Brussels.\n\nLatin America was represented by Colombia and Brazil."
        
        result = extract_entities(multi_page_text)
        
        # Should have two pages
        assert len(result) == 2
        
        # Both pages should have entities
        assert len(result[0]["organizations"]) > 0 or len(result[0]["geopolitical_entities"]) > 0
        assert len(result[1]["organizations"]) > 0 or len(result[1]["geopolitical_entities"]) > 0
    
    def test_extract_entities_from_documents(self):
        """Test the extract_entities_from_documents function."""
        documents = {
            "doc1": "The European Union visited Colombia.",
            "doc2": "Google opened an office in Brazil."
        }
        
        result = extract_entities_from_documents(documents)
        
        # Check both documents were processed
        assert "doc1" in result
        assert "doc2" in result
        assert len(result["doc1"]) > 0
        assert len(result["doc2"]) > 0
    
    @mock.patch("src.nlp.nlp")
    def test_extract_entities_with_mock(self, mock_nlp):
        """Test extract_entities with mocked spaCy model for predictable results."""
        # Create mock entities
        mock_ent1 = mock.MagicMock()
        mock_ent1.label_ = "ORG"
        mock_ent1.text = "European Union"
        
        mock_ent2 = mock.MagicMock()
        mock_ent2.label_ = "GPE"
        mock_ent2.text = "Colombia"
        
        # Set up the mock
        mock_doc = mock.MagicMock()
        mock_doc.ents = [mock_ent1, mock_ent2]
        mock_nlp.return_value = mock_doc
        
        # Call the function
        result = extract_entities("Test text")
        
        # Assertions
        assert len(result) == 1
        assert result[0]["organizations"] == ["European Union"]
        assert result[0]["geopolitical_entities"] == ["Colombia"]
    
    @pytest.mark.integration
    def test_integration_with_real_pdfs(self):
        """Integration test using the real PDFs in the documents folder."""
        # Load PDFs from the documents directory
        documents_text = load_pdfs_from_documents(DOCUMENTS_PATH)
        
        # Check that we found documents
        assert documents_text, "No documents were loaded from the documents folder"
        
        # Process all the documents
        results = extract_entities_from_documents(documents_text)
        
        # Check that results were returned for each document folder
        assert results, "No entities extracted from documents"
        assert len(results) == len(documents_text), "Results don't match number of document folders"
        
        # Collect all entities across all documents
        all_orgs = []
        all_gpes = []
        
        # Print summary of results for debugging
        print(f"\nProcessed documents from {len(results)} folders/sources")
        
        for folder_name, pages in results.items():
            folder_orgs = [org for page in pages for org in page.get("organizations", [])]
            folder_gpes = [gpe for page in pages for gpe in page.get("geopolitical_entities", [])]
            
            all_orgs.extend(folder_orgs)
            all_gpes.extend(folder_gpes)
            
            # Log summary for each folder
            print(f"\nEntities in '{folder_name}':")
            print(f"  - Organizations: {len(folder_orgs)} (sample: {folder_orgs[:3]})")
            print(f"  - Geopolitical entities: {len(folder_gpes)} (sample: {folder_gpes[:3]})")
            
            # Check that we found at least some entities in each folder
            assert len(pages) > 0, f"No pages with entities found in '{folder_name}'"
        
        # Check for expected entities across all documents
        # EU/European related orgs
        eu_related = any(
            any(term in org.lower() for term in ["eu", "european", "digital alliance"])
            for org in all_orgs
        )
        assert eu_related, "No EU-related organizations found in documents"
        
        # Colombia should be mentioned somewhere
        colombia_mentioned = any("colombia" in gpe.lower() for gpe in all_gpes)
        assert colombia_mentioned, "Colombia not found in geopolitical entities"