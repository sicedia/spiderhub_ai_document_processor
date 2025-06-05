from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import logging
logger = logging.getLogger(__name__)
class DocumentReport(BaseModel):
    """Structure for the document report."""
    title: Optional[str] = Field(description="Document title", default=None)
    date: Optional[str] = Field(description="Document date in YYYY-MM-DD or YYYY-MM format", default=None)
    location: Optional[str] = Field(description="Principal location", default=None) 
    executive_summary: Optional[str] = Field(description="Executive summary of the document", default=None)
    characteristics: List[str] = Field(description="Key characteristics as bullet points", default_factory=list)
    themes: Dict[str, List[str]] = Field(description="Main themes categorized", default_factory=dict)
    actors: Dict[str, List[str]] = Field(description="Key actors and stakeholders categorized", default_factory=dict)
    practical_applications: List[str] = Field(description="Existing practical applications", default_factory=list)
    commitments: List[str] = Field(description="Future quantifiable commitments", default_factory=list)
    score: Optional[int] = Field(description="Faithfulness score (0-100)", default=None)
    quality_breakdown: Optional[Dict[str, Any]] = Field(description="Detailed quality assessment breakdown", default=None)
    extra_data: Optional[Dict[str, Any]] = Field(description="Additional strategic extra data extracted for analysis", default=None)
    top_actors: List[Dict[str, Any]] = Field(
        description="Top 3 most relevant actors with scores",
        default_factory=list
    )
    top_themes: List[Dict[str, Any]] = Field(
        description="Top 3 most important themes with scores", 
        default_factory=list
    )
