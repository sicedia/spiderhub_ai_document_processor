from typing import Dict, List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSequence
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NormalizedEntities(BaseModel):
    """Esquema de salida: listas únicas y limpias"""
    organizations: List[str] = Field(
        description="Lista normalizada y sin duplicados de organizaciones"
    )
    geopolitical_entities: List[str] = Field(
        description="Lista normalizada y sin duplicados de entidades geopolíticas"
    )


def normalize_entities_with_llm(
    orgs: List[str],
    gpes: List[str],
    llm
) -> NormalizedEntities:
    """
    Llama al LLM para:
     - Normalizar mayúsculas/minúsculas y ortografía
     - Eliminar duplicados y falsos positivos
    """
    parser = PydanticOutputParser(pydantic_object=NormalizedEntities)
    fmt = parser.get_format_instructions()
    prompt = PromptTemplate(
        input_variables=["orgs", "gpes", "fmt"],
        template="""
        Deduplicate and normalize the following entities.

        Organizations:
        {orgs}

        Geopolitical entities:
        {gpes}

        Return a JSON matching the schema:

        {fmt}
        """
    )
    pipeline = prompt | llm
    try:
        raw = pipeline.invoke({"orgs": orgs, "gpes": gpes, "fmt": fmt})
        return parser.parse(raw.content)
    except Exception as e:
        logger.error(f"Error normalizing entities: {e}")
        # Devuelve sin normalizar para continuar el flujo
        return NormalizedEntities(
            organizations=[],
            geopolitical_entities=[]
        )


def process_folder_entities(
    entities_by_doc: Dict[str, Dict[str, List[str]]],
    llm
) -> Dict[str, List[str]]:
    """
    Aplanar todas las listas de un folder, llamar a normalize_entities_with_llm
    y devolver el mismo dict con listas limpias.
    """
    all_orgs: List[str] = []
    all_gpes: List[str] = []
    for doc_name, ents in entities_by_doc.items():
        all_orgs.extend(ents.get("organizations", []))
        all_gpes.extend(ents.get("geopolitical_entities", []))
    normalized = normalize_entities_with_llm(all_orgs, all_gpes, llm)
    return {
        "organizations": normalized.organizations,
        "geopolitical_entities": normalized.geopolitical_entities
    }


def process_all_entities(
    entities_by_folder: Dict[str, Dict[str, List[str]]],
    llm
) -> Dict[str, Dict[str, List[str]]]:
    """
    Itera sobre cada carpeta, normaliza y deduplica directamente con el LLM.
    """
    results: Dict[str, Dict[str, List[str]]] = {}
    for folder, ents in entities_by_folder.items():
        orgs = ents.get("organizations", [])
        gpes = ents.get("geopolitical_entities", [])
        normalized = normalize_entities_with_llm(orgs, gpes, llm)
        results[folder] = {
            "organizations": normalized.organizations,
            "geopolitical_entities": normalized.geopolitical_entities
        }
    return results