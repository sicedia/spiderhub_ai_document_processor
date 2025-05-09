from typing import List, Dict, Any
from fuzzywuzzy import fuzz

def get_all_labels_from_taxonomy(taxonomy: Dict[str, List[Dict[str, str]]]) -> List[str]:
    """
    Extracts all unique labels from a given taxonomy.

    Args:
        taxonomy: The taxonomy dictionary, where keys are categories and values are lists
                  of dictionaries, each containing a 'label' key.
                  Example: {"Category": [{"label": "Label1", "description": "..."}]}

    Returns:
        A list of unique label strings from the taxonomy.
    """
    all_labels = set()
    for category_items in taxonomy.values():
        for item in category_items:
            if "label" in item:
                all_labels.add(item["label"])
    return list(all_labels)

def filter_entities_by_similarity(
    entities_to_filter: List[str],
    reference_labels: List[str],
    similarity_threshold: int = 80
) -> List[str]:
    """
    Filters a list of entities based on their similarity to a list of reference labels.

    Args:
        entities_to_filter: A list of entity strings to be filtered.
        reference_labels: A list of reference label strings (e.g., from a taxonomy).
        similarity_threshold: An integer (0-100) representing the minimum similarity
                              ratio for an entity to be considered a match.

    Returns:
        A list of entities from `entities_to_filter` that meet the similarity
        threshold with at least one of the `reference_labels`.
    """
    if not entities_to_filter or not reference_labels:
        return []

    matched_entities = []
    for entity in entities_to_filter:
        for label in reference_labels:
            # Using token_set_ratio for more robust matching against phrases
            score = fuzz.token_set_ratio(entity, label)
            if score >= similarity_threshold:
                matched_entities.append(entity)
                break  # Found a match for this entity, move to the next
    return matched_entities

def filter_normalized_organizations_by_taxonomy(
    normalized_entities_for_folder: Dict[str, List[str]],
    organization_taxonomy_labels: List[str],
    similarity_threshold: int = 80
) -> Dict[str, List[str]]:
    """
    Filters the 'organizations' list within a normalized entities dictionary
    for a single folder based on similarity to taxonomy labels.

    Args:
        normalized_entities_for_folder: Dictionary containing lists of normalized entities,
                                       e.g., {"organizations": [...], "geopolitical_entities": [...]}.
        organization_taxonomy_labels: List of reference labels for organizations.
        similarity_threshold: Minimum similarity score for an entity to be kept.

    Returns:
        The normalized_entities_for_folder dictionary with the 'organizations' list
        potentially filtered.
    """
    if "organizations" in normalized_entities_for_folder and normalized_entities_for_folder["organizations"]:
        orgs_to_filter = normalized_entities_for_folder["organizations"]
        
        filtered_orgs = filter_entities_by_similarity(
            entities_to_filter=orgs_to_filter,
            reference_labels=organization_taxonomy_labels,
            similarity_threshold=similarity_threshold
        )
        # Return a new dictionary or modify in place. Here, we update in place for simplicity,
        # assuming the caller is aware or it's the intended behavior.
        # For safety, you might return a modified copy:
        # result = normalized_entities_for_folder.copy()
        # result["organizations"] = filtered_orgs
        # return result
        normalized_entities_for_folder["organizations"] = filtered_orgs
        
    return normalized_entities_for_folder