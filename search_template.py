"""
Search functionality for Elasticsearch queries generated from natural language.
This module provides both template-based and direct query building approaches.
"""

import json
import pandas as pd
from typing import Dict, Any, List, Tuple

# Simplified search template with minimal Mustache syntax
SEARCH_TEMPLATE = {
    "script": {
        "lang": "mustache",
        "source": """
{
  "query": {
    "bool": {
      "must": [
        {{#must_include_all}}
        {
          "multi_match": {
            "query": "{{value}}",
            "fields": ["*"],
            "type": "best_fields"
          }
        }
        {{#comma}},{{/comma}}
        {{/must_include_all}}
      ],
      "should": [
        {{#must_atleast_one_of}}
        {
          "multi_match": {
            "query": "{{value}}",
            "fields": ["*"],
            "type": "best_fields"
          }
        }
        {{#comma}},{{/comma}}
        {{/must_atleast_one_of}}
      ],
      "must_not": [
        {{#must_not_include}}
        {
          "multi_match": {
            "query": "{{value}}",
            "fields": ["*"],
            "type": "best_fields"
          }
        }
        {{#comma}},{{/comma}}
        {{/must_not_include}}
      ],
      "filter": [
        {{#filters}}
        {
          "term": {
            "{{field}}": "{{value}}"
          }
        }
        {{#comma}},{{/comma}}
        {{/filters}}
      ]
    }
  },
  "size": {{size}}
}
"""
    }
}

def register_search_template(es_client, template_id="nlq_search_template", force_overwrite=False):
    """
    Register the search template with Elasticsearch.
    
    Args:
        es_client: Elasticsearch client
        template_id: ID to assign to the template
        force_overwrite: Whether to overwrite existing template without confirmation
        
    Returns:
        Response from Elasticsearch or None if operation was cancelled
    """
    try:
        # Check if template exists
        template_exists = False
        try:
            es_client.get_script(id=template_id)
            template_exists = True
        except:
            template_exists = False
        
        if template_exists and not force_overwrite:
            print(f"WARNING: Template '{template_id}' already exists!")
            print("To register the template and overwrite the existing one, call this function with force_overwrite=True")
            return None
        
        # Delete the template if it already exists and we're forcing overwrite
        if template_exists and force_overwrite:
            es_client.delete_script(id=template_id)
            print(f"Deleted existing template: {template_id}")
        
        # Create the new template
        response = es_client.put_script(id=template_id, body=SEARCH_TEMPLATE)
        print(f"Created search template: {template_id}")
        return response
    except Exception as e:
        print(f"Error creating template '{template_id}': {e}")
        return None

# Constants for template id and index name
TEMPLATE_ID = "nlq_search_template"
INDEX_NAME = "webai*"

def build_direct_query(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a direct Elasticsearch query from parameters without using a search template.
    
    Args:
        params: Parameters for the query.
                Example:
                {
                    "must_include_all": ["packaging", "machine"],
                    "filters": [
                        {"field": "country.keyword", "value": "Germany"}
                    ],
                    "size": 100
                }
    
    Returns:
        Elasticsearch query as a dictionary
    """
    # Initialize the bool query structure
    bool_query = {
        "bool": {
            "must": [],
            "should": [],
            "must_not": [],
            "filter": []
        }
    }
    
    # Add must clauses (terms that must be included)
    if "must_include_all" in params and params["must_include_all"]:
        for term in params["must_include_all"]:
            bool_query["bool"]["must"].append({
                "multi_match": {
                    "query": term,
                    "fields": ["*"],
                    "type": "best_fields"
                }
            })
    
    # Add should clauses (at least one must match)
    if "must_atleast_one_of" in params and params["must_atleast_one_of"]:
        for term in params["must_atleast_one_of"]:
            bool_query["bool"]["should"].append({
                "multi_match": {
                    "query": term,
                    "fields": ["*"],
                    "type": "best_fields"
                }
            })
        
        # If we have should clauses, set minimum_should_match to 1
        if bool_query["bool"]["should"]:
            bool_query["bool"]["minimum_should_match"] = 1
    
    # Add must_not clauses (terms that must not be included)
    if "must_not_include" in params and params["must_not_include"]:
        for term in params["must_not_include"]:
            bool_query["bool"]["must_not"].append({
                "multi_match": {
                    "query": term,
                    "fields": ["*"],
                    "type": "best_fields"
                }
            })
    
    # Add filters (exact matches)
    if "filters" in params and params["filters"]:
        for filter_item in params["filters"]:
            bool_query["bool"]["filter"].append({
                "term": {
                    filter_item["field"]: filter_item["value"]
                }
            })
    
    # Build the complete query
    query = {
        "query": bool_query,
        "size": params.get("size", 100)
    }
    
    return query

def prepare_template_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare parameters for the search template by adding comma flags.
    
    Args:
        params: Original parameters
    
    Returns:
        Modified parameters with comma flags
    """
    template_params = params.copy()
    
    # Add comma flags for arrays to handle JSON formatting
    for key in ["must_include_all", "must_atleast_one_of", "must_not_include", "filters"]:
        if key in template_params and template_params[key]:
            # Add a 'comma' flag to all but the last item
            for i in range(len(template_params[key]) - 1):
                if isinstance(template_params[key][i], dict):
                    template_params[key][i]["comma"] = True
                else:
                    # For simple arrays, convert items to dicts with value and comma
                    template_params[key][i] = {
                        "value": template_params[key][i],
                        "comma": True
                    }
    
    return template_params

def execute_search(params: Dict[str, Any], use_template: bool = True) -> Tuple[pd.DataFrame, int]:
    """
    Execute a search using either search template or direct query.
    
    Args:
        params: Parameters for the search.
                Example:
                {
                    "must_include_all": ["packaging", "machine"],
                    "filters": [
                        {"field": "country.keyword", "value": "Germany"}
                    ],
                    "size": 100
                }
        use_template: Whether to use the search template (True) or direct query (False)
        
    Returns:
        Tuple of (search results as DataFrame, total count)
    """
    try:
        # Ensure size parameter exists
        if "size" not in params:
            params["size"] = 100
            
        # Print the exact parameters being used for debugging
        print("\n=== SEARCH PARAMETERS ===")
        print(json.dumps(params, indent=2))
        
        # Initialize Elasticsearch client from helper module
        from helper import init_elastic_client
        es = init_elastic_client()
        
        if use_template:
            # Register or update the template
            register_search_template(es, force_overwrite=True)
            
            # Prepare parameters for the template
            template_params = prepare_template_params(params)
            
            # Using search template
            query_body = {
                "id": TEMPLATE_ID,
                "params": template_params
            }
            
            print("\n=== USING SEARCH TEMPLATE ===")
            print(json.dumps(query_body, indent=2))
            
            # Execute search with the template
            response = es.search_template(
                index=INDEX_NAME,
                body=query_body
            )
        else:
            # Using direct query
            query = build_direct_query(params)
            
            print("\n=== USING DIRECT QUERY ===")
            print(json.dumps(query, indent=2))
            
            # Execute search with direct query
            response = es.search(
                index=INDEX_NAME,
                body=query
            )
        
        # Extract search hits and total count
        hits = response["hits"]["hits"]
        total = response["hits"]["total"]["value"]
        
        # Convert hits to a Pandas DataFrame
        if hits:
            df = pd.DataFrame([hit["_source"] for hit in hits])
            print(f"\nFound {total} matches, returning {len(hits)} results")
            
            # Debug: Print filter fields to verify filtering
            if "filters" in params:
                for filter_item in params["filters"]:
                    field_name = filter_item["field"].split('.')[0]  # Remove .keyword suffix
                    if field_name in df.columns:
                        unique_values = df[field_name].unique()
                        print(f"{field_name} values in results: {unique_values}")
                        
                        expected_value = filter_item["value"]
                        all_match = all(value == expected_value for value in df[field_name])
                        print(f"All results match filter '{field_name}={expected_value}': {all_match}")
            
            return df, total
        else:
            print(f"\nNo results found")
            return pd.DataFrame(), 0
            
    except Exception as e:
        print(f"Error executing search: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), 0

# For backward compatibility
def execute_template_search(template_params: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
    """
    Execute a search using the search template (legacy function).
    
    Args:
        template_params: Parameters for the search template.
        
    Returns:
        Tuple of (search results as DataFrame, total count)
    """
    return execute_search(template_params, use_template=True)