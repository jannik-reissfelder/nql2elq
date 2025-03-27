"""
Natural Language to Elasticsearch Query Application

This application provides a simple interface for converting natural language queries
to Elasticsearch queries using a search template.
"""
import os
import json
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
import asyncio
from openai import AsyncOpenAI

from helper import init_elastic_client, get_openai_secret
from search_template import SEARCH_TEMPLATE, execute_search
from location import run_agent_async as location_run_agent_async

# Constants
TEMPLATE_ID = "nlq_search_template"
INDEX_NAME = "webai*"

# Initialize Elasticsearch client
es = init_elastic_client()

# Initialize OpenAI client
try:
    # Try to get the API key from AWS Secrets Manager
    openai_secret = get_openai_secret("openai/app_analyst_tool_playground")
    os.environ["OPENAI_API_KEY"] = openai_secret["key"]
except Exception as e:
    print(f"Warning: Could not retrieve OpenAI API key from AWS Secrets Manager: {e}")
    print("Please set the OPENAI_API_KEY environment variable manually.")
    # Check if the API key is already set in the environment
    if "OPENAI_API_KEY" not in os.environ:
        # Prompt for API key if running in CLI mode
        if __name__ == "__main__":
            api_key = input("Enter your OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = api_key

# Initialize shared OpenAI client
openai_client = AsyncOpenAI()

async def extract_search_parameters_async(query: str) -> Dict[str, Any]:
    """
    Extract structured search parameters from a natural language query using OpenAI function calling.
    
    Args:
        query: The natural language query to extract parameters from
        
    Returns:
        A dictionary with the extracted search parameters
    """
    # Define the function schema for parameter extraction
    function_schema = {
        "name": "extract_search_parameters",
        "description": "Extract structured search parameters from a natural language query",
        "parameters": {
            "type": "object",
            "properties": {
                "must_include_all": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords that must all be present in the search results"
                },
                "must_atleast_one_of": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords where at least one must be present in the search results"
                },
                "must_not_include": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords that must not be present in the search results"
                }
            },
            "required": []
        }
    }
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are a search parameter extraction assistant specialized in converting natural language queries into structured search parameters.

For each query, carefully analyze the intent and extract parameters into these categories:

1. must_include_all: Keywords that MUST ALL be present in the results (AND logic)
2. must_atleast_one_of: Keywords where AT LEAST ONE must be present (OR logic) - use this for synonyms, alternatives, or related concepts
3. must_not_include: Keywords that should NOT be present in the results (NOT logic) - use this for exclusions or negative filters

Be thorough in your analysis and make sure to populate all relevant parameter types."""},
                {"role": "user", "content": query}
            ],
            functions=[function_schema],
            function_call={"name": "extract_search_parameters"}
        )
        
        function_call = response.choices[0].message.function_call
        function_args = json.loads(function_call.arguments)
        
        params = {
            "must_include_all": function_args.get("must_include_all", []),
            "must_atleast_one_of": function_args.get("must_atleast_one_of", []),
            "must_not_include": function_args.get("must_not_include", [])
        }
        
        return params
    except Exception as e:
        print(f"Error extracting search parameters: {e}")
        return {
            "must_include_all": [],
            "must_atleast_one_of": [],
            "must_not_include": []
        }

async def extract_location_parameters_async(query: str) -> Dict[str, Any]:
    """
    Extract location parameters from a natural language query.
    
    Args:
        query: The natural language query to extract location from
        
    Returns:
        A dictionary with location parameters
    """
    try:
        location_data = await location_run_agent_async(query)
        return location_data
    except Exception as e:
        print(f"Error extracting location parameters: {e}")
        return {"country": None, "state": None, "region": None, "district": None, "continent": None}

async def enhance_query_async(query: str) -> str:
    """
    Enhance a natural language query with additional relevant keywords based on industry context.
    
    Args:
        query: The original natural language query
        
    Returns:
        Enhancement details as a string
    """
    system_prompt = """
    You are an expert in business intelligence and industry classification. Your task is to:
    
    1. Analyze the user's search query about companies
    2. Identify the industry or sector the user is interested in
    3. Expand the query with additional relevant keywords that would help find companies in that industry
    4. Provide a brief analysis of what defines the core of this search, alternative terms that could be relevant,
       and any industry-specific technical terms
    
    Important: ignore any geographic filters like country, city or state names that may be in the original query. This step is all about enhancing the query with relevant keywords. 
        
    Respond with:
    A brief analysis of important keywords - core terms, alternatives, and industry-specific terms.
    Do not use any geographic keywords in the response, it is not relevant.
    """
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Enhance this company search query: {query}"}
            ]
        )
        
        # Extract the response text
        response_text = response.choices[0].message.content
        return response_text
        
    except Exception as e:
        print(f"Error enhancing query: {e}")
        return f"Error enhancing query: {e}"

def prepare_template_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare parameters for the search template.
    
    Args:
        params: Extracted parameters
        
    Returns:
        Parameters formatted for the search template
    """
    template_params = {
        "must_include_all": [],
        "must_atleast_one_of": [],
        "must_not_include": [],
        "filters": [],
        "size": 100
    }
    
    # Add must_include_all terms with comma flag for proper JSON formatting
    if params["must_include_all"]:
        for i, term in enumerate(params["must_include_all"]):
            item = {"value": term}
            if i < len(params["must_include_all"]) - 1:
                item["comma"] = True
            template_params["must_include_all"].append(item)
    
    # Add must_atleast_one_of terms with comma flag
    if params["must_atleast_one_of"]:
        for i, term in enumerate(params["must_atleast_one_of"]):
            item = {"value": term}
            if i < len(params["must_atleast_one_of"]) - 1:
                item["comma"] = True
            template_params["must_atleast_one_of"].append(item)
    
    # Add must_not_include terms with comma flag
    if params["must_not_include"]:
        for i, term in enumerate(params["must_not_include"]):
            item = {"value": term}
            if i < len(params["must_not_include"]) - 1:
                item["comma"] = True
            template_params["must_not_include"].append(item)
    
    # Add geographic filters to the filters array
    filters = []
    
    # Add continent filter if provided
    if params.get("continent"):
        filters.append({
            "field": "continent.keyword",
            "value": params["continent"]
        })
    
    # Add country filter if provided
    if params.get("country"):
        filters.append({
            "field": "country.keyword",
            "value": params["country"]
        })
    
    # Add state filter if provided
    if params.get("state"):
        filters.append({
            "field": "state.keyword",
            "value": params["state"]
        })
    
    # Add region filter if provided
    if params.get("region"):
        filters.append({
            "field": "region.keyword",
            "value": params["region"]
        })
    
    # Add district filter if provided
    if params.get("district"):
        filters.append({
            "field": "district.keyword",
            "value": params["district"]
        })
    
    # Add comma flags to filters for proper JSON formatting
    for i, filter_item in enumerate(filters):
        if i < len(filters) - 1:
            filter_item["comma"] = True
    
    template_params["filters"] = filters
    
    return template_params

def execute_template_search(template_params: Dict[str, Any], create_template: bool = False) -> Tuple[List[Dict[str, Any]], int]:
    """
    Execute a search using the search template.
    
    Args:
        template_params: Parameters for the search template
        create_template: Whether to create/update the template before searching (default: False)
        
    Returns:
        Tuple of (search results, total count)
    """
    try:
        # Print the exact parameters being used for debugging
        print("\n=== SEARCH TEMPLATE PARAMETERS ===")
        print(json.dumps(template_params, indent=2))
        
        # Execute the search using the search_template module
        df, total_count = execute_search(template_params, use_template=True, create_template=create_template)
        
        # If we have results, convert the DataFrame back to the expected format
        if not df.empty:
            # Convert DataFrame to list of hits with _source and _score
            hits = []
            for _, row in df.iterrows():
                # Create a hit dictionary with _source containing all fields
                hit = {"_source": {}, "_score": 1.0}  # Default score
                
                # Add all columns to _source except score
                for col in df.columns:
                    if col != 'score':
                        hit["_source"][col] = row[col]
                
                # Add score if it exists
                if 'score' in df.columns:
                    hit["_score"] = row['score']
                
                hits.append(hit)
            
            return hits, total_count
        else:
            return [], 0
            
    except Exception as e:
        print(f"Error executing template search: {e}")
        import traceback
        traceback.print_exc()
        return [], 0

async def natural_language_to_elasticsearch_query_async(query: str, create_template: bool = False, progress_callback: Optional[Callable[[str], None]] = None) -> Tuple[pd.DataFrame, Dict[str, Any], int, str]:
    """
    Convert a natural language query to an Elasticsearch query and execute it with parallel processing.
    
    Args:
        query: The natural language query to convert
        create_template: Whether to create/update the Elasticsearch template (default: False)
        progress_callback: Optional callback function to report progress
        
    Returns:
        Tuple of (results DataFrame, extracted parameters, total count, enhancement details)
    """
    try:
        # Initial progress update
        if progress_callback:
            progress_callback("Discovering your market...")
            
        # Create async tasks for operations that can run in parallel
        enhance_task = asyncio.create_task(enhance_query_async(query))
        location_task = asyncio.create_task(extract_location_parameters_async(query))
        
        # Update progress after initiating tasks
        if progress_callback:
            progress_callback("Understanding your market...")
            
        # Await the enhancement result
        enhancement_details = await enhance_task
        
        # Update progress before keyword extraction
        if progress_callback:
            progress_callback("Analyzing keywords...")
            
        # Start keyword extraction (depends on enhancement_details)
        keyword_params = await extract_search_parameters_async(enhancement_details)
        
        # Update progress before location processing
        if progress_callback:
            progress_callback("Processing location data...")
            
        # Get location parameters
        location_params = await location_task
        
        # Combine parameters
        params = {
            **keyword_params,
            "continent": location_params.get("continent"),
            "country": location_params.get("country"),
            "state": location_params.get("state"),
            "region": location_params.get("region"),
            "district": location_params.get("district")
        }
        
        # Update progress before search execution
        if progress_callback:
            progress_callback("Fetching your market results...")
            
        # Continue with existing flow
        template_params = prepare_template_params(params)
        results, total_count = execute_template_search(template_params, create_template=create_template)
        
        # Process results
        data = []
        for hit in results:
            row = {'score': hit['_score']}
            for field, value in hit['_source'].items():
                row[field] = value
            data.append(row)
        
        results_df = pd.DataFrame(data)
        results_df['enhancement_details'] = enhancement_details
        
        return results_df, params, total_count, enhancement_details
    
    except Exception as e:
        print(f"Error in natural language query processing: {e}")
        import traceback
        traceback.print_exc()
        # Return empty results
        return pd.DataFrame(), {}, 0, f"Error: {e}"

def natural_language_to_elasticsearch_query(query: str, create_template: bool = False, progress_callback: Optional[Callable[[str], None]] = None) -> Tuple[pd.DataFrame, Dict[str, Any], int, str]:
    """
    Convert a natural language query to an Elasticsearch query and execute it.
    This function uses the async implementation for better performance.
    
    Args:
        query: The natural language query to convert
        create_template: Whether to create/update the Elasticsearch template (default: False)
        progress_callback: Optional callback function to report progress
        
    Returns:
        Tuple of (results DataFrame, extracted parameters, total count, enhancement details)
    """
    return asyncio.run(natural_language_to_elasticsearch_query_async(query, create_template=create_template, progress_callback=progress_callback))

def explain_query(params: Dict[str, Any]) -> str:
    """
    Generate a human-readable explanation of the search query.
    
    Args:
        params: The extracted search parameters
        
    Returns:
        A string explaining the query
    """
    explanation = "Searching for "
    
    # Add must_include_all terms
    if params["must_include_all"]:
        explanation += "entries that contain ALL of these terms: "
        explanation += ", ".join([f"'{term}'" for term in params["must_include_all"]])
        explanation += ". "
    else:
        explanation += "all entries "
    
    # Add must_atleast_one_of terms
    if params["must_atleast_one_of"]:
        explanation += "That contain AT LEAST ONE of these terms: "
        explanation += ", ".join([f"'{term}'" for term in params["must_atleast_one_of"]])
        explanation += ". "
    
    # Add must_not_include terms
    if params["must_not_include"]:
        explanation += "Excluding entries that contain ANY of these terms: "
        explanation += ", ".join([f"'{term}'" for term in params["must_not_include"]])
        explanation += ". "
    
    # Add geographic filters
    if params.get("continent"):
        explanation += f"Located in continent: {params['continent']}. "
    
    if params.get("country"):
        explanation += f"Located in country: {params['country']}. "
    
    if params.get("state"):
        explanation += f"Located in state/province: {params['state']}. "
    
    if params.get("region"):
        explanation += f"Located in region: {params['region']}. "
    
    if params.get("district"):
        explanation += f"Located in district: {params['district']}. "
    
    return explanation

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        
        # Only create the template once when running from command line
        # Set to True for the first run, then set back to False
        create_template = False
        
        results_df, params, total_count, enhancement_details = natural_language_to_elasticsearch_query(query, create_template=create_template)
        
        print("\nQuery Interpretation:")
        print(explain_query(params))
        
        print("\nExtracted Parameters:")
        print(json.dumps(params, indent=2))
        
        print("\nQuery Enhancement Details:")
        print(enhancement_details)
        
        print(f"\nSearch Results ({total_count} total):")
        if not results_df.empty:
            print(results_df.shape)
        else:
            print("No results found.")
    else:
        print("Please provide a query as a command line argument.")
        print("Example: python nlq2elastic.py 'Find packaging companies in Germany'")
