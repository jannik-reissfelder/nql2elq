"""
Natural Language to Elasticsearch Query Application

This application provides a simple interface for converting natural language queries
to Elasticsearch queries using a search template.
"""
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
from elasticsearch import Elasticsearch
import os
from openai import OpenAI

from helper import init_elastic_client
from search_template import SEARCH_TEMPLATE, execute_search
from helper import get_openai_secret
# Import location module
from location import run_agent as location_run_agent

# Define the template ID and index name
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
    # Check if the API key is already set in the environment
    if "OPENAI_API_KEY" not in os.environ:
        # Prompt for API key if running in CLI mode
        if not hasattr(st, 'session_state'):  # Not running in Streamlit
            api_key = input("Please enter your OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = api_key

# Create the OpenAI client
client = OpenAI()

def register_template():
    """Register the search template with Elasticsearch"""
    try:
        # Delete the template if it already exists
        try:
            es.delete_script(id=TEMPLATE_ID)
            print(f"Deleted existing template: {TEMPLATE_ID}")
        except Exception as e:
            # Template doesn't exist yet, which is fine
            pass
        
        # Register the template
        es.put_script(id=TEMPLATE_ID, body=SEARCH_TEMPLATE)
        print(f"Created search template: {TEMPLATE_ID}")
        
        # Verify the template was registered
        template = es.get_script(id=TEMPLATE_ID)
        if template:
            return True
        else:
            return False
    
    except Exception as e:
        print(f"Error registering template: {e}")
        return False

def extract_search_parameters(query: str) -> Dict[str, Any]:
    """
    Extract search parameters from a natural language query using OpenAI's function calling.
    
    Args:
        query: The natural language query or enhancement details
        
    Returns:
        A dictionary of extracted parameters
    """
    functions = [
        {
            "type": "function",
            "function": {
                "name": "extract_search_parameters",
                "description": "Extract search parameters from a natural language query for Elasticsearch",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "must_include_all": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keywords that MUST ALL be present in the results"
                        },
                        "must_atleast_one_of": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keywords where AT LEAST ONE must be present in the results"
                        },
                        "must_not_include": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keywords that must NOT be present in the results"
                        }
                        # Location properties removed as they will be handled by location.py
                    },
                    "required": []
                }
            }
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use an appropriate model
            messages=[
                {"role": "system", "content": "You are a search query analyzer that extracts structured parameters from natural language queries. Focus on identifying keywords that should be categorized as must_include_all, must_atleast_one_of, or must_not_include."},
                {"role": "user", "content": query}
            ],
            tools=functions,
            tool_choice={"type": "function", "function": {"name": "extract_search_parameters"}}
        )
        
        # Extract the function call arguments
        function_args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        
        # Ensure all expected keys exist in the result (location keys removed)
        params = {
            "must_include_all": function_args.get("must_include_all", []),
            "must_atleast_one_of": function_args.get("must_atleast_one_of", []),
            "must_not_include": function_args.get("must_not_include", [])
        }
        
        return params
    except Exception as e:
        print(f"Error extracting parameters with OpenAI: {e}")
        # Return empty parameters if there's an error (location keys removed)
        return {
            "must_include_all": [],
            "must_atleast_one_of": [],
            "must_not_include": []
        }

def extract_location_parameters(query: str) -> Dict[str, Any]:
    """
    Extract location parameters from a natural language query using location.py
    
    Args:
        query: The natural language query
        
    Returns:
        A dictionary of extracted location parameters
    """
    try:
        # Get location data from location.py
        location_data = location_run_agent(query)
        return location_data
    except Exception as e:
        print(f"Error extracting location parameters: {e}")
        # Return empty location parameters if there's an error
        return {
            "location": None,
            "country": None,
            "state": None,
            "region": None,
            "district": None
        }

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

def execute_template_search(template_params: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Execute a search using the search template.
    
    Args:
        template_params: Parameters for the search template
        
    Returns:
        Tuple of (search results, total count)
    """
    try:
        # Print the exact parameters being used for debugging
        print("\n=== SEARCH TEMPLATE PARAMETERS ===")
        print(json.dumps(template_params, indent=2))
        
        # Import the execute_search function from search_template
        from search_template import execute_search
        
        # Execute the search using the search_template module
        df, total_count = execute_search(template_params, use_template=True)
        
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

def natural_language_to_elasticsearch_query(query: str) -> Tuple[pd.DataFrame, Dict[str, Any], int]:
    """
    Convert a natural language query to an Elasticsearch query and execute it.
    
    Args:
        query: The natural language query
        
    Returns:
        Tuple of (results DataFrame, extracted parameters, total count)
    """
    # Step 1: Enhance the query
    enhanced_query, enhancement_details = enhance_query(query)
    
    # Step 2: Extract location parameters from the original query
    location_params = extract_location_parameters(query)
    
    # Step 3: Extract keyword parameters from the enhancement details
    keyword_params = extract_search_parameters(enhancement_details)
    
    # Step 4: Combine parameters
    params = {
        **keyword_params,
        "country": location_params.get("country"),
        "state": location_params.get("state"),
        "region": location_params.get("region"),
        "district": location_params.get("district")
    }
    
    # Step 5: Prepare parameters for the template
    template_params = prepare_template_params(params)
    
    # Step 6: Execute the search
    results, total_count = execute_template_search(template_params)
    
    # Step 7: Convert results to DataFrame
    data = []
    for hit in results:
        # Extract score
        row = {'score': hit['_score']}
        
        # Add all fields from _source
        for field, value in hit['_source'].items():
            row[field] = value
        
        data.append(row)
    
    results_df = pd.DataFrame(data)
    
    # Step 8: Return results
    return results_df, params, total_count

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
    if params["country"]:
        explanation += f"Located in country: {params['country']}. "
    
    if params["state"]:
        explanation += f"Located in state/province: {params['state']}. "
    
    if params["region"]:
        explanation += f"Located in region: {params['region']}. "
    
    if params["district"]:
        explanation += f"Located in district: {params['district']}. "
    
    return explanation

def enhance_query(query: str) -> Tuple[str, str]:
    """
    Enhance a natural language query with additional relevant keywords based on industry context.
    
    Args:
        query: The original natural language query
        
    Returns:
        Tuple of (enhanced query, enhancement details)
    """
    system_prompt = """
    You are an expert in business intelligence and industry classification. Your task is to:
    
    1. Analyze the user's search query about companies
    2. Identify the industry or sector the user is interested in
    3. Expand the query with additional relevant keywords that would help find companies in that industry
    4. Provide a brief analysis of what defines the core of this search, alternative terms that could be relevant,
       and any industry-specific technical terms
    
    Focus on B2B industries, manufacturing, and industrial sectors. Do not modify any geographic filters like country or state names that may be in the original query.
    
    Respond with:
    1. Enhanced Query: [An enhanced version of the query that includes additional relevant keywords]
    2. Industry: [The identified industry/sector]
    3. Analysis: [A brief analysis of important keywords - core terms, alternatives, and industry-specific terms]
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Enhance this company search query: {query}"}
            ]
        )
        
        # Extract the response text
        response_text = response.choices[0].message.content
        
        # Parse the response to extract the enhanced query and details
        enhanced_query = query  # Default to original query if parsing fails
        
        # Try to extract the enhanced query from the response
        if "Enhanced Query:" in response_text:
            parts = response_text.split("Enhanced Query:", 1)[1].split("\n", 1)
            if parts:
                enhanced_query = parts[0].strip()
                # Remove any markdown formatting
                enhanced_query = enhanced_query.replace("[", "").replace("]", "").strip()
        
        return enhanced_query, response_text
        
    except Exception as e:
        print(f"Error enhancing query: {e}")
        return query, f"Error enhancing query: {str(e)}"


def run_streamlit_app():
    """Run the Streamlit app"""
    st.title("Natural Language to Elasticsearch Query")
    st.write("Enter a natural language query to search the Elasticsearch index.")
    
    # API key input (only shown if not already set)
    if "OPENAI_API_KEY" not in os.environ:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API key set successfully!")
    
    # Input for natural language query
    query = st.text_input("Enter your query:", "Show me packaging machine companies in Germany")
    
    # Register template button
    if st.button("Register Search Template"):
        with st.spinner("Registering template..."):
            success = register_template()
            if success:
                st.success("Template registered successfully!")
            else:
                st.error("Failed to register template.")
    
    # Search button
    if st.button("Search") or query:
        if "OPENAI_API_KEY" not in os.environ:
            st.error("Please enter your OpenAI API key first.")
            return
            
        with st.spinner("Searching..."):
            enhanced_query, enhancement_details = enhance_query(query)
            results_df, params, total_count = natural_language_to_elasticsearch_query(enhanced_query)
            
            # Display explanation
            st.subheader("Query Interpretation")
            st.write(explain_query(params))
            
            # Display extracted parameters
            st.subheader("Extracted Parameters")
            st.json(params)
            
            # Display enhancement details
            st.subheader("Query Enhancement Details")
            st.json(enhancement_details)
            
            # Display results
            st.subheader(f"Search Results ({total_count} total)")
            if not results_df.empty:
                st.dataframe(results_df)
            else:
                st.write("No results found.")

def run_cli_app():
    """Run the command-line interface app"""
    print("\n===== Natural Language to Elasticsearch Query =====")
    print("This tool converts natural language queries to Elasticsearch queries.")
    
    # Check if template is registered
    try:
        es.get_script(id=TEMPLATE_ID)
        print(f"Template '{TEMPLATE_ID}' is already registered.")
    except:
        print(f"Template '{TEMPLATE_ID}' is not registered.")
        register = input("Would you like to register it now? (y/n): ")
        if register.lower() == 'y':
            success = register_template()
            if success:
                print("Template registered successfully!")
            else:
                print("Failed to register template.")
                return
    
    while True:
        # Get query from user
        query = input("\nEnter your query (or 'quit' to exit): ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        # Process query
        print("\nProcessing query...")
        enhanced_query, enhancement_details = enhance_query(query)
        results_df, params, total_count = natural_language_to_elasticsearch_query(enhanced_query)
        
        # Display explanation
        print("\nQuery Interpretation:")
        print(explain_query(params))
        
        # Display extracted parameters
        print("\nExtracted Parameters:")
        print(json.dumps(params, indent=2))
        
        # Display enhancement details
        print("\nQuery Enhancement Details:")
        print(json.dumps(enhancement_details, indent=2))
        
        # Display results
        print(f"\nSearch Results ({total_count} total):")
        if not results_df.empty:
            print(results_df.head(10).to_string())
            if len(results_df) > 10:
                print(f"... and {len(results_df) - 10} more results.")
        else:
            print("No results found.")

if __name__ == "__main__":
    run_cli_app()
