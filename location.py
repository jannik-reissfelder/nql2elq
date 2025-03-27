import json
import os
from helper import get_openai_secret
import pandas as pd
from pydantic import BaseModel, Field
# Try to get the API key from AWS Secrets Manager
openai_secret = get_openai_secret("openai/app_analyst_tool_playground")
os.environ["OPENAI_API_KEY"] = openai_secret["key"]

import json
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Updated prompt template with structured output
location_prompt = PromptTemplate(
    input_variables=["query"],
    template="""(
        "You are an assistant that extracts precise geographical details from a user query, "
        "using the conventions found in a dataset with administrative areas. "
        "This dataset has four columns: 'country', 'state', 'region', and 'district'. "
        "Countries are always recorded using their internationally recognized English names (e.g., Germany, France, Italy), "
        "while sub-national divisions (state, region, district) are recorded in their native or official local form. "
        "For instance, in Germany, the dataset uses 'Bayern' rather than 'Bavaria'; in France, it might use 'Auvergne-Rhône-Alpes' rather than an anglicized version; "
        "in China, 'Sichuan' (in pinyin) is used; and in Brazil, you might see 'São Paulo' as is. \n\n"
        "When processing the query, consider that users might be vague, have spelling errors, or provide incomplete information. "
        "Use your best judgment to extract the intended location along with any available geographical hints. "
        "If a field is not clearly provided, return its value as null. \n\n"
        "IMPORTANT: If the query does not mention any location at all, set location_requested to false and all location fields to null. "
        "Only set location_requested to true if the user is specifically asking about companies or entities in a particular location. \n\n"
        "Return the result as a JSON object with the keys: 'location_requested', 'continent', 'country', 'state', 'region', and 'district'. "
        "Ensure that the 'continent' and 'country' are always in English, and the other keys are returned in the native language as found in the dataset. \n\n"
        "Examples:\n"
        "1. Input: 'Show me all companies in München'\n"
        "   Expected Output: {{\"location_requested\": true, \"continent\": \"Europe\", \"country\": \"Germany\", \"state\": \"Bayern\", \"region\": \"München\", \"district\": null}}\n\n"
        "2. Input: 'Liste des entreprises à Moulins'\n"
        "   Expected Output: {{\"location_requested\": true, \"continent\": \"Europe\", \"country\": \"France\", \"state\": \"Auvergne-Rhône-Alpes\", \"region\": \"Allier\", \"district\": null}}\n\n"
        "3. Input: 'Find businesses in Batdâmbâng'\n"
        "   Expected Output: {{\"location_requested\": true, \"continent\": \"Asia\", \"country\": \"Cambodia\", \"state\": null, \"region\": null, \"district\": null}}\n\n"
        "4. Input: 'Which companies specialize in warehouse automation and robotics?'\n"
        "   Expected Output: {{\"location_requested\": false, \"continent\": null, \"country\": null, \"state\": null, \"region\": null, \"district\": null}}\n\n"
        "Query: {query}\n"
        "Please only give back the dictionary, no additional explanations needed"
    )"""
)

# Define structured output schema using Pydantic
class LocationSchema(BaseModel):
    location_requested: bool = Field(description="Whether the query specifically requests information about a location")
    continent: str = Field(description="Continent name in English. Valid answers are North America, South America, Europe, Asia, Africa, Antarctica, Australia, and Oceania", default=None)
    country: str = Field(description="Country name in English", default=None)
    state: str = Field(description="State/province in native language", default=None)
    region: str = Field(description="Region in native language", default=None)
    district: str = Field(description="District in native language", default=None)

# Initialize LLM with proper JSON configuration
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    model_kwargs={
        "response_format": {"type": "json_object"}  # Correct parameter location
    }
).with_structured_output(LocationSchema)


# Create modern runnable pipeline
runnable_chain = (
    RunnablePassthrough.assign(query=lambda x: x["query"])
    | location_prompt
    | llm
    # | JsonOutputParser()
)

def run_agent(query):
    try:
        # Modern invocation pattern
        result = runnable_chain.invoke({"query": query})

        return {
            "location_requested": result.location_requested,
            "continent": result.continent,
            "country": result.country,
            "state": result.state,
            "region": result.region,
            "district": result.district
        }
    except json.JSONDecodeError:
        return {
            "location_requested": False,
            "continent": None,
            "country": None,
            "state": None,
            "region": None,
            "district": None
        }

async def run_agent_async(query: str) -> dict:
    """
    Asynchronous version of the location extraction agent.
    
    Args:
        query: The natural language query to extract location from
        
    Returns:
        A dictionary with location parameters
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    
    chain = (
        {"query": RunnablePassthrough()} 
        | PromptTemplate.from_template(location_prompt.template) 
        | llm.with_structured_output(LocationSchema)
    )
    
    try:
        location_data = await chain.ainvoke({"query": query})
        return location_data.dict()
    except Exception as e:
        print(f"Error extracting location parameters: {e}")
        return {"location_requested": False, "continent": None, "country": None, "state": None, "region": None, "district": None}

def extract_location_parameters(query):
    try:
        location_data = run_agent(query)
        return location_data
    except Exception as e:
        print(f"Error extracting location parameters: {e}")
        return {"location_requested": False, "continent": None, "country": None, "state": None, "region": None, "district": None}

# Usage remains the same
user_query = "Finde Unternehmen in Baden-Württemberg"
final_result = run_agent(user_query)
