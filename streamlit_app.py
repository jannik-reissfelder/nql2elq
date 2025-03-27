"""
Streamlit App for Natural Language to Elasticsearch Query

This file is specifically designed to be run with `streamlit run streamlit_app.py`
"""
import os
import json
import pandas as pd
import streamlit as st
from typing import Dict, List, Any, Tuple

# Import functions from nlq2elastic.py
from nlq2elastic import (
    natural_language_to_elasticsearch_query,
    prepare_template_params
)

# Import functions from search_template.py for direct search execution
from search_template import execute_search

# Set page configuration
st.set_page_config(
    page_title="NLQ to Elasticsearch",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add some basic styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    @keyframes pulse {
        0% { opacity: 0.3; }
        50% { opacity: 1; }
        100% { opacity: 0.3; }
    }
    .loading-animation {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin: 20px 0;
        text-align: center;
    }
    .loading-animation span {
        display: inline-block;
        animation: pulse 1.5s infinite;
        margin-right: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

def create_loading_animation(container):
    """Create a continuous loading animation that keeps moving until cleared"""
    message = "Discovering your market"
    html = '<div class="loading-animation">'
    
    # Add each word with proper spacing
    words = message.split()
    for i, word in enumerate(words):
        # Add each letter with its own animation delay
        for j, char in enumerate(word):
            delay = (i * len(word) + j) * 0.1
            html += f'<span style="animation-delay: {delay}s;">{char}</span>'
        
        # Add space between words (if not the last word)
        if i < len(words) - 1:
            html += '<span>&nbsp;</span>'
    
    # Add animated dots at the end
    for i in range(3):
        delay = (len(message) + i) * 0.1
        html += f'<span style="animation-delay: {delay}s;">.</span>'
        
    html += '</div>'
    return container.markdown(html, unsafe_allow_html=True)

def animated_text(text, container):
    """Display text with a sequential letter animation"""
    html = '<div class="animated-text">'
    for i, char in enumerate(text):
        # Add increasing delay for each character
        delay = i * 0.1
        html += f'<span style="animation-delay: {delay}s;">{char}</span>'
    html += '</div>'
    container.markdown(html, unsafe_allow_html=True)

def render_editable_parameters():
    """
    Create a simple UI for editing search parameters using standard Streamlit components
    """
    st.subheader("Edit Parameters")
    
    # Must include parameters (multiselect)
    if "must_include_all" in st.session_state.working_params:
        selected_must = st.multiselect(
            "Must include ALL of these terms:",
            options=st.session_state.working_params["must_include_all"] + st.session_state.working_params.get("must_atleast_one_of", []) + st.session_state.working_params.get("must_not_include", []),
            default=st.session_state.working_params["must_include_all"],
            key="edit_must_include"
        )
        st.session_state.working_params["must_include_all"] = selected_must
    
    # Must include at least one parameters (multiselect)
    if "must_atleast_one_of" in st.session_state.working_params:
        selected_atleast = st.multiselect(
            "Must include AT LEAST ONE of these terms:",
            options=st.session_state.working_params["must_atleast_one_of"] + st.session_state.working_params.get("must_include_all", []) + st.session_state.working_params.get("must_not_include", []),
            default=st.session_state.working_params["must_atleast_one_of"],
            key="edit_must_atleast"
        )
        st.session_state.working_params["must_atleast_one_of"] = selected_atleast
    
    # Must not include parameters (multiselect)
    if "must_not_include" in st.session_state.working_params:
        selected_mustnot = st.multiselect(
            "Must NOT include these terms:",
            options=st.session_state.working_params["must_not_include"] + st.session_state.working_params.get("must_include_all", []) + st.session_state.working_params.get("must_atleast_one_of", []),
            default=st.session_state.working_params["must_not_include"],
            key="edit_must_not"
        )
        st.session_state.working_params["must_not_include"] = selected_mustnot
    
    # Location parameters (multiselect for easier removal)
    st.markdown("**Location Parameters**")
    
    # Country selection
    if st.session_state.working_params.get("country"):
        countries = [st.session_state.working_params["country"]]
        selected_country = st.multiselect(
            "Country:",
            options=countries,
            default=countries,
            key="edit_country_select"
        )
        st.session_state.working_params["country"] = selected_country[0] if selected_country else None
    
    # State selection
    if st.session_state.working_params.get("state"):
        states = [st.session_state.working_params["state"]]
        selected_state = st.multiselect(
            "State:",
            options=states,
            default=states,
            key="edit_state_select"
        )
        st.session_state.working_params["state"] = selected_state[0] if selected_state else None
    
    # Region selection
    if st.session_state.working_params.get("region"):
        regions = [st.session_state.working_params["region"]]
        selected_region = st.multiselect(
            "Region:",
            options=regions,
            default=regions,
            key="edit_region_select"
        )
        st.session_state.working_params["region"] = selected_region[0] if selected_region else None
    
    # District selection
    if st.session_state.working_params.get("district"):
        districts = [st.session_state.working_params["district"]]
        selected_district = st.multiselect(
            "District:",
            options=districts,
            default=districts,
            key="edit_district_select"
        )
        st.session_state.working_params["district"] = selected_district[0] if selected_district else None
    
    # Button to update search
    if st.button("Update Search", type="primary"):
        update_search_with_modified_params()

def update_search_with_modified_params():
    """
    Execute a search using the modified parameters without reprocessing the natural language query
    """
    try:
        # Convert parameters to template format
        template_params = prepare_template_params(st.session_state.working_params)
        
        # Execute search with modified parameters
        results_df, total_count = execute_search(
            template_params, 
            use_template=True,
            create_template=False  # Use existing template
        )
        
        # Update results in session state
        st.session_state.results_df = results_df
        st.session_state.total_count = total_count
        
        # Update display message
        st.session_state.search_message = f"🔍 Found {total_count} results with modified parameters"
        
        return True
    except Exception as e:
        st.error(f"Error updating search: {str(e)}")
        return False

def main():
    """Main Streamlit application"""
    st.title("Natural Language to Elasticsearch Query")
    st.write("Convert natural language queries to Elasticsearch search parameters")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API key input (only shown if not already set)
        if "OPENAI_API_KEY" not in os.environ:
            api_key = st.text_input("Enter your OpenAI API key:", type="password")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("API key set successfully!")
        else:
            st.success("OpenAI API key is configured")
        
        # Template registration
        st.subheader("Search Template")
        create_template = st.checkbox("Create/Update Template", value=False, 
                                    help="Create or update the Elasticsearch template (only needed once or when template changes)")
        
        # Information about location extraction
        st.subheader("About")
        st.info("This application converts natural language queries to Elasticsearch parameters and extracts location information automatically.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter your query")
        # Input for natural language query
        query = st.text_area(
            "Natural language query:", 
            "Show me packaging machine companies in Germany",
            height=100
        )
        
        search_button = st.button("Search", type="primary")
        
        if search_button:
            if "OPENAI_API_KEY" not in os.environ:
                st.error("Please enter your OpenAI API key in the sidebar first.")
            else:
                try:
                    # Create a placeholder for the animated text
                    loading_placeholder = st.empty()
                    
                    # Display continuous animation
                    create_loading_animation(loading_placeholder)
                    
                    # Process the query
                    results_df, params, total_count, enhancement_details = natural_language_to_elasticsearch_query(
                        query, 
                        create_template=create_template
                    )
                    
                    # Clear the loading animation
                    loading_placeholder.empty()
                    
                    # Store results in session state
                    st.session_state.results_df = results_df
                    st.session_state.params = params
                    st.session_state.total_count = total_count
                    st.session_state.query = query
                    st.session_state.enhancement_details = enhancement_details
                    
                    # Extract location parameters from the params dictionary
                    location_params = {
                        "country": params.get("country"),
                        "state": params.get("state"),
                        "region": params.get("region"),
                        "district": params.get("district")
                    }
                    
                    # Filter out None values
                    st.session_state.location_params = {k: v for k, v in location_params.items() if v is not None}
                    
                    # Initialize working parameters
                    st.session_state.working_params = params.copy()
                    
                    # Display filter verification
                    if results_df.empty:
                        st.warning("No results found for your query.")
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.exception(e)
    
    with col2:
        if 'params' in st.session_state:
            # Show enhancement details if available
            if 'enhancement_details' in st.session_state:
                with st.expander("View Query Enhancement Details", expanded=False):
                    st.markdown(st.session_state.enhancement_details)
            
            # Only keep the raw parameters view
            with st.expander("View Raw Parameters"):
                st.json(st.session_state.params)
            
            # Editable parameters
            render_editable_parameters()
    
    # Results section (full width)
    if 'results_df' in st.session_state:
        st.markdown("---")
        st.subheader(f"Search Results ({st.session_state.total_count} total)")
        
        if not st.session_state.results_df.empty:
            # Allow sorting and filtering
            st.dataframe(
                st.session_state.results_df,
                use_container_width=True,
                height=400
            )
        else:
            st.info("No results found for your query.")

# Run the Streamlit app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
