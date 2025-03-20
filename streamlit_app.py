"""
Streamlit App for Natural Language to Elasticsearch Query

This file is specifically designed to be run with `streamlit run streamlit_app.py`
"""
import os
import json
import pandas as pd
import streamlit as st
from typing import Dict, List, Any, Tuple
from openai import OpenAI

# Import functions from nlq2elq_app.py
from nlq2elq_app import (
    extract_search_parameters,
    extract_location_parameters,
    prepare_template_params,
    execute_template_search,
    natural_language_to_elasticsearch_query,
    explain_query,
    register_template,
    enhance_query
)

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
</style>
""", unsafe_allow_html=True)

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
        
        # Query enhancement option
        st.subheader("Search Options")
        enhance_search = st.checkbox("Enhance Search Query", value=True, 
                                    help="Use AI to expand your query with industry-specific keywords")
        
        # Template registration
        st.subheader("Search Template")
        if st.button("Register Search Template"):
            with st.spinner("Registering template..."):
                try:
                    success = register_template()
                    if success:
                        st.success("Template registered successfully!")
                    else:
                        st.error("Failed to register template.")
                except Exception as e:
                    st.error(f"Error registering template: {str(e)}")
        
        # Information about location extraction
        st.subheader("Location Extraction")
        st.info("The application now automatically extracts location information from your query using advanced natural language processing.")
    
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
                    with st.spinner("Processing query..."):
                        # Use the new integrated workflow
                        results_df, params, total_count = natural_language_to_elasticsearch_query(query)
                        
                        # Store results in session state
                        st.session_state.results_df = results_df
                        st.session_state.params = params
                        st.session_state.total_count = total_count
                        st.session_state.query = query
                        
                        # Extract location information for display
                        location_params = extract_location_parameters(query)
                        st.session_state.location_params = location_params
                        
                        # Display filter verification
                        if results_df.empty:
                            st.warning("No results found for your query.")
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.exception(e)
    
    with col2:
        st.subheader("Query Interpretation")
        if 'params' in st.session_state:
            st.info(explain_query(st.session_state.params))
            
            # Show enhancement details if available
            if 'enhancement_details' in st.session_state:
                st.subheader("Query Enhancement")
                
                with st.expander("View enhancement details", expanded=True):
                    st.markdown(st.session_state.enhancement_details)
            
            # Show extracted location information
            if 'location_params' in st.session_state:
                st.subheader("Extracted Location")
                location_info = st.session_state.location_params
                
                # Create a clean display of location information
                location_display = {}
                for key, value in location_info.items():
                    if value is not None and key != 'location':
                        location_display[key] = value
                
                if location_display:
                    st.json(location_display)
                else:
                    st.info("No location information detected in query.")
            
            # Display filter groups with visual tags
            st.subheader("Filter Groups")
            
            # Must Include All - Green tags
            if st.session_state.params.get("must_include_all"):
                st.markdown("**Must Have All**")
                tags_html = ""
                for term in st.session_state.params["must_include_all"]:
                    tags_html += f'<span style="background-color: #28a745; color: white; padding: 0.3rem 0.6rem; margin: 0.2rem; border-radius: 1rem; display: inline-block;">{term} <span style="font-weight: bold;">×</span></span>'
                st.markdown(f'<div style="margin-bottom: 1rem;">{tags_html}</div>', unsafe_allow_html=True)
            
            # Must Include At Least One - Blue tags
            if st.session_state.params.get("must_atleast_one_of"):
                st.markdown("**Must Have at least One**")
                tags_html = ""
                for term in st.session_state.params["must_atleast_one_of"]:
                    tags_html += f'<span style="background-color: #17a2b8; color: white; padding: 0.3rem 0.6rem; margin: 0.2rem; border-radius: 1rem; display: inline-block;">{term} <span style="font-weight: bold;">×</span></span>'
                st.markdown(f'<div style="margin-bottom: 1rem;">{tags_html}</div>', unsafe_allow_html=True)
            
            # Must Not Include - Red tags
            if st.session_state.params.get("must_not_include"):
                st.markdown("**Must Not Include**")
                tags_html = ""
                for term in st.session_state.params["must_not_include"]:
                    tags_html += f'<span style="background-color: #dc3545; color: white; padding: 0.3rem 0.6rem; margin: 0.2rem; border-radius: 1rem; display: inline-block;">{term} <span style="font-weight: bold;">×</span></span>'
                st.markdown(f'<div style="margin-bottom: 1rem;">{tags_html}</div>', unsafe_allow_html=True)
            
            # Add a dummy text input for additional keywords (for visual similarity to the image)
            st.markdown('<div style="margin-top: 1rem; margin-bottom: 1rem; opacity: 0.6;">', unsafe_allow_html=True)
            st.text_input("Add optional keywords", "", key="dummy_keywords", disabled=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show raw parameters in expandable section
            with st.expander("View Raw Parameters"):
                st.json(st.session_state.params)
    
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
