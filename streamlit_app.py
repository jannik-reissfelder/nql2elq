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
            
            # Display filter groups with visual tags
            st.subheader("Extracted Parameters")
            
            # Location Parameters - Orange tags
            if st.session_state.get('location_params'):
                st.markdown("**Location Parameters**")
                tags_html = ""
                for key, value in st.session_state.location_params.items():
                    if value:
                        tags_html += f'<span style="background-color: #fd7e14; color: white; padding: 0.3rem 0.6rem; margin: 0.2rem; border-radius: 1rem; display: inline-block;">{key.capitalize()}: {value}</span>'
                st.markdown(f'<div style="margin-bottom: 1rem;">{tags_html}</div>', unsafe_allow_html=True)
            
            # Must Include All - Green tags
            if st.session_state.params.get("must_include_all"):
                st.markdown("**Must Have All**")
                tags_html = ""
                for term in st.session_state.params["must_include_all"]:
                    tags_html += f'<span style="background-color: #28a745; color: white; padding: 0.3rem 0.6rem; margin: 0.2rem; border-radius: 1rem; display: inline-block;">{term}</span>'
                st.markdown(f'<div style="margin-bottom: 1rem;">{tags_html}</div>', unsafe_allow_html=True)
            
            # Must Include At Least One - Blue tags
            if st.session_state.params.get("must_atleast_one_of"):
                st.markdown("**Must Have at least One**")
                tags_html = ""
                for term in st.session_state.params["must_atleast_one_of"]:
                    tags_html += f'<span style="background-color: #17a2b8; color: white; padding: 0.3rem 0.6rem; margin: 0.2rem; border-radius: 1rem; display: inline-block;">{term}</span>'
                st.markdown(f'<div style="margin-bottom: 1rem;">{tags_html}</div>', unsafe_allow_html=True)
            
            # Must Not Include - Red tags
            if st.session_state.params.get("must_not_include"):
                st.markdown("**Must Not Include**")
                tags_html = ""
                for term in st.session_state.params["must_not_include"]:
                    tags_html += f'<span style="background-color: #dc3545; color: white; padding: 0.3rem 0.6rem; margin: 0.2rem; border-radius: 1rem; display: inline-block;">{term}</span>'
                st.markdown(f'<div style="margin-bottom: 1rem;">{tags_html}</div>', unsafe_allow_html=True)
            
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
