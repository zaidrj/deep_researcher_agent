import asyncio
import streamlit as st
from typing import Dict, Any, List

# --- Import necessary components from the 'agents' library ---
# Using a try-except block as shown in your Google example for robustness
try:
    from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
    from agents.tool import function_tool
    # set_default_openai_key is not needed for Google, so we don't import it
except ImportError as e:
    st.error(f"Failed to import necessary components from 'agents': {e}")
    st.stop() # Stop the app if essential components can't be imported

from firecrawl import FirecrawlApp

# Set page configuration
st.set_page_config(
    page_title="Google Gemini Deep Research Agent By Agenex Labs", # Updated title
    page_icon="	U+1F9BE",
    layout="wide"
)

# Initialize session state for API keys if not exists
if "google_api_key" not in st.session_state: # Changed to google_api_key
    st.session_state.google_api_key = ""
if "firecrawl_api_key" not in st.session_state:
    st.session_state.firecrawl_api_key = ""

# Sidebar for API keys
with st.sidebar:
    st.title("API Configuration")
    # Changed label and session state key
    google_api_key = st.text_input(
        "Google API Key",
        value=st.session_state.google_api_key,
        type="password"
    )
    firecrawl_api_key = st.text_input(
        "Firecrawl API Key",
        value=st.session_state.firecrawl_api_key,
        type="password"
    )

    if google_api_key:
        st.session_state.google_api_key = google_api_key
        # No equivalent to set_default_openai_key for Google via this method,
        # we'll pass configuration explicitly to Runner.run
    if firecrawl_api_key:
        st.session_state.firecrawl_api_key = firecrawl_api_key

# Main content
st.title("Google Gemini Deep Research Agent By Agenex Labs") # Updated title
st.markdown("This Agent performs deep research on any topic using Google Gemini via the `agents` library and Firecrawl") # Updated description

# Research topic input
research_topic = st.text_input("Enter your research topic:", placeholder="e.g., Latest developments in AI")

# --- Google API Configuration (based on your example) ---
# Only attempt to configure if Google API key is available
google_run_config = None
if st.session_state.google_api_key:
    try:
        google_provider = AsyncOpenAI(
            api_key=st.session_state.google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/", # Google's OpenAI-compatible endpoint
        )

        # Choose a Google model compatible with this endpoint (e.g., gemini-1.5-flash-latest, gemini-1.5-pro-latest)
        # Note: gemini-2.0-flash might be an older name, check current valid models
        google_model = OpenAIChatCompletionsModel(
            model="gemini-1.5-flash-latest", # Using a currently valid model name
            openai_client=google_provider, # Pass the google_provider here
        )

        google_run_config = RunConfig(
            model=google_model,
            model_provider=google_provider,
            tracing_disabled=True # Set to False if you want tracing, but often not needed in Streamlit UI
        )
        # st.success("Google Gemini configured successfully!") # Optional success message
    except Exception as e:
         st.error(f"Error configuring Google Gemini: {str(e)}")
         google_run_config = None # Ensure config is None if setup fails

# Keep the original deep_research tool (it uses Firecrawl, not the LLM directly)
@function_tool
async def deep_research(query: str, max_depth: int, time_limit: int, max_urls: int) -> Dict[str, Any]:
    """
    Perform comprehensive web research using Firecrawl's deep research endpoint.
    """
    # This tool doesn't need the google_api_key, only the firecrawl_api_key
    # which is accessed via session_state inside the function.
    
    # Add a check for firecrawl_api_key here as well, just in case the button
    # disabled logic isn't foolproof or the session state changes unexpectedly.
    if not st.session_state.firecrawl_api_key:
         st.error("Firecrawl API Key is required for deep_research.")
         return {"error": "Firecrawl API Key not available", "success": False}

    try:
        # Initialize FirecrawlApp with the API key from session state
        firecrawl_app = FirecrawlApp(api_key=st.session_state.firecrawl_api_key)


        # Set up a callback for real-time updates
        # Note: Streamlit's st.write can sometimes be tricky with real-time callbacks
        # inside threads or async functions. This *might* need adjustment depending
        # on the Firecrawl library's callback execution context.
        def on_activity(activity):
             # Use st.status or st.info for better real-time feedback in Streamlit
             # st.write(f"[{activity['type']}] {activity['message']}") # Original line
             st.info(f"Firecrawl Activity: {activity['message']}", icon="🔍")


        # Run deep research
        with st.spinner("Performing deep research via Firecrawl..."): # Clarified spinner text
            results = firecrawl_app.deep_research(
                query=query,
                max_depth=max_depth,
                time_limit=time_limit,  
                max_urls=max_urls,
                # Use lambda to avoid passing the full function context if needed,
                # or just pass the function directly if it works. Let's try directly.
                on_activity=on_activity
            )

        return {
            "success": True,
            "final_analysis": results.get('data', {}).get('finalAnalysis', 'No analysis found.'),
            "sources_count": len(results.get('data', {}).get('sources', [])),
            "sources": results.get('data', {}).get('sources', [])
        }
    except Exception as e:
        # Ensure specific Firecrawl errors are caught and reported
        st.error(f"Deep research error (Firecrawl): {str(e)}")
        return {"error": str(e), "success": False}

# Keep the original agents (their instructions are general)
research_agent = Agent(
    name="research_agent",
    instructions="""You are a research assistant that can perform deep web research on any topic.

    When given a research topic or question:
    1. Use the deep_research tool to gather comprehensive information
       - Always use these parameters:
         * max_depth: 3 (for moderate depth)
         * time_limit: 180 (3 minutes)
         * max_urls: 10 (sufficient sources)
    2. The tool will search the web, analyze multiple sources, and provide a synthesis
    3. Review the research results and organize them into a well-structured report
    4. Include proper citations for all sources
    5. Highlight key findings and insights
    """,
    tools=[deep_research] # This agent uses the Firecrawl tool
)

elaboration_agent = Agent(
    name="elaboration_agent",
    instructions="""You are an expert content enhancer specializing in research elaboration.

    When given a research report:
    1. Analyze the structure and content of the report
    2. Enhance the report by:
       - Adding more detailed explanations of complex concepts
       - Including relevant examples, case studies, and real-world applications
       - Expanding on key points with additional context and nuance
       - Adding visual elements descriptions (charts, diagrams, infographics)
       - Incorporating latest trends and future predictions
       - Suggesting practical implications for different stakeholders
    3. Maintain academic rigor and factual accuracy
    4. Preserve the original structure while making it more comprehensive
    5. Ensure all additions are relevant and valuable to the topic
    """
    # This agent does NOT need tools in this specific workflow; it just processes text
)

async def run_research_process(topic: str, config: RunConfig):
    """Run the complete research process using the provided RunConfig."""
    # Step 1: Initial Research
    with st.spinner("Conducting initial research using Research Agent..."):
        # Pass the run_config explicitly
        research_result = await Runner.run(research_agent, topic, run_config=config)
        initial_report = research_result.final_output # Assuming final_output contains the generated text

        # Handle potential errors or unexpected output structure
        if not initial_report or not isinstance(initial_report, str):
             st.error("Research Agent failed to produce a valid report.")
             # You might want to inspect research_result object here
             st.write("Research Result object:", research_result)
             return None # Indicate failure

    # Display initial report in an expander
    with st.expander("View Initial Research Report"):
        st.markdown(initial_report)

    # Step 2: Enhance the report
    with st.spinner("Enhancing the report with Elaboration Agent..."):
        elaboration_input = f"""
        RESEARCH TOPIC: {topic}

        INITIAL RESEARCH REPORT:
        {initial_report}

        Please enhance this research report with additional information, examples, case studies,
        and deeper insights while maintaining its academic rigor and factual accuracy.
        """
        # Pass the same run_config explicitly
        elaboration_result = await Runner.run(elaboration_agent, elaboration_input, run_config=config)
        enhanced_report = elaboration_result.final_output # Assuming final_output contains the enhanced text

        # Handle potential errors or unexpected output structure for elaboration
        if not enhanced_report or not isinstance(enhanced_report, str):
            st.error("Elaboration Agent failed to produce a valid enhanced report.")
            st.write("Elaboration Result object:", elaboration_result)
            # Decide if you want to return the initial report or nothing
            return initial_report if initial_report else "Error: Could not enhance report."


    return enhanced_report

# Main research process execution block
# Check if both keys are provided AND the Google config was successfully created
start_button_disabled = not (st.session_state.google_api_key and st.session_state.firecrawl_api_key and research_topic and google_run_config is not None)

if st.button("Start Research", disabled=start_button_disabled):
    if not st.session_state.google_api_key or not st.session_state.firecrawl_api_key:
        st.warning("Please enter both API keys in the sidebar.")
    elif not research_topic:
        st.warning("Please enter a research topic.")
    elif google_run_config is None:
        st.error("Google Gemini configuration failed. Please check your API key and connection.")
    else:
        try:
            # Create placeholder for the final report
            report_placeholder = st.empty()

            # Run the research process, passing the google_run_config
            enhanced_report = asyncio.run(run_research_process(research_topic, google_run_config))

            if enhanced_report: # Only display and offer download if report was generated
                # Display the enhanced report
                report_placeholder.markdown("## Enhanced Research Report")
                report_placeholder.markdown(enhanced_report)

                # Add download button
                st.download_button(
                    "Download Report",
                    enhanced_report,
                    file_name=f"{research_topic.replace(' ', '_').lower().replace('.', '')}_gemini_report.md", # Clean filename
                    mime="text/markdown"
                )

        except Exception as e:
            # Catch any other unexpected errors during the run process
            st.exception(e) # Use st.exception to show traceback

# Footer
st.markdown("---")
st.markdown("Powered by `agents` library, Google Gemini, and Firecrawl") # Updated footer
