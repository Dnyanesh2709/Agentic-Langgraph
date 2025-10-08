import streamlit as st
import json
import traceback
from src.langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.ui.streamlitui.display_result import DisplayResultStreamlit

# MAIN Function START
def load_langgraph_agenticai_app():
    """
    Loads and runs the LangGraph AgenticAI application with Streamlit UI.
    This function initializes the UI, handles user input, configures the LLM model,
    sets up the graph based on the selected use case, and displays the output while 
    implementing exception handling for robustness.
    """

    try:
        # Load UI
        ui = LoadStreamlitUI()
        user_input = ui.load_streamlit_ui()

        if not user_input:
            st.error("Error: Failed to load user input from the UI.")
            st.stop()

        # Text input for user message
        if st.session_state.get("IsFetchButtonClicked"):
            user_message = st.session_state.get("timeframe")
        else:
            user_message = st.chat_input("Enter your message:")

        # Main logic
        if user_message:
            try:
                # Configure LLM
                obj_llm_config = GroqLLM(user_controls_input=user_input)
                model = obj_llm_config.get_llm_model()

                if not model:
                    st.error("Error: LLM model could not be initialized.")
                    st.stop()

                # Initialize and set up the graph based on use case
                usecase = user_input.get("selected_usecase")
                if not usecase:
                    st.error("Error: No use case selected.")
                    st.stop()

                ### Graph Builder
                graph_builder = GraphBuilder(model)

                try:
                    graph = graph_builder.setup_graph(usecase)
                    DisplayResultStreamlit(usecase, graph, user_message).display_result_on_ui()
                except Exception as e:
                    st.error(f"Graph setup failed: {e}")
                    st.code(traceback.format_exc())
                    st.stop()

            except Exception as e:
                st.error(f"An unexpected error occurred while configuring the app: {e}")
                st.code(traceback.format_exc())
                st.stop()

    except Exception as e:
        st.error("Critical error during app initialization.")
        st.code(traceback.format_exc())
        st.stop()
