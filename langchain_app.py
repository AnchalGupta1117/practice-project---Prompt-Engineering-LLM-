
import streamlit as st
import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_utils import (
    run_prompt_template,
    run_chat_prompt_template,
    run_summarization_chain
)


# ------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)


# ------------------------------------------------
# STREAMLIT CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="LangChain + Gemini Application",
    page_icon="✦",
    layout="centered"
)


# ------------------------------------------------
# INITIALIZE GEMINI THROUGH LANGCHAIN
# ------------------------------------------------

@st.cache_resource
def get_langchain_model(api_key, model_name, temperature):

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature
    )


# ------------------------------------------------
# APPLICATION UI
# ------------------------------------------------

st.title("✦ LangChain + Gemini Application")

st.write(
    "Explore LangChain prompt templates, "
    "chat prompts and reusable LLM chains."
)

st.divider()


# Select LangChain technique

st.subheader("1. Select LangChain Technique")

task = st.selectbox(
    "Choose a technique",
    [
        "PromptTemplate - Technical Explanation",
        "ChatPromptTemplate - Email Generation",
        "LCEL Chain - Text Summarization"
    ]
)


# Model configuration

st.subheader("2. Model Configuration")

temperature = st.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1
)


# ------------------------------------------------
# DYNAMIC INPUT FIELDS
# ------------------------------------------------

variables = {}


if task == "PromptTemplate - Technical Explanation":

    st.subheader("Technical Explanation Details")

    variables["topic"] = st.text_input(
        "Enter Topic",
        placeholder="e.g. Retrieval Augmented Generation"
    )

    variables["audience"] = st.selectbox(
        "Target Audience",
        ["Beginner", "Intermediate", "Advanced"]
    )

    variables["detail_level"] = st.selectbox(
        "Explanation Level",
        ["Simple", "Moderate", "Detailed"]
    )


elif task == "ChatPromptTemplate - Email Generation":

    st.subheader("Email Details")

    variables["recipient"] = st.text_input(
        "Recipient",
        placeholder="e.g. Project Manager"
    )

    variables["purpose"] = st.text_input(
        "Email Purpose",
        placeholder="e.g. Request project status"
    )

    variables["tone"] = st.selectbox(
        "Email Tone",
        ["Professional", "Formal", "Friendly", "Polite"]
    )

    variables["key_points"] = st.text_area(
        "Important Points",
        placeholder="Enter important points..."
    )


elif task == "LCEL Chain - Text Summarization":

    st.subheader("Summarization Details")

    variables["input_text"] = st.text_area(
        "Enter Text",
        height=220,
        placeholder="Paste text to summarize..."
    )

    variables["summary_length"] = st.selectbox(
        "Summary Length",
        ["Short", "Medium", "Detailed"]
    )


# ------------------------------------------------
# GENERATE RESPONSE
# ------------------------------------------------

if st.button("Generate Response", type="primary"):

    if not API_KEY:

        st.error(
            "Gemini API key not found. "
            "Please check your .env file."
        )

    elif not all(
        str(value).strip()
        for value in variables.values()
    ):

        st.warning("Please complete all required fields.")

    else:

        try:

            # Initialize LangChain Gemini model

            model = get_langchain_model(
                API_KEY,
                MODEL_NAME,
                temperature
            )

            with st.spinner("Running LangChain chain..."):

                # TECHNICAL EXPLANATION

                if task == "PromptTemplate - Technical Explanation":

                    result, prompt_preview = run_prompt_template(
                        model,
                        variables["topic"],
                        variables["audience"],
                        variables["detail_level"]
                    )


                # EMAIL GENERATION

                elif task == "ChatPromptTemplate - Email Generation":

                    result, prompt_preview = run_chat_prompt_template(
                        model,
                        variables["recipient"],
                        variables["purpose"],
                        variables["tone"],
                        variables["key_points"]
                    )


                # TEXT SUMMARIZATION

                elif task == "LCEL Chain - Text Summarization":

                    result, prompt_preview = run_summarization_chain(
                        model,
                        variables["input_text"],
                        variables["summary_length"]
                    )


            # DISPLAY RESULT

            st.success("Response generated successfully!")

            st.subheader("Generated Response")

            st.write(result)

            with st.expander("View Formatted Prompt"):

                st.code(
                    prompt_preview,
                    language="text"
                )

            with st.expander("View Model Configuration"):

                st.write("Provider: Google Gemini")
                st.write("Integration: LangChain")
                st.write("Model:", MODEL_NAME)
                st.write("Temperature:", temperature)

        except Exception as e:

            st.error(f"Something went wrong: {e}")
