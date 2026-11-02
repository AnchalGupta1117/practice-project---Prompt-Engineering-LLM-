
import streamlit as st
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompt_templates import (
    create_prompt,
    get_system_prompt,
    add_few_shot_examples,
    create_extraction_prompt,
    create_analysis_prompt,
    PROMPT_VARIANTS,
    EVALUATION_CASES,
    create_evaluation_prompt
)


# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


# Streamlit configuration
st.set_page_config(
    page_title="Prompt Engineering Application",
    page_icon="✦",
    layout="centered"
)


# Gemini client
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=API_KEY)


# Generate response using Gemini
def generate_response(user_prompt, system_prompt, temperature=0.7):

    client = get_gemini_client()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature
        )
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text.strip()


# PROMPT CHAINING
def run_invoice_chain(document_text):

    business_system_prompt = get_system_prompt("Business Analyst")

    extraction_prompt = create_extraction_prompt(document_text)

    extracted_information = generate_response(
        extraction_prompt,
        business_system_prompt
    )

    analysis_prompt = create_analysis_prompt(
        extracted_information
    )

    final_analysis = generate_response(
        analysis_prompt,
        business_system_prompt
    )

    return extracted_information, final_analysis


# Extract sentiment label from model response
def extract_label(response):

    normalized = response.strip().lower()

    for label in ["positive", "negative", "neutral"]:

        if normalized == label:
            return label.capitalize()

    # Handle responses that contain a label with extra text
    first_line = normalized.splitlines()[0].strip(" .,:*-")

    for label in ["positive", "negative", "neutral"]:

        if first_line.startswith(label):
            return label.capitalize()

    return "Invalid Output"


# Evaluate one prompt version
def evaluate_prompt_version(version):

    system_prompt = get_system_prompt("Classification Assistant")

    results = []

    correct = 0
    total = len(EVALUATION_CASES)

    for case in EVALUATION_CASES:

        prompt = create_evaluation_prompt(
            version,
            case["text"]
        )

        start_time = time.perf_counter()

        response = generate_response(
            prompt,
            system_prompt,
            temperature=0
        )

        latency = time.perf_counter() - start_time

        predicted = extract_label(response)

        is_correct = predicted == case["expected"]

        if is_correct:
            correct += 1

        results.append({
            "Review": case["text"],
            "Expected": case["expected"],
            "Predicted": predicted,
            "Result": "Correct" if is_correct else "Incorrect",
            "Latency (seconds)": round(latency, 2)
        })

    accuracy = (correct / total) * 100

    return results, accuracy, correct, total


# Application UI
st.title("✦ Prompt Engineering Application")

st.write(
    "Explore dynamic templates, system prompts, few-shot prompting, "
    "prompt chaining and prompt evaluation."
)

st.divider()


# Select system role
st.subheader("1. Select AI Behaviour")

role = st.selectbox(
    "Choose System Prompt",
    [
        "Professional Assistant",
        "Technical Instructor",
        "Business Analyst",
        "Classification Assistant"
    ]
)


# Select task
st.subheader("2. Select Your Task")

task = st.selectbox(
    "Choose Task",
    [
        "Professional Email",
        "Text Summarization",
        "Technical Explanation",
        "Sentiment Classification",
        "Invoice Processing Pipeline (Prompt Chaining)",
        "Prompt Evaluation Dashboard"
    ]
)


# Dynamic input fields
variables = {}
document_text = ""
use_few_shot = False


if task == "Professional Email":

    st.subheader("Email Details")

    variables["recipient"] = st.text_input(
        "Recipient",
        placeholder="e.g. Project Manager"
    )

    variables["purpose"] = st.text_input(
        "Purpose",
        placeholder="e.g. Request a project update"
    )

    variables["tone"] = st.selectbox(
        "Email Tone",
        ["Professional", "Friendly", "Formal", "Polite"]
    )

    variables["key_points"] = st.text_area(
        "Important Points",
        placeholder="Enter points to include"
    )


elif task == "Text Summarization":

    st.subheader("Summarization Details")

    variables["input_text"] = st.text_area(
        "Enter Text",
        placeholder="Paste text to summarize..."
    )

    variables["summary_length"] = st.selectbox(
        "Summary Length",
        ["Short", "Medium", "Detailed"]
    )

    variables["audience"] = st.selectbox(
        "Target Audience",
        ["General Audience", "Students", "Professionals", "Technical Experts"]
    )


elif task == "Technical Explanation":

    st.subheader("Explanation Details")

    variables["concept"] = st.text_input(
        "Enter Concept",
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


elif task == "Sentiment Classification":

    st.subheader("Classification Details")

    variables["input_text"] = st.text_area(
        "Enter Text to Classify",
        placeholder="e.g. The product quality is excellent!"
    )

    use_few_shot = st.checkbox(
        "Enable Few-Shot Examples",
        value=True
    )


elif task == "Invoice Processing Pipeline (Prompt Chaining)":

    st.subheader("Invoice Processing")

    document_text = st.text_area(
        "Enter Invoice Text",
        height=220,
        placeholder="""
Invoice Number: INV-2026-101
Supplier Name: ABC Technologies Pvt Ltd
Invoice Date: 01 November 2026
Due Date: 15 November 2026
Total Amount: INR 45,000
Description: Software development services
Payment Terms: Net 14 days
"""
    )


elif task == "Prompt Evaluation Dashboard":

    st.subheader("Prompt Evaluation")

    st.write(
        "Compare two prompt versions using the same sentiment "
        "classification test dataset."
    )

    st.write("Number of test cases:", len(EVALUATION_CASES))

    st.write("Prompt versions:", len(PROMPT_VARIANTS))

    with st.expander("View Evaluation Dataset"):

        for index, case in enumerate(EVALUATION_CASES, start=1):

            st.write(f"Test {index}: {case['text']}")
            st.write(f"Expected Label: {case['expected']}")
            st.divider()


# Generate button
if st.button("Generate Response", type="primary"):

    if not API_KEY:

        st.error("Gemini API key not found. Check your .env file.")

    elif task == "Invoice Processing Pipeline (Prompt Chaining)" and not document_text.strip():

        st.warning("Please enter invoice text.")

    elif task != "Invoice Processing Pipeline (Prompt Chaining)" and task != "Prompt Evaluation Dashboard" and not all(
        str(value).strip() for value in variables.values()
    ):

        st.warning("Please complete all required input fields.")

    else:

        try:

            # PROMPT EVALUATION
            if task == "Prompt Evaluation Dashboard":

                all_results = {}

                progress = st.progress(0)

                total_runs = len(PROMPT_VARIANTS) * len(EVALUATION_CASES)
                completed_runs = 0

                for version in PROMPT_VARIANTS:

                    results = evaluate_prompt_version(version)

                    all_results[version] = results

                    completed_runs += len(EVALUATION_CASES)

                    progress.progress(
                        completed_runs / total_runs
                    )

                st.success("Prompt evaluation completed!")

                st.subheader("Evaluation Comparison")

                for version, evaluation in all_results.items():

                    results, accuracy, correct, total = evaluation

                    st.markdown(f"### {version}")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Accuracy",
                        f"{accuracy:.1f}%"
                    )

                    col2.metric(
                        "Correct",
                        f"{correct}/{total}"
                    )

                    col3.metric(
                        "Incorrect",
                        f"{total - correct}/{total}"
                    )

                    st.dataframe(
                        results,
                        use_container_width=True,
                        hide_index=True
                    )

                st.subheader("Final Comparison")

                v1_accuracy = all_results[
                    "Version 1 - Basic Prompt"
                ][1]

                v2_accuracy = all_results[
                    "Version 2 - Optimized Prompt"
                ][1]

                st.write(
                    f"Basic Prompt Accuracy: {v1_accuracy:.1f}%"
                )

                st.write(
                    f"Optimized Prompt Accuracy: {v2_accuracy:.1f}%"
                )

                if v2_accuracy > v1_accuracy:

                    st.success(
                        "Version 2 performed better on this test dataset."
                    )

                elif v1_accuracy > v2_accuracy:

                    st.info(
                        "Version 1 performed better on this test dataset."
                    )

                else:

                    st.info(
                        "Both prompts achieved the same accuracy."
                    )

                st.caption(
                    "Results are specific to this small test dataset. "
                    "Run more diverse tests before choosing a production prompt."
                )


            # INVOICE PROMPT CHAINING
            elif task == "Invoice Processing Pipeline (Prompt Chaining)":

                with st.spinner("Running Step 1: Extracting invoice information..."):

                    extracted_information, final_analysis = run_invoice_chain(
                        document_text
                    )

                st.subheader("Step 1 — Extracted Information")
                st.write(extracted_information)

                st.divider()

                st.subheader("Step 2 — Business Analysis")
                st.write(final_analysis)


            # OTHER TASKS
            else:

                final_prompt = create_prompt(task, **variables)

                if task == "Sentiment Classification" and use_few_shot:

                    final_prompt = add_few_shot_examples(
                        task,
                        final_prompt
                    )

                system_prompt = get_system_prompt(role)

                with st.spinner("Generating response..."):

                    result = generate_response(
                        final_prompt,
                        system_prompt
                    )

                st.subheader("Generated Response")
                st.write(result)

                with st.expander("View System Prompt"):
                    st.code(system_prompt, language="text")

                with st.expander("View Final User Prompt"):
                    st.code(final_prompt, language="text")


        except Exception as e:

            st.error(f"Something went wrong: {e}")
