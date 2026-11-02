# Prompt Engineering Playground

A Python-based LLM application built using **Streamlit, Google Gemini API, and LangChain** to explore practical prompt engineering techniques and develop reusable LLM workflows.

This project demonstrates how different prompting strategies can be implemented in real-world LLM applications, from dynamic prompt generation to multi-step workflows and prompt evaluation.

## Key Features

### 1. Prompt Engineering Application

- **Dynamic Prompt Templates:** Generate prompts using reusable templates and user-provided variables.
- **System Prompts:** Control the behaviour and role of the LLM through system instructions.
- **Few-Shot Prompting:** Guide model responses using example inputs and outputs.
- **Prompt Chaining:** Connect multiple LLM calls to perform sequential tasks.
- **Invoice Processing Workflow:** Demonstrate invoice information extraction followed by analysis using synthetic invoice data.
- **Prompt Optimization:** Compare different prompt versions.
- **Prompt Evaluation:** Evaluate model responses against a test dataset and compare accuracy and response latency.

### 2. LangChain Integration

A separate application demonstrating LangChain-based LLM development.

- **PromptTemplate:** Create reusable prompts with dynamic variables.
- **ChatPromptTemplate:** Structure conversations using system and human messages.
- **LCEL (LangChain Expression Language):** Connect prompts, language models, and output parsers into reusable chains.
- **StrOutputParser:** Convert model responses into plain text.
- **Gemini Integration:** Connect Google Gemini models with LangChain.

## Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM Provider | Google Gemini API |
| LLM Framework | LangChain |
| Environment Management | Python Virtual Environment |
| Configuration Management | python-dotenv |

## Project Structure

```text
practice-project---Prompt-Engineering-LLM-/
│
├── app.py
├── prompt_templates.py
├── langchain_app.py
├── langchain_utils.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Application Architecture

The project contains two separate applications.

### Application 1: Prompt Engineering Application

```text
          User Input
               |
               v
        Prompt Templates
               |
               v
     Prompt Engineering Techniques
               |
               v
        Google Gemini API
               |
               v
       Generated Response
               |
               v
        Streamlit Interface
```

### Application 2: LangChain Application

```text
          User Input
               |
               v
   PromptTemplate / ChatPromptTemplate
               |
               v
           LCEL Chain
               |
               v
       Google Gemini Model
               |
               v
       StrOutputParser
               |
               v
        Final Response
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AnchalGupta1117/practice-project---Prompt-Engineering-LLM-.git
```

Navigate to the project directory:

```bash
cd practice-project---Prompt-Engineering-LLM-
```

### 2. Create a Virtual Environment

```bash
python -m venv llmvenv
```

### 3. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\llmvenv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the root directory.

Add the following:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Replace the placeholder with your own Google Gemini API key.

**Important:** Never upload your `.env` file or expose your API key on GitHub.

### 6. Run the Applications

**Prompt Engineering Application:**

```bash
streamlit run app.py
```

**LangChain Application:**

```bash
streamlit run langchain_app.py
```

## Learning Outcomes

Through this project, I explored and implemented:

- Designing reusable prompt templates.
- Working with dynamic prompt variables.
- Controlling LLM behaviour using system instructions.
- Implementing few-shot prompting.
- Building sequential LLM workflows using prompt chaining.
- Comparing prompt versions through evaluation.
- Integrating Google Gemini with LangChain.
- Creating reusable LCEL chains.
- Developing interactive LLM applications using Streamlit.

## Limitations

- A valid Google Gemini API key is required to run the applications.
- The invoice processing workflow is a demonstration using synthetic data.
- Prompt evaluation uses a small test dataset and is not a production-grade benchmark.
- The project is an educational implementation and is not intended for production financial-document processing.

## Future Improvements

- Expand the evaluation dataset and introduce more robust evaluation metrics.
- Add automated unit and integration tests.
- Implement structured JSON output validation.
- Integrate document-based Retrieval-Augmented Generation (RAG).
- Explore advanced LLM orchestration and agentic workflows.