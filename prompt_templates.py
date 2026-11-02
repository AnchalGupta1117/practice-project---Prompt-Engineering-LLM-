
PROMPT_TEMPLATES = {
    "Professional Email": """
Write an email to {recipient}.

Purpose:
{purpose}

Tone:
{tone}

Important points:
{key_points}

Instructions:
- Include a suitable subject line.
- Keep the email clear and professional.
- Do not invent information.
""",

    "Text Summarization": """
Summarize the following text:

{input_text}

Summary length:
{summary_length}

Target audience:
{audience}

Instructions:
- Preserve important information.
- Remove unnecessary repetition.
- Do not add information that is not in the original text.
""",

    "Technical Explanation": """
Explain the following concept:

Concept:
{concept}

Target audience:
{audience}

Explanation level:
{detail_level}

Instructions:
- Explain clearly.
- Include a practical example.
- Explain real-world applications.
""",

    "Sentiment Classification": """
Classify the sentiment of the given text.

Text:
{input_text}

Return only one label:
Positive, Negative, or Neutral.
"""
}


SYSTEM_PROMPTS = {
    "Professional Assistant": """
You are a professional AI assistant.
Use clear, polite and professional language.
Keep responses structured and useful.
""",

    "Technical Instructor": """
You are an experienced technical instructor.
Explain technical concepts in simple English.
Use practical examples and real-world applications.
Avoid unnecessary complexity.
""",

    "Business Analyst": """
You are an experienced business analyst.
Analyse information logically.
Present insights in a structured format.
Highlight important findings and practical recommendations.
Do not invent facts or data.
""",

    "Classification Assistant": """
You are an accurate text classification assistant.
Follow the given instructions carefully.
Return the requested classification label only.
"""
}


FEW_SHOT_EXAMPLES = {
    "Sentiment Classification": """
Example 1:
Input: I absolutely love this product.
Output: Positive

Example 2:
Input: The customer service was very disappointing.
Output: Negative

Example 3:
Input: The product is average. Nothing special.
Output: Neutral

Example 4:
Input: The delivery was excellent and very fast.
Output: Positive

Example 5:
Input: I am unhappy with the quality.
Output: Negative
"""
}


def create_prompt(task, **variables):
    template = PROMPT_TEMPLATES[task]
    return template.format(**variables)


def get_system_prompt(role):
    return SYSTEM_PROMPTS[role]


def add_few_shot_examples(task, prompt):
    examples = FEW_SHOT_EXAMPLES.get(task)

    if examples:
        return f"""
Learn the following examples:

{examples}

Now perform the task below:

{prompt}
"""

    return prompt


# PROMPT CHAINING

def create_extraction_prompt(document_text):
    return f"""
You are an invoice information extraction system.

Extract the following information from the invoice text:

1. Invoice Number
2. Supplier Name
3. Invoice Date
4. Due Date
5. Total Amount
6. Currency
7. Description of Goods or Services
8. Payment Terms

Invoice Text:
{document_text}

Important rules:
- Extract only information explicitly present in the document.
- If a field is missing, write "Not Provided".
- Do not guess or invent values.
- Present the result in a clear structured format.
"""


def create_analysis_prompt(extracted_information):
    return f"""
You are a business analyst working in an invoice processing system.

Analyze the following extracted invoice information:

{extracted_information}

Perform these tasks:

1. Summarize the invoice.
2. Identify missing information.
3. Identify unusual or potentially concerning details.
4. Mention which fields require human verification.
5. Give a final processing recommendation.

Use these recommendation categories:
- Ready for Review
- Requires Human Verification
- Incomplete Information

Important rules:
- Do not invent missing information.
- Do not assume that an invoice is genuine or approved.
- Clearly distinguish actual missing fields from potential concerns.
- Explain your reasoning briefly.
"""


# PROMPT OPTIMIZATION

PROMPT_VARIANTS = {
    "Version 1 - Basic Prompt": """
Classify the sentiment of this customer review.

Review:
{input_text}

Return Positive, Negative, or Neutral.
""",

    "Version 2 - Optimized Prompt": """
You are a professional sentiment classification system.

Analyze the customer's review carefully.

Review:
{input_text}

Classification rules:

1. Positive: The customer expresses satisfaction,
   happiness, appreciation or a favourable opinion.

2. Negative: The customer expresses dissatisfaction,
   frustration, disappointment or an unfavourable opinion.

3. Neutral: The customer expresses neither a clearly
   positive nor a clearly negative opinion.

Instructions:
- Consider the overall meaning of the review.
- Do not classify based on a single word alone.
- Return exactly one label.
- Do not provide explanations.

Allowed output:
Positive
Negative
Neutral
"""
}


# EVALUATION DATASET

EVALUATION_CASES = [
    {
        "text": "I absolutely love this product. It works perfectly!",
        "expected": "Positive"
    },
    {
        "text": "The delivery was late and customer support was terrible.",
        "expected": "Negative"
    },
    {
        "text": "The product arrived yesterday. I have not used it yet.",
        "expected": "Neutral"
    },
    {
        "text": "Excellent quality and wonderful customer service!",
        "expected": "Positive"
    }
]


def create_evaluation_prompt(version, input_text):
    template = PROMPT_VARIANTS[version]

    return template.format(input_text=input_text)
