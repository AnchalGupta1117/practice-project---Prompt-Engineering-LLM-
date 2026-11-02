
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ------------------------------------------------
# 1. PROMPT TEMPLATE
# ------------------------------------------------

def run_prompt_template(model, topic, audience, detail_level):

    prompt = PromptTemplate.from_template(
        """
You are an experienced technical instructor.

Explain the following topic:

Topic: {topic}

Target Audience: {audience}

Explanation Level: {detail_level}

Instructions:
- Explain the concept clearly.
- Include a practical example.
- Explain its real-world applications.
- Use simple and understandable language.
"""
    )

    # Create LCEL chain
    chain = prompt | model | StrOutputParser()

    # Prepare prompt preview
    formatted_prompt = prompt.format(
        topic=topic,
        audience=audience,
        detail_level=detail_level
    )

    # Execute chain
    result = chain.invoke({
        "topic": topic,
        "audience": audience,
        "detail_level": detail_level
    })

    return result, formatted_prompt


# ------------------------------------------------
# 2. CHAT PROMPT TEMPLATE
# ------------------------------------------------

def run_chat_prompt_template(
    model,
    recipient,
    purpose,
    tone,
    key_points
):

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a professional email-writing assistant.

Write clear, polite and professional emails.
Do not invent information.
"""
            ),

            (
                "human",
                """
Write an email to: {recipient}

Purpose: {purpose}

Tone: {tone}

Important Points: {key_points}

Include a suitable subject line.
"""
            )
        ]
    )

    # Create LCEL chain
    chain = prompt | model | StrOutputParser()

    # Prepare formatted messages for preview
    messages = prompt.format_messages(
        recipient=recipient,
        purpose=purpose,
        tone=tone,
        key_points=key_points
    )

    preview = "\n\n".join(
        f"{message.type.upper()}:\n{message.content}"
        for message in messages
    )

    # Execute chain
    result = chain.invoke({
        "recipient": recipient,
        "purpose": purpose,
        "tone": tone,
        "key_points": key_points
    })

    return result, preview


# ------------------------------------------------
# 3. REUSABLE SUMMARIZATION CHAIN
# ------------------------------------------------

def run_summarization_chain(model, input_text, summary_length):

    prompt = PromptTemplate.from_template(
        """
You are an expert text summarization assistant.

Summarize the following text:

{input_text}

Summary Length: {summary_length}

Instructions:
- Preserve all important information.
- Remove unnecessary repetition.
- Do not introduce information that is not in the original text.
- Use clear and simple language.
"""
    )

    # Build reusable chain
    chain = prompt | model | StrOutputParser()

    # Prepare preview
    formatted_prompt = prompt.format(
        input_text=input_text,
        summary_length=summary_length
    )

    # Execute chain
    result = chain.invoke({
        "input_text": input_text,
        "summary_length": summary_length
    })

    return result, formatted_prompt
