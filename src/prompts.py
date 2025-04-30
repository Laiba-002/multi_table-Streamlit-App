import os
from openai import OpenAI
import streamlit as st
# Initialize OpenAI client
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
GPT_MODEL = "gpt-4o-mini"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.

def call_openai(prompt, system_prompt="You are a helpful assistant.", conversation_history=None):
    """
    Call the OpenAI API with the specified prompt and optional conversation history.

    Args:
        prompt: The current user query
        system_prompt: The system prompt to set the assistant's behavior
        conversation_history: A list of previous messages in the format [{"role": "user|assistant", "content": "message"}]
    """
    # Prepare messages with system prompt first
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history if provided
    if conversation_history and len(conversation_history) > 0:
        messages.extend(conversation_history)

    # Add current prompt
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}"

def generate_introduction(schema_info, table_name="OEESHIFTWISE_AI"):
    """Generate an introduction about the OEE data based on the schema."""

    prompt = f"""
    You are an assistant that specializes in Overall Equipment Effectiveness (OEE) analysis.
    You have access to a database table called {table_name} with the following columns and data types:

    {schema_info}

    Please provide a brief introduction explaining what OEE is and how this data can help users
    analyze manufacturing equipment performance. Also suggest a few types of questions the user
    could ask about this data.
    """

    system_prompt = "You are a helpful assistant that specializes in manufacturing analytics."

    try:
        response = call_openai(prompt, system_prompt)
        return response
    except Exception as e:
        return f"I encountered an error generating an introduction: {str(e)}. You can ask me questions about OEE data, and I'll do my best to answer them."

def get_llm_response(user_query, table_name, schema_name, database_name, column_info, conversation_history=None):
    """
    Generate a response to a user query about OEE data.

    Args:
        user_query: The user's question
        table_name: The name of the table in Snowflake
        schema_name: The name of the schema in Snowflake
        database_name: The name of the database in Snowflake
        column_info: Information about the columns in the table
        conversation_history: List of previous messages in the format [{"role": "user|assistant", "content": "message"}]
    """

    prompt = f"""
    You are an assistant that helps users query and analyze OEE (Overall Equipment Effectiveness) data.

    The user is querying a Snowflake database with the following details:
    - Database: {database_name}
    - Schema: {schema_name}
    - Table: {table_name}

    The table has these columns with their data types:
    {column_info}

    The user's current query is: "{user_query}"

    When responding, consider the conversation context and previous questions if they are provided.
    


    If the query is asking for data that can be retrieved with SQL, please:
    1. Write a SQL query to get the requested information from the {table_name} table
    2. Format your SQL code block with ```sql at the beginning and ``` at the end
    3. Make sure to write standard SQL compatible with Snowflake
    4. Use proper column names as shown above
    5. Keep your SQL query efficient and focused on answering the specific question
    6. In case of sql query did not generate it don't explixitly show error message,instead it show user friendly response for example;"My bad,please rephrase your question or try again later."

    If the query is not asking for data or cannot be answered with SQL:
    1. Provide a helpful explanation about OEE concepts
    2. Suggest a reformulation of their question that could be answered with the available data

    Remember, OEE (Overall Equipment Effectiveness) is a standard metric in manufacturing that measures 
    productivity by combining availability, performance, and quality metrics.
    """

    system_prompt = "You are a helpful assistant that specializes in SQL queries and OEE analytics."

    try:
        response = call_openai(prompt, system_prompt, conversation_history)
        return response
    except Exception as e:
        return f"I encountered an error: {str(e)}. Please try rephrasing your question or try again later."