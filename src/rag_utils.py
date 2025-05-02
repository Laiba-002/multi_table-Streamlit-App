import jwt
import pandas as pd
import numpy as np
import json
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import snowflake.connector
import logging
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
import time
import requests
from datetime import datetime

        
# Configure logging
logging.basicConfig(
    filename="rag_utils.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# Initialize OpenAI client
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
GPT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

@st.cache_data(show_spinner=False, ttl=3600)
def get_openai_embedding(texts):
    try:
        # Batch embedding request
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts if isinstance(texts, list) else [texts]
        )
        embeddings = [data.embedding for data in response.data]
        logging.info(f"Generated {len(embeddings)} embeddings")
        return embeddings if isinstance(texts, list) else embeddings[0]
    except Exception as e:
        st.error(f"Error getting embeddings: {str(e)}")
        logging.error(f"Embedding error: {str(e)}")
        return [] if isinstance(texts, list) else []

def call_openai(prompt, system_prompt="You are a helpful assistant.", conversation_history=None):
    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": prompt})
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI API error: {str(e)}")
        return f"Error connecting to OpenAI: {str(e)}"

@st.cache_data(show_spinner=False, ttl=3600)
def infer_user_intent(user_query, table_names_input):
    # Handle both list and string inputs for safety
    if isinstance(table_names_input, list):
        table_names = table_names_input
    else:
        table_names = table_names_input.split(", ")
    # table_names = table_names_str.split(", ")
    try:
        intent_prompt= f"""
        Analyze the following user query to determine which table(s) it pertains to:
        Query: "{user_query}"
        Available tables: {', '.join(table_names)}
        Table descriptions:
        - OEESHIFTWISE_AI: Shift-level OEE performance data (e.g., OEE, availability, production metrics).
        - MACHINE_ACCESS_INFO_AI: Machine or plantcode  access information (e.g., machine ID,plantcode).
        Respond with a JSON object:
        {{
            "tables": ["table_name1", "table_name2"] or ["table_name1"] or ["table_name2"] 
            "reason": "Explanation of table selection"
        }}
        If the query involves all tables (e.g., combining shift performance  with specific plant access ), include all.
        """
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are an intent detection expert."},
                {"role": "user", "content": intent_prompt}
            ],
            response_format={"type": "json_object"}
        )
        intent = json.loads(response.choices[0].message.content)
        logging.info(f"Inferred intent: {intent}")
        return intent
    except Exception as e:
        logging.error(f"Intent detection error: {str(e)}")
        return {"tables": table_names, "reason": "Defaulting to all tables due to error"}

def create_document_chunks(df, chunk_size=5):
    chunks = []
    total_rows = len(df)
    for i in range(0, total_rows, chunk_size):
        end_idx = min(i + chunk_size, total_rows)
        chunk_df = df.iloc[i:end_idx]
        chunk_text = "Data chunk:\n"
        for _, row in chunk_df.iterrows():
            row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
            chunk_text += row_text + "\n"
        chunks.append(chunk_text)
    schema_chunk = "Table schema:\n"
    for col in df.columns:
        schema_chunk += f"Column: {col}, Type: {df[col].dtype}\n"
    chunks.append(schema_chunk)
    for col in df.select_dtypes(include=[np.number]).columns:
        stats_chunk = f"Stats for {col}:\n"
        stats_chunk += f"Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean()}, Median: {df[col].median()}\n"
        chunks.append(stats_chunk)
    return chunks

# @st.cache_resource
def init_snowflake_connection():
    try:
        conn = snowflake.connector.connect(
            user=st.secrets["snowflake"]["user"],
            password=st.secrets["snowflake"]["password"],
            account=st.secrets["snowflake"]["account"],
            warehouse=st.secrets["snowflake"]["warehouse"],
            database="O3_AI_DB",
            schema="O3_AI_DB_SCHEMA"
        )
        logging.info("Snowflake connection initialized")
        return conn
    except Exception as e:
        st.error(f"Error connecting to Snowflake: {str(e)}")
        logging.error(f"Snowflake connection error: {str(e)}")
        return None

@st.cache_data(show_spinner=False, ttl=86400)
def initialize_vector_store(df_str, table_name):
    df = pd.read_json(StringIO(df_str))
    conn = init_snowflake_connection()
    if not conn:
        return None
    cursor = conn.cursor()
    try:
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM O3_SNOWFLAKE_EMBEDDINGS 
            WHERE TABLE_NAME = '{table_name}'
        """)
        count = cursor.fetchone()[0]
        if count > 0:
            cursor.execute(f"""
                SELECT RECORD_ID, CHUNK_TEXT, EMBEDDING 
                FROM O3_SNOWFLAKE_EMBEDDINGS 
                WHERE TABLE_NAME = '{table_name}'
            """)
            results = cursor.fetchall()
            chunks = [json.loads(row[1]) if row[1] else "" for row in results]
            embeddings = [json.loads(row[2]) if row[2] else [] for row in results]
            # st.success(f"Loaded embeddings for {table_name}")
            logging.info(f"Loaded {len(chunks)} embeddings from Snowflake for {table_name}")
            return {"embeddings": embeddings, "chunks": chunks}
        chunks = create_document_chunks(df)
        # Batch embedding generation
        batch_size = 50
        embeddings = []
        progress_placeholder = st.empty()
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            progress_placeholder.text(f"Creating embeddings for {table_name}... ({i+1}/{len(chunks)})")
            batch_embeddings = get_openai_embedding(batch_chunks)
            if not batch_embeddings:
                st.warning(f"Failed to generate embeddings for batch {i+1} in {table_name}.")
                continue
            embeddings.extend(batch_embeddings)
            # Store in Snowflake
            for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                cursor.execute("""
                    INSERT INTO O3_SNOWFLAKE_EMBEDDINGS (RECORD_ID, TABLE_NAME, CHUNK_TEXT, EMBEDDING)
                    VALUES (%s, %s, %s, %s)
                """, (f"{table_name}_{i+j}", table_name, json.dumps(chunk), json.dumps(embedding)))
        conn.commit()
        # st.success(f"Embeddings stored for {table_name}")
        logging.info(f"Stored {len(embeddings)} embeddings in Snowflake for {table_name}")
        return {"embeddings": embeddings, "chunks": chunks}
    except Exception as e:
        st.error(f"Error initializing vector store for {table_name}: {str(e)}")
        logging.error(f"Vector store error: {str(e)}")
        return None
    finally:
        cursor.close()
        conn.close()

def retrieve_relevant_contexts(query, vector_stores, top_k=3):
    query_embedding = get_openai_embedding(query)
    if not query_embedding:
        return [{"text": chunk, "source": table_name} for table_name, vs in vector_stores.items() for chunk in vs["chunks"][:min(top_k, len(vs["chunks"]))]]

    def process_table(table_name, vector_store):
        chunks = vector_store["chunks"]
        embeddings = vector_store["embeddings"]
        similarities = []
        for emb in embeddings:
            if len(emb) == 0:
                similarities.append(0)
                continue
            emb_array = np.array(emb).reshape(1, -1)
            query_array = np.array(query_embedding).reshape(1, -1)
            similarity = cosine_similarity(emb_array, query_array)[0][0]
            similarities.append(similarity)
        indices = np.argsort(similarities)[-top_k:][::-1]
        return [{"text": chunks[idx], "source": table_name, "score": similarities[idx]} for idx in indices]

    all_chunks = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda kv: process_table(kv[0], kv[1]), vector_stores.items())
        for table_chunks in results:
            all_chunks.extend(table_chunks)
    
    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    logging.info(f"Retrieved {len(all_chunks[:top_k])} relevant contexts for query: {query[:50]}...")
    return all_chunks[:top_k]



def process_query_with_rag(user_query, vector_stores, table_names, schema_name, database_name, column_info, table_metadata, conversation_history=None):

    # Ensure user is authenticated
    if not st.session_state.get('user_authenticated', False):
        return "You don't have access to this data. Please verify your credentials."

    # Get user credentials from session state
    user_code = st.session_state.get('user_code')
    plant_code = st.session_state.get('plant_code')

    if not user_code or not plant_code:
        logging.error("User code or plant code is missing in session state.")
        return "You don't have access to this data. Please verify your credentials."
    
    # Get user credentials from session state
    user_code = st.session_state.user_code
    plant_code = st.session_state.plant_code

    relevant_contexts = retrieve_relevant_contexts(user_query, vector_stores)
    combined_context = "\n".join([f"From {ctx['source']}: {ctx['text']}" for ctx in relevant_contexts])
    table_info = "\n".join([
        f"Table {table}: Columns: {', '.join([col[0] for col in column_info.get(table, [])])}"
        for table in table_names
    ])
    relationship_info = "\n".join([
        f"Relationship: {table} {rel.get('join_type', 'JOIN')} {rel['table']} ON {', '.join([f'{table}.{k1} = {rel['table']}.{k2}' for k1, k2 in rel['join_keys']])}"
        for table in table_names for rel in table_metadata.get(table, {}).get("relationships", [])
    ])

    prompt = f"""
    You are an assistant that helps users query and analyze OEE (Overall Equipment Effectiveness) data.

    The user is querying a Snowflake database with the following details:
    - Database: {database_name}
    - Schema: {schema_name}
    - Tables: {', '.join(table_names)}

    The tables have these columns with their data types:
    {column_info}

    Table relationships:
    {relationship_info}

    I've retrieved some relevant context from the database to help you answer:
    {combined_context}

    IMPORTANT ACCESS CONTROL INFORMATION:
    - The current user has user_code='{user_code}' and plant_code='{plant_code}'
    - This user can ONLY access data with this user_code and plant_code

    The user's current query is: "{user_query}"

    When responding, consider the conversation context and previous questions if they are provided.

    If the query is asking for data that can be retrieved with SQL:
    1. Write a SQL query to get the requested information from the appropriate tables
    2. Format your SQL code block with ```sql at the beginning and ``` at the end
    3. Make sure to write standard SQL compatible with Snowflake
    4. Use proper column names as shown above
    5. ALWAYS apply these filters to ANY query you write:
       - For machine_access_info_ai table: user_code='{user_code}' AND Plantcode='{plant_code}'
       - When joining with other tables, ensure you're filtering through the relationship with machine_access_info_ai
    6. Keep your SQL query efficient and focused on answering the specific question
    7. If the query would return data for other users or plants, add a WHERE clause to filter for this user's access only
    8. If the query cannot be satisfied with the user's access level, respond with: "You don't have access to this data. Please verify your credentials."
    9. In case the SQL query generation fails, provide a user-friendly response: "I'm having trouble understanding that request. Could you please rephrase your question?"

    If the query is not asking for data or cannot be answered with SQL:
    1. Provide a helpful explanation about OEE concepts
    2. Suggest a reformulation of their question that could be answered with the available data

    Remember, OEE (Overall Equipment Effectiveness) is a standard metric in manufacturing that measures 
    productivity by combining availability, performance, and quality metrics.
    """
    system_prompt = "You are a helpful assistant that specializes in SQL queries and OEE analytics."
    try:
        response = call_openai(prompt, system_prompt, conversation_history)
        logging.info(f"RAG response for query '{user_query}': {response[:100]}...")
        return response
    except Exception as e:
        logging.error(f"RAG processing error: {str(e)}")
        return "Please rephrase your question or try again later."







