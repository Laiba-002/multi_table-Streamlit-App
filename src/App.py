import streamlit as st
import snowflake.connector
import pandas as pd
import numpy as np
import os
from datetime import datetime
import re
import json
from io import StringIO
from prompts import generate_introduction, get_llm_response, call_openai
from rag_utils import (
    initialize_vector_store,
    process_query_with_rag,
    get_openai_embedding,
    infer_user_intent
)
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tiktoken
import logging
import jwt  # PyJWT
import speech_recognition as sr

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
APP_ENV = st.secrets.get("APP_ENV", os.getenv("APP_ENV", "development"))

# Safe full paths to avatars
BASE_DIR = Path(__file__).parent
user_avatar = (BASE_DIR / "user.png").resolve().as_posix()
assistant_avatar = (BASE_DIR / "Assistant.png").resolve().as_posix()

# Set page config
st.set_page_config(
    page_title="O3 Agent",
    page_icon="📊",
    layout="wide",
    
)
# OpenAI client
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
GPT_MODEL = "gpt-4o-mini"

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'has_started' not in st.session_state:
    st.session_state.has_started = False
# if 'user_code' not in st.session_state:
    # st.session_state.user_id = 'A-1704902113649'  # replace with actual user ID
    # st.session_state.plantcode = 'O-2372838648201'     
if 'full_responses' not in st.session_state:
    st.session_state.full_responses = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'dfs' not in st.session_state:
    st.session_state.dfs = {}
if 'vector_stores' not in st.session_state:
    st.session_state.vector_stores = {}
if 'embedding_status' not in st.session_state:
    st.session_state.embedding_status = "Not Started"
if 'show_history' not in st.session_state:
    st.session_state.show_history = True
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'table_names' not in st.session_state:
    st.session_state.table_names = ["OEESHIFTWISE_AI","MACHINE_ACCESS_INFO_AI"]  # Add your table names here
if 'schema_columns' not in st.session_state:
    st.session_state.schema_columns = {}
if 'selected_history_index' not in st.session_state:
    st.session_state.selected_history_index = None
if 'user_code' not in st.session_state:
    st.session_state.user_code = None    
if 'snowflake_conn' not in st.session_state:
    st.session_state.snowflake_conn = None

if 'table_metadata' not in st.session_state:
    st.session_state.table_metadata = {
        "OEESHIFTWISE_AI": {
            "description": "Shift-level OEE performance data",
            "key_columns": ["PUID", "ShiftStartTime", "ShiftEndTime"],
            "relationships": [
                # {
                #     "table": "OEEBREAKDOWNAI",
                #     "join_keys": [
                #         ("PUID", "PUID"),
                #         ("ShiftStartTime", "ShiftStartTime"),
                #         ("ShiftEndTime", "ShiftEndTime")
                #     ],
                #     # "join_type": "INNER JOIN"
                # },
                {
                    "table": "machine_access_info_ai",
                    "join_keys": [
                        ("PUID", "PUID")
                    ],
                    "join_type": "INNER JOIN"
                }
            ]
        },
        # "OEEBREAKDOWNAI": {
        #     "description": "Breakdown and downtime data",
        #     "key_columns": ["PUID", "ShiftStartTime", "ShiftEndTime"],
        #     "relationships": [{
        #         "table": "OEESHIFTWISE_AI",
        #         "join_keys": [
        #             ("PUID", "PUID"),
        #             ("ShiftStartTime", "ShiftStartTime"),
        #             ("ShiftEndTime", "ShiftEndTime")
        #         ],
        #         "join_type": "INNER JOIN"
        #     }]
        # },
        "machine_access_info_ai": {
            "description": "User-level machine and plant access info",
            "key_columns": ["PUID", "user_code", "Plantcode", "groupcode"],
            "relationships": [{
                "table": "OEESHIFTWISE_AI",
                "join_keys": [
                    ("PUID", "PUID")
                ],
                "join_type": "INNER JOIN"
            }]
        }
    }
if 'query_cache' not in st.session_state:
    st.session_state.query_cache = {}
if "token" not in st.session_state:
    st.session_state.token = ""
# Get query parameters using st.query_params
query_params = st.query_params
token = query_params.get("token", [None])  # Ensure token is a string or None
# Store the token in session state if it exists
st.session_state.token = token if token else None     # Always update session state

def decode_jwt_token(token):
    try:
        # Ensure token is not None and is a string or bytes
        if not token:
            logging.error("Token is missing or None.")
            return None, None, None
        logging.info(f"Token before decoding: {token}")

        # Check if the token has three segments
        if len(token.split(".")) != 3:
            logging.error("Invalid token format: Not enough segments.")
            return None, None, None

        if isinstance(token, str):
            token = token.encode("utf-8")  # Convert string to bytes
        # Decode the token 
        decoded_token = jwt.decode(token, options={"verify_signature": False})
        logging.info(f"Decoded token: {decoded_token}")

        # Extract user information
        user_code = decoded_token.get("userCode")
        plant_code = decoded_token.get("plantCode")
        if not user_code or not plant_code:
            logging.error("Decoded token is missing userCode or plantCode.")
        else:    
            logging.info(f"Extracted user_code: {user_code}, plant_code: {plant_code}")

        return user_code, plant_code, decoded_token
    except Exception as e:
        logging.error(f"Error decoding JWT token: {str(e)}")
        return None, None, None


def initialize_user_session():
    if 'user_code' not in st.session_state or 'plant_code' not in st.session_state:
        token = st.session_state.get("token")
        user_code, plant_code, decoded_token = decode_jwt_token(token)
        st.session_state.user_code = user_code
        st.session_state.plant_code = plant_code
        if user_code and plant_code:
          st.session_state.user_authenticated = True 
          st.session_state.user_info = decoded_token
          logging.info(f"User authenticated: {user_code}, Plant: {plant_code}")    
        else:
            st.session_state.user_authenticated = False
            logging.error("User authentication failed. Invalid token.")

def initialize_app():
    # This function will be called when the app starts
    if 'user_authenticated' not in st.session_state:
        initialize_user_session()  # From the code above

# Call initialization
initialize_app()       
def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception as e:
        logging.error(f"Token counting error: {str(e)}")
        return len(text.split())

def log_token_usage(nlp_tokens, table_tokens, viz_tokens):
    total_tokens = nlp_tokens + table_tokens + viz_tokens
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nlp_tokens": nlp_tokens,
        "table_tokens": table_tokens,
        "viz_tokens": viz_tokens,
        "total_tokens": total_tokens
    }
    os.makedirs("logs", exist_ok=True)
    try:
        with open("logs/token_usage_log.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data) + "\n")
    except Exception as e:
        logging.error(f"Error writing to token usage log: {str(e)}")



def audio_to_text():
    """Convert audio input to text using speech recognition."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        # st.info("Listening... Please speak into the microphone.")
        try:
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio)
            # st.success(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    return ""

@st.cache_resource
def init_snowflake_connection():
    if st.session_state.snowflake_conn:
        return st.session_state.snowflake_conn
    try:
        conn = snowflake.connector.connect(
            user=st.secrets["snowflake"]["user"],
            password=st.secrets["snowflake"]["password"],
            account=st.secrets["snowflake"]["account"],
            warehouse=st.secrets["snowflake"]["warehouse"],
            database="O3_AI_DB",
            schema="O3_AI_DB_SCHEMA"
        )
        st.session_state.snowflake_conn = conn
        logging.info("Snowflake connection initialized")
        return conn
    except Exception as e:
        st.error(f"Error connecting to Snowflake: {str(e)}")
        logging.error(f"Snowflake connection error: {str(e)}")
        return None

# @st.cache_data(show_spinner=False, ttl=3600)
def execute_snowflake_query(query):
    conn = init_snowflake_connection()
    if not conn:
        return None
    # Add security check to ensure all queries filter by user access
    # if "machine_access_info_ai" not in query.upper() and "WHERE" in query.upper() and st.session_state.user_code:
    #     # Add appropriate access control to query
        # query = query.replace("WHERE", f"WHERE machine_access_info_ai.user_code='{st.session_state.user_code}' AND machine_access_info_ai.Plantcode='{st.session_state.plant_code}' AND ")
    query_hash = hash(query)
    if query_hash in st.session_state.query_cache:
        logging.info(f"Cache hit for query: {query}")
        return st.session_state.query_cache[query_hash]
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(result, columns=columns)
        cursor.close()
        global table_tokens
        table_tokens = count_tokens(df.head(10).to_csv(index=False))
        st.session_state.query_cache[query_hash] = df
        logging.info(f"Executed and cached query: {query}")
        return df
    except Exception as e:
        st.error("Error executing query. Please rephrase your question or try again later.")
        logging.error(f"Query execution error: {str(e)} - Query: {query}")
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def determine_visualization_type(user_query, sql_query, result_df_str):
    try:
        result_df = pd.read_json(StringIO(result_df_str))
        vis_prompt = f"""
        I need to visualize the following SQL query results for the question: "{user_query}"
        The SQL query was:
        ```sql
        {sql_query}
        ```
        The query returned {len(result_df)} rows with columns:
        {[(col, str(result_df[col].dtype)) for col in result_df.columns]}
        Respond with a JSON object:
        {{
            "viz_type": "line|bar|scatter|pie|histogram|heatmap|none",
            "x_column": "x-axis column",
            "y_column": "y-axis column",
            "color_column": "color column (optional, can be null)",
            "title": "Visualization title",
            "description": "Rationale for visualization choice"
        }}
        """
        system_prompt = "You are a data visualization expert."
        vis_response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": vis_prompt}
            ],
            response_format={"type": "json_object"}
        )
        vis_recommendation = json.loads(vis_response.choices[0].message.content)
        global viz_tokens
        viz_tokens = count_tokens(vis_recommendation.get("description", ""))
        logging.info(f"Visualization type determined: {vis_recommendation['viz_type']}")
        return vis_recommendation
    except Exception as e:
        st.warning("Could not determine visualization type.")
        logging.error(f"Visualization type error: {str(e)}")
        return {"viz_type": "none"}

def create_visualization(result_df, vis_recommendation):
    try:
        viz_type = vis_recommendation.get("viz_type", "none")
        if viz_type == "none" or len(result_df) == 0:
            return None
        x_col = vis_recommendation.get("x_column")
        y_col = vis_recommendation.get("y_column")
        color_col = vis_recommendation.get("color_column")
        title = vis_recommendation.get("title", "Data Visualization")
        available_cols = result_df.columns.tolist()
        if x_col and x_col not in available_cols:
            x_col = available_cols[0] if available_cols else None
        if y_col and y_col not in available_cols:
            y_col = available_cols[1] if len(available_cols) > 1 else None
        if color_col and color_col not in available_cols:
            color_col = None
        if not x_col or not y_col:
            return None
        if viz_type == "bar":
            if len(result_df) > 25:
                result_df = result_df.groupby(x_col, as_index=False)[y_col].agg('sum')
            fig = px.bar(result_df, x=x_col, y=y_col, color=color_col, title=title)
        elif viz_type == "line":
            if pd.api.types.is_datetime64_any_dtype(result_df[x_col]):
                result_df = result_df.sort_values(by=x_col)
            fig = px.line(result_df, x=x_col, y=y_col, color=color_col, title=title, markers=True)
        elif viz_type == "scatter":
            fig = px.scatter(result_df, x=x_col, y=y_col, color=color_col, title=title)
        elif viz_type == "pie":
            fig = px.pie(result_df, names=x_col, values=y_col, title=title)
        elif viz_type == "histogram":
            fig = px.histogram(result_df, x=x_col, title=title)
        elif viz_type == "heatmap":
            if color_col:
                pivot_df = result_df.pivot_table(values=color_col, index=y_col, columns=x_col, aggfunc='mean')
                fig = px.imshow(pivot_df, title=title)
            else:
                return None
        else:
            return None
        fig.update_layout(
            template="plotly_dark",
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        logging.info(f"Visualization created: {viz_type}")
        return fig
    except Exception as e:
        st.warning("Could not create visualization.")
        logging.error(f"Visualization creation error: {str(e)}")
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def generate_nlp_summary(user_query, sql_query, result_df_str):
    try:
        result_df = pd.read_json(StringIO(result_df_str))
        nlp_summary_prompt = f"""
        Summarize the following SQL query results for the question: "{user_query}"
        SQL Query:
        ```sql
        {sql_query}
        ```
        Results ({len(result_df)} rows):
        {result_df.to_string(index=False, max_rows=10)}
        Provide a 1-2 sentence summary answering the user's question, focusing on key metrics or trends.
        """
        nlp_summary = call_openai(nlp_summary_prompt, "You are a data analyst.")
        global nlp_tokens
        nlp_tokens = count_tokens(nlp_summary)
        logging.info(f"NLP summary generated: {nlp_summary[:50]}...")
        return nlp_summary
    except Exception as e:
        st.warning("Could not generate summary.")
        logging.error(f"NLP summary error: {str(e)}")
        return "Unable to generate summary."

# Sidebar
with st.sidebar:
    snowflake_creds = st.secrets.get("snowflake")
    if snowflake_creds and all(k in snowflake_creds for k in ["user", "password", "account", "warehouse"]):
        if not st.session_state.initialized:
            with st.spinner("Connecting to Snowflake..."):
                if not st.secrets.get("OPENAI_API_KEY"):
                    st.error("OpenAI API key missing.")
                    st.stop()
                conn = init_snowflake_connection()
                if conn:
                    with st.spinner("Fetching sample data..."):
                        # Batch fetch sample data
                        cursor = conn.cursor()
                        try:
                            for table_name in st.session_state.table_names:
                                sample_query = f"SELECT * FROM {table_name} limit 1000"
                                df = execute_snowflake_query(sample_query)
                                if df is not None:
                                    st.session_state.dfs[table_name] = df
                                    st.session_state.schema_columns[table_name] = list(df.columns)
                            st.session_state.initialized = True
                            with st.spinner("Checking embeddings..."):
                                st.session_state.embedding_status = "In Progress"
                                for table_name in st.session_state.table_names:
                                    st.session_state.vector_stores[table_name] = initialize_vector_store(st.session_state.dfs[table_name].to_json(), table_name)  
                                if all(st.session_state.vector_stores.values()):
                                    st.session_state.embedding_status = "Completed"
                                    # st.success("Embeddings loaded/created successfully!")
                                else:
                                    st.session_state.embedding_status = "Failed"
                                    st.warning("Embedding initialization failed.")
                        finally:
                            cursor.close()
        st.markdown("""
        <style>
        button[kind="secondary"] {
            background-color: transparent !important;
            border: none !important;
            font-size: 12px !important;
            color: #333 !important;
            padding: 2px 4px !important;
        }
        button[kind="secondary"]:hover {
            background-color: #f0f0f0 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.header("Chat History")
        st.text('Today')
        for i, resp in reversed(list(enumerate(st.session_state.full_responses))):
            query_preview = f"{resp['user_query'][:50]}{'...' if len(resp['user_query']) > 50 else ''}"
            if st.sidebar.button(query_preview, key=f"history_btn_{i}"):
                st.session_state.selected_history_index = i
                st.rerun()
        st.divider()
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.full_responses = []
            st.session_state.selected_history_index = None
            st.session_state.query_cache = {}
            st.success("Chat history and cache cleared!")
            st.rerun()
    else:
        st.error("Snowflake credentials missing.")

# Chat interface
if st.session_state.initialized:
    chat_container = st.container()
    with chat_container:
        if st.session_state.selected_history_index is not None:
            selected_response = st.session_state.full_responses[st.session_state.selected_history_index]
            with st.chat_message("user", avatar=user_avatar):
                st.markdown(f"<div style='color: white;'>{selected_response.get('user_query', '')}</div>", unsafe_allow_html=True)
            with st.chat_message("assistant", avatar=assistant_avatar):
                st.markdown(f"<div>{selected_response.get('text_response', '')}</div>", unsafe_allow_html=True)
                if selected_response.get("data") is not None:
                    st.dataframe(selected_response["data"])
                if selected_response.get("visualization") is not None:
                    st.plotly_chart(selected_response["visualization"], use_container_width=True)
        else:
            for idx, resp in enumerate(st.session_state.full_responses):
                with st.chat_message("user", avatar=user_avatar):
                    st.markdown(f"<div style='color: black;'>{resp.get('user_query', '')}</div>", unsafe_allow_html=True)
                with st.chat_message("assistant", avatar=assistant_avatar):
                    st.markdown(f"<div>{resp.get('text_response', '')}</div>", unsafe_allow_html=True)
                    if resp.get("data") is not None:
                        st.dataframe(resp["data"])
                    if resp.get("visualization") is not None:
                        st.plotly_chart(resp["visualization"], use_container_width=True)
            if not st.session_state.has_started:
                with st.chat_message("assistant", avatar=assistant_avatar):
                    st.write("Hi! How can I help you with OEE data?")
    
    
    # This creates empty space that will fill the area between messages and input
    # st.markdown('<div class="spacer" style="flex: 1;"></div>', unsafe_allow_html=True)

    # This empty element takes up available space
    for _ in range(25):  # Adjust this number as needed
      st.write("")
    
    input_container = st.container()
    # Add audio input button
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.chat_input("Ask about OEE data")
    with col2:
       
        if st.button("🎙️"):
            audio_query = audio_to_text()
            if audio_query:
                user_query = audio_query


    if user_query:
        st.session_state.has_started = True
        st.session_state.selected_history_index = None
        st.session_state.messages.append({"role": "user", "content": user_query})
        if st.session_state.show_history:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
    # if user_query := st.chat_input("Ask about OEE data"):
    #     st.session_state.has_started = True
    #     st.session_state.selected_history_index = None
    #     st.session_state.messages.append({"role": "user", "content": user_query})
    #     if st.session_state.show_history:
    #         st.session_state.chat_history.append({"role": "user", "content": user_query})

        with chat_container:
            with st.chat_message("user", avatar=user_avatar):
                st.markdown(f"<div style='color: black;'>{user_query}</div>", unsafe_allow_html=True)
            with st.spinner("Generating response..."):
                # Infer user intent
                intent = infer_user_intent(user_query, st.session_state.table_names)
                logging.info(f"User query: {user_query}, Intent: {intent}")

                column_info = {
                    table_name: [
                        (col, str(dtype)) for col, dtype in zip(
                            st.session_state.schema_columns[table_name],
                            st.session_state.dfs[table_name].dtypes
                        )
                    ] for table_name in st.session_state.dfs.keys()
                }
                conversation_history = st.session_state.chat_history if st.session_state.show_history else None

                if st.session_state.embedding_status == "Completed":
                    rag_response = process_query_with_rag(
                        user_query=user_query,
                        vector_stores=st.session_state.vector_stores,
                        table_names=intent["tables"],
                        schema_name="O3_AI_DB_SCHEMA",
                        database_name="O3_AI_DB",
                        column_info=column_info,
                        table_metadata=st.session_state.table_metadata,
                        conversation_history=conversation_history
                    )

                    if "```sql" in rag_response:
                        sql_query = rag_response.split("```sql")[1].split("```")[0].strip()
                        result_df = execute_snowflake_query(sql_query)
                        if result_df is not None and not result_df.empty:
                            result_df_str = result_df.to_json(orient="records", indent=2)
                            nlp_summary = generate_nlp_summary(user_query, sql_query, result_df_str)
                            final_response = f"{nlp_summary}\n\nDetailed results:\n" if not st.session_state.debug_mode else \
                                            f"SQL Query:\n```sql\n{sql_query}\n```\n{nlp_summary}\n\nDetailed results:\n"
                            vis_recommendation = determine_visualization_type(user_query, sql_query, result_df_str)
                            fig = create_visualization(result_df, vis_recommendation)

                            with st.chat_message("assistant", avatar=assistant_avatar):
                                st.markdown(f"<div>{final_response}</div>", unsafe_allow_html=True)
                                st.dataframe(result_df)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.caption(f"Visualization notes: {vis_recommendation.get('description', '')}")

                            st.session_state.messages.append({"role": "assistant", "content": final_response})
                            st.session_state.full_responses.append({
                                "user_query": user_query,
                                "text_response": final_response,
                                "data": result_df,
                                "visualization": fig,
                                "sql_query": sql_query if st.session_state.debug_mode else None
                            })
                            if st.session_state.show_history:
                                st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                            log_token_usage(nlp_tokens, table_tokens, viz_tokens)
                        else:
                            no_data_msg = "No results found. Please rephrase your question."
                            with st.chat_message("assistant", avatar=assistant_avatar):
                                st.warning(no_data_msg)
                            st.session_state.messages.append({"role": "assistant", "content": no_data_msg})
                            st.session_state.full_responses.append({
                                "user_query": user_query,
                                "text_response": no_data_msg,
                                "data": None,
                                "visualization": None,
                                "sql_query": sql_query if st.session_state.debug_mode else None
                            })
                    else:
                        with st.chat_message("assistant", avatar=assistant_avatar):
                            st.markdown(f"<div>{rag_response}</div>", unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": rag_response})
                        st.session_state.full_responses.append({
                            "user_query": user_query,
                            "text_response": rag_response,
                            "data": None,
                            "visualization": None
                        })
                else:
                    llm_response = get_llm_response(
                        user_query=user_query,
                        table_name=", ".join(st.session_state.table_names),
                        schema_name="O3_AI_DB",
                        database_name="O3_AI_DB_SCHEMA",
                        column_info=column_info,
                        conversation_history=conversation_history
                    )
                    with st.chat_message("assistant", avatar=assistant_avatar):
                        st.markdown(f"<div>{llm_response}</div>", unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": llm_response})
                    st.session_state.full_responses.append({
                        "user_query": user_query,
                        "text_response": llm_response,
                        "data": None,
                        "visualization": None
                    })

        st.rerun()
    
        
else:
    st.info("Please connect to Snowflake to use the chatbot.")










