"""
FIFA Database Agent — Streamlit Chat App
------------------------------------------
Same agent logic as fifa_agent.py, wrapped in a web chat interface.

LOCAL TESTING:
1. Create a file: .streamlit/secrets.toml (see instructions from Claude)
2. Run: streamlit run app.py

DEPLOYMENT:
Deployed via Streamlit Community Cloud, with secrets entered in the
Cloud dashboard's Secrets manager (not stored in this file or on GitHub).
"""

import streamlit as st
import anthropic
import psycopg2
import psycopg2.extras

# ── Secrets (never hardcoded — pulled from Streamlit's secrets system) ──
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
DB_CONNECTION_STRING = st.secrets["DB_CONNECTION_STRING"]
# ──────────────────────────────────────────────────────────────────────

TABLE_NAMES = ["Fifa_15", "Fifa_16", "Fifa_17", "Fifa_18",
               "Fifa_19", "Fifa_20", "Fifa_21", "Fifa_22"]

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def query_database(sql: str) -> str:
    """
    Runs a READ-ONLY SQL query against the database. Blocks anything
    that isn't a SELECT, since this app is public-facing and we never
    want a user to be able to modify or delete data through the chatbot.
    """
    cleaned = sql.strip().lower()
    if not cleaned.startswith("select"):
        return "REJECTED: Only SELECT queries are allowed through this chatbot."

    try:
        conn = psycopg2.connect(DB_CONNECTION_STRING)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return "Query ran successfully but returned no rows."

        preview = rows[:50]
        result_text = f"{len(rows)} row(s) returned. Showing up to 50:\n"
        for row in preview:
            result_text += str(dict(row)) + "\n"
        return result_text

    except Exception as e:
        return f"SQL ERROR: {e}"


tools = [
    {
        "name": "query_database",
        "description": (
            "Run a read-only SELECT SQL query against the Postgres "
            "database and return the results. Use this to explore table "
            "schemas (e.g. SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'Fifa_22') and to answer the user's "
            "question with real data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SELECT SQL query to run.",
                }
            },
            "required": ["sql"],
        },
    }
]

SYSTEM_PROMPT = f"""You are a data analyst agent with access to a Postgres
database containing FIFA player data. The database has these tables,
one per FIFA game edition: {", ".join(TABLE_NAMES)}.

You do NOT know the exact column names yet. Before answering a
question, if you're unsure of column names, first query
information_schema.columns to check the schema of the relevant
table(s), for example:
  SELECT column_name FROM information_schema.columns WHERE table_name = 'Fifa_22';

Table and column names are case-sensitive in this database — wrap
them in double quotes in your SQL, e.g. SELECT * FROM "Fifa_22" LIMIT 5;

You may only run SELECT queries — you cannot modify data, and should
not attempt to. Run multiple queries if needed (e.g. explore schema,
then query data, then refine). Once you have enough information, give
a clear, direct final answer in plain English — don't just dump raw
rows, explain what they mean.
"""


def run_agent(user_goal: str, status_area) -> str:
    messages = [{"role": "user", "content": user_goal}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [b for b in response.content if b.type == "tool_use"]

        if not tool_calls:
            final_text = "".join(
                b.text for b in response.content if b.type == "text"
            )
            return final_text

        tool_results = []
        for call in tool_calls:
            sql = call.input["sql"]
            status_area.write(f"🔍 Running: `{sql}`")
            result = query_database(sql)

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": result,
                }
            )

        messages.append({"role": "user", "content": tool_results})


# ── Streamlit UI ─────────────────────────────────────────────────
st.set_page_config(page_title="FIFA Data Agent", page_icon="⚽")
st.title("⚽ FIFA Data Agent")
st.caption("Ask a question about FIFA player data (2015–2022) and the agent will query the database to answer it.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask something, e.g. 'Which club has the highest average wage in Fifa_22?'")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        status_area = st.container()
        with st.spinner("Thinking..."):
            answer = run_agent(user_input, status_area)
        st.write(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
