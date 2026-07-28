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


@st.cache_resource(ttl=3600)  # re-check schema at most once per hour
def get_database_schema() -> str:
    """
    Discovers all tables and their columns ONCE (cached across all users
    and questions for up to an hour) instead of making the agent
    rediscover this from scratch on every single question. This saves
    tool-call round trips, tokens, and money.
    """
    try:
        conn = psycopg2.connect(DB_CONNECTION_STRING)
        cur = conn.cursor()

        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' ORDER BY table_name;
        """)
        table_names = [row[0] for row in cur.fetchall()]

        schema_lines = []
        for table in table_names:
            cur.execute("""
                SELECT column_name, data_type FROM information_schema.columns
                WHERE table_name = %s ORDER BY ordinal_position;
            """, (table,))
            columns = cur.fetchall()
            col_list = ", ".join(f"{c[0]} ({c[1]})" for c in columns)
            schema_lines.append(f'Table "{table}": {col_list}')

        cur.close()
        conn.close()
        return "\n".join(schema_lines)

    except Exception as e:
        # If schema discovery fails at startup, fall back to letting the
        # agent discover it live per-question rather than crashing the app
        return f"(Schema auto-discovery failed: {e}. Agent should discover schema manually via information_schema queries.)"


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


def build_system_prompt(schema_text: str) -> str:
    return f"""You are a data analyst agent with access to a Postgres
database containing FIFA VIDEO GAME player data, roughly one table per
FIFA game edition (around editions 15 through 22).

Here is the ACTUAL current database schema (already discovered for you,
you do NOT need to query information_schema — go straight to querying
real data):

{schema_text}

SCOPE — READ CAREFULLY: This database contains player attributes and
ratings from the FIFA video game series (overall rating, pace, wages,
club, position, etc.) for each game edition. It does NOT contain:
- Real-world match results, tournament outcomes, or World Cup winners
- Real-world team standings, fixtures, or scores
- Any data outside player/club attributes as rated in the video game

If a question is about real-world football events (e.g. "who won the
World Cup in 2022") rather than video game player data, do NOT query
the database for it — you will not find it there. Instead, politely
tell the user this database only covers FIFA video game player
ratings/attributes, not real-world match or tournament results.

Use the schema above directly — write SELECT queries against the exact
table and column names shown. If a query returns 0 rows or an error, do
NOT repeat the exact same query again — try a different approach.

You may only run SELECT queries — you cannot modify data, and should
not attempt to. Once you have enough information, give a clear, direct
final answer in plain English — don't just dump raw rows, explain what
they mean.

CRITICAL RULE: You must ONLY answer using data actually returned by
the query_database tool. If a query fails, errors out, or the database
is unreachable, you must NOT fall back on your own general knowledge
about FIFA or players. Instead, tell the user plainly and honestly
that you could not retrieve the data and why (e.g. "the database
connection failed"). Never present a guessed or remembered answer as
if it came from the database.
"""


MAX_ITERATIONS = 8  # hard safety cap so the agent can never loop forever


def run_agent(user_goal: str, status_area) -> str:
    schema_text = get_database_schema()  # cached — instant after first call
    system_prompt = build_system_prompt(schema_text)

    messages = [{"role": "user", "content": user_goal}]
    seen_queries = set()

    for iteration in range(MAX_ITERATIONS):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=system_prompt,
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

            # If it tries the exact same query again, don't run it — tell it to stop repeating
            normalized = sql.strip().lower()
            if normalized in seen_queries:
                status_area.write(f"⚠️ Blocked repeated query: `{sql}`")
                result = "You already ran this exact query with no new outcome. Try a different query or give your best answer with what you have so far."
            else:
                seen_queries.add(normalized)
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

    return "I wasn't able to find a confident answer within my query limit. Try rephrasing your question, or it's possible the data needed isn't in the database."


# ── Streamlit UI ─────────────────────────────────────────────────
st.set_page_config(page_title="FIFA Data Agent", page_icon="⚽")
st.title("⚽ FIFA Data Agent")
st.caption("Ask a question about FIFA **video game** player data (2015–2022) and the agent will query the database to answer it.")

MAX_QUESTIONS_PER_SESSION = 10  # basic cost guardrail for a public app

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
    st.warning("You've reached the question limit for this session. Please refresh the page to start a new session.")
else:
    user_input = st.chat_input("Ask something, e.g. 'Which club has the highest average wage in Fifa_22?'")

    if user_input:
        st.session_state.question_count += 1
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            status_area = st.container()
            with st.spinner("Thinking..."):
                answer = run_agent(user_input, status_area)
            st.write(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
