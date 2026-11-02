"""Streamlit UI. Talks to the FastAPI backend over HTTP.

Run:  streamlit run frontend/streamlit_app.py
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

EXAMPLES = [
    "What is the total revenue?",
    "Which product generated the highest revenue?",
    "What is the monthly revenue trend?",
    "Which region generated the highest sales?",
    "How did revenue change between January and February 2026?",
]

st.set_page_config(page_title="Mini AI BI Assistant", page_icon="📊", layout="wide")
st.title("📊 Mini AI Business Intelligence Assistant")
st.caption(
    "Ask a question about the sales data in plain English. The question is interpreted by an LLM "
    "(or a keyword fallback), but every number comes from a SQL query on the database."
)


# ------------------------------------------------------------------ helpers
def call_api(method: str, path: str, **kwargs):
    """Return (status_code, json) or (None, error message) if the backend is unreachable."""
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=30, **kwargs)
        return response.status_code, response.json()
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach the backend at {API_URL}. Start it with: uvicorn app.main:app --reload"
    except requests.exceptions.RequestException as exc:
        return None, f"Request failed: {exc}"
    except ValueError:
        return None, "The backend returned a response that is not valid JSON."


def use_example(question: str):
    st.session_state["question"] = question
    st.session_state["run_now"] = True


# ------------------------------------------------------------------ sidebar: dataset overview
with st.sidebar:
    st.header("Dataset overview")
    status, body = call_api("GET", "/dataset/summary")
    if status == 200 and body.get("summary"):
        s = body["summary"]
        col1, col2 = st.columns(2)
        col1.metric("CSV records", s["total_records"])
        col2.metric("Accepted", s["accepted_records"])
        col1.metric("Completed", s["completed_records"])
        col2.metric("Returned", s["returned_records"])
        st.metric("Rejected", s["rejected_records"])
        st.caption(f"Source: {s['source_file']}")
        if body["rejected_records"]:
            with st.expander("Why were records rejected?"):
                for r in body["rejected_records"]:
                    st.write(f"**{r['transaction_id']}** (CSV row {r['source_row']}): {r['reason']}")
        st.caption("Revenue = quantity × unit price − discount, completed sales only. Returned sales are excluded.")
    elif status is None:
        st.error(body)
    else:
        st.warning("Dataset summary is not available.")

    status, health = call_api("GET", "/health")
    if status == 200:
        if health["interpreter"] == "llm":
            st.success(f"Question interpreter: LLM ({health['llm_provider']})")
        else:
            st.info("Question interpreter: keyword fallback (no LLM configured)")

# ------------------------------------------------------------------ main: ask a question
st.subheader("Try an example")
for example in EXAMPLES:
    st.button(example, on_click=use_example, args=(example,), key=f"ex_{example}")

with st.form("ask_form"):
    question = st.text_input("Your question", key="question", placeholder="e.g. Which region generated the highest sales?")
    submitted = st.form_submit_button("Ask")

if submitted or st.session_state.pop("run_now", False):
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Working on it..."):
            status, body = call_api("POST", "/ask", json={"question": question})

        if status is None:
            st.error(body)
        elif body.get("success"):
            st.success(body["answer"])

            headline = {k: v for k, v in body["data"].items() if k.endswith("_formatted")}
            if headline:
                cols = st.columns(min(len(headline), 4))
                for col, (key, value) in zip(cols, list(headline.items())[:4]):
                    col.metric(key.replace("_formatted", "").replace("_", " ").title(), value)

            rows = body.get("supporting_table") or []
            if rows:
                df = pd.DataFrame(rows)
                if body["intent"] == "monthly_revenue_trend":
                    # month_key is YYYY-MM, so the chart sorts chronologically (not alphabetically by month name)
                    st.bar_chart(df.set_index("month_key")["revenue"])
                    df = df.drop(columns=["month_key"])
                if "revenue_display" in df.columns:
                    df["revenue"] = df.pop("revenue_display")
                st.subheader("Supporting data")
                st.dataframe(df, hide_index=True)

            st.caption(body["explanation"])
            if body["interpreter"] == "llm":
                st.caption("Interpreted by: LLM (Gemini)")
            else:
                st.caption(f"Interpreted by: keyword fallback. {body.get('interpreter_note') or ''}")
            with st.expander("Raw API response"):
                st.json(body)
        else:
            st.error(body.get("answer", "The question could not be answered."))
            suggestions = (body.get("data") or {}).get("supported_questions")
            if suggestions:
                st.write("Questions I can answer:")
                for s in suggestions:
                    st.write(f"- {s}")
