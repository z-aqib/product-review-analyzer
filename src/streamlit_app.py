import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001/recommend")


st.set_page_config(page_title="Product Review & Recommendation Chat", page_icon="🤖")

st.title("🤖 Product Review & Recommendation Assistant")

# ---- Session state to store chat history ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Chat history display ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---- Chat input ----
user_query = st.chat_input("Ask me for reviews or product advice...")

if user_query:
    # 1) Show + store user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # 2) Call FastAPI instead of run_pipeline directly
    with st.spinner("Thinking..."):
        user_id = "AG3D6O4STAQKAY2UVGEUV46KN35Q"  # hardcoded for now
        payload = {"user_id": user_id, "user_query": user_query}

        try:
            resp = requests.post(API_URL, json=payload)
            data = resp.json()

            if resp.status_code != 200:
                # ---- FRIENDLY HANDLING FOR GUARDRail ERRORS ----
                detail = data.get("detail", {}) if isinstance(data, dict) else {}

                error_type = detail.get("error")
                kind = detail.get("kind")
                message = detail.get("message", "")

                # Prompt injection blocked
                if error_type == "unsafe_input" and kind == "input_prompt_injection":
                    answer = (
                        "❌ Your message was blocked by the safety guardrails.\n\n"
                        "It looks like it contains instructions trying to override or bypass "
                        "the system rules (for example, telling me to ignore previous instructions "
                        "or act as an unfiltered model).\n\n"
                        "Please rephrase your question in a normal way, and I’ll be happy to help 🙂"
                    )

                # PII blocked (if you added that rule)
                elif error_type == "unsafe_input" and "pii" in (kind or ""):
                    answer = (
                        "❌ Your message was blocked because it includes personal or sensitive "
                        "information (like your email or phone number).\n\n"
                        "For safety reasons, I’m not allowed to process queries that contain this. "
                        "Please remove such details and try again."
                    )

                # Generic fallback for any other 4xx/5xx
                else:
                    answer = (
                        f"⚠️ The server could not process your request (status {resp.status_code}).\n\n"
                        f"More info: {message or detail or 'No additional details provided.'}"
                    )

            else:
                # ---- Normal success ----
                answer = data.get("final_answer", "No answer returned.")

        except Exception as e:
            answer = f"⚠️ Request failed: {e}"

    # 3) Show assistant response in chat
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)

    # 4) (Optional but NICE for MLOps report) show debug panels
    if "resp" in locals() and resp.status_code == 200:
        with st.expander("🔍 ML Candidates"):
            st.json(data.get("ml_candidates"))

        with st.expander("📚 RAG Result"):
            st.json(data.get("rag_result"))

        with st.expander("🛡 Guardrail Events"):
            st.json(data.get("guardrail_events"))
