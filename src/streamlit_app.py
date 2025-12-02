import streamlit as st
from pipeline import run_pipeline

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
    # Store the user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # ---- Pipeline execution ----
    with st.spinner("Thinking..."):
        try:
            # TEMP: Hardcoded user ID — later you can make login
            user_id = "AG3D6O4STAQKAY2UVGEUV46KN35Q"

            output = run_pipeline(user_id=user_id, user_query=user_query)
            answer = output["final_answer"]

        except Exception as e:
            answer = f"⚠️ An error occurred: {e}"

    # Display assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
