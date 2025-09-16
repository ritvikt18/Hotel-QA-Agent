import streamlit as st
import traceback

st.set_page_config(page_title="🏨 Hotel QA Agent")
st.title("🏨 Hotel QA Agent")

try:
    from agent import build_agent
except Exception:
    st.error("Failed to import agent.py:")
    st.code(traceback.format_exc())
    st.stop()

if "agent" not in st.session_state:
    try:
        st.session_state["agent"] = build_agent()
    except Exception:
        st.error("Failed to build the agent:")
        st.code(traceback.format_exc())
        st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Ask e.g. 'Find 5 hotels in Paris with star rating ≥ 4'")
if user_text:
    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        try:
            state = {"messages": [("user", user_text)]}
            result = st.session_state["agent"].invoke(state)
            msgs = result.get("messages", [])
            if not msgs:
                reply = "No response."
            else:
                last = msgs[-1]
                if hasattr(last, "content"):
                    reply = last.content
                elif isinstance(last, tuple) and len(last) == 2:
                    reply = last[1]
                else:
                    reply = str(last)
        except Exception:
            reply = "Agent error:\n\n" + traceback.format_exc()

        st.markdown(reply)
        st.session_state["messages"].append({"role": "assistant", "content": reply})
