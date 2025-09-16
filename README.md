# Hotel-QA-Agent
An interactive Hotel Question Answering Agent built with LangGraph, Streamlit, and a local LLM (Ollama). The agent can answer natural language queries like:  “Find 5 hotels in Paris with star rating ≥ 4”  “List top 10 hotels in Spain by comfort”  “What are the best rated hotels in Tokyo?”

**Features**

Uses LangGraph to orchestrate query parsing and tool execution.

A single custom tool (query_hotels) filters hotel data from a CSV dataset.

Filters by city, country, star rating, cleanliness, comfort, and facilities.

Results displayed in a clean Markdown table within Streamlit chat UI.

Runs locally with Ollama (no API cost), or can fall back to OpenAI if configured.

**Tech Stack**

Python

Streamlit

LangGraph

Pandas

Ollama (local LLM)
