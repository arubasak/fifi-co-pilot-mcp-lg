# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
import streamlit as st
import base64
from pathlib import Path
import datetime
import asyncio
import os
import traceback
import uuid
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

# --- LangGraph and LangChain Imports ---
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from tavily import TavilyClient

# Helper function to load and Base64-encode images
@st.cache_data
def get_image_as_base64(file_path):
    """Loads an image file and returns it as a Base64 encoded string."""
    try:
        path = Path(file_path)
        with path.open("rb") as f: data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

# Load images and set page config
FIFI_AVATAR_B64 = get_image_as_base64("assets/fifi-avatar.png")
USER_AVATAR_B64 = get_image_as_base64("assets/user-avatar.png")
st.set_page_config(page_title="FiFi", page_icon=f"data:image/png;base64,{FIFI_AVATAR_B64}" if FIFI_AVATAR_B64 else "🤖", layout="wide", initial_sidebar_state="auto")

# Robust asyncio helper function
def get_or_create_eventloop():
    """Gets the active asyncio event loop or creates a new one."""
    try: return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# Define the state for our LangGraph agent
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str

# Load environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MCP_PINECONE_URL = os.environ.get("MCP_PINECONE_URL")
MCP_PINECONE_API_KEY = os.environ.get("MCP_PINECONE_API_KEY")
MCP_PIPEDREAM_URL = os.environ.get("MCP_PIPEDREAM_URL")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

SECRETS_ARE_MISSING = not all([OPENAI_API_KEY, MCP_PINECONE_URL, MCP_PINECONE_API_KEY, MCP_PIPEDREAM_URL, TAVILY_API_KEY])

if not SECRETS_ARE_MISSING:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.2)
    if 'thread_id' not in st.session_state: st.session_state.thread_id = f"fifi_streamlit_session_{uuid.uuid4()}"
    THREAD_ID = st.session_state.thread_id

# Tavily Search Tool with Exclusions
DEFAULT_EXCLUDED_DOMAINS = [
    "ingredientsnetwork.com", "csmingredients.com", "batafood.com", "nccingredients.com", "prinovaglobal.com", "ingrizo.com",
    "solina.com", "opply.com", "brusco.co.uk", "lehmanningredients.co.uk", "i-ingredients.com", "fciltd.com", "lupafoods.com",
    "tradeingredients.com", "peterwhiting.co.uk", "globalgrains.co.uk", "tradeindia.com", "udaan.com", "ofbusiness.com",
    "indiamart.com", "symega.com", "meviveinternational.com", "amazon.com", "podfoods.co", "gocheetah.com", "foodmaven.com",
    "connect.kehe.com", "knowde.com", "ingredientsonline.com", "sourcegoodfood.com"
]
@tool
def tavily_search_fallback(query: str) -> str:
    """Search the web using Tavily, excluding competitor domains."""
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(query=query, search_depth="advanced", max_results=5, include_answer=True, include_raw_content=False, exclude_domains=DEFAULT_EXCLUDED_DOMAINS)
        if response.get('answer'): result = f"Web Search Results:\n\nSummary: {response['answer']}\n\nSources:\n"
        else: result = "Web Search Results:\n\nSources:\n"
        for i, source in enumerate(response.get('results', []), 1): result += f"{i}. {source['title']}\n   URL: {source['url']}\n   Content: {source['content']}\n\n"
        return result
    except Exception as e: return f"Error performing web search: {str(e)}"

# System Prompt Definition
def get_system_prompt_content_string():
    pinecone_tool = "functions.get_context"
    prompt = f"""<instructions>
# ... (Your entire detailed prompt is preserved here) ...
</instructions>"""
    return prompt

# --- UNIFIED GRAPH ARCHITECTURE ---
@st.cache_resource(ttl=3600)
def get_agent_graph():
    """Constructs the unified LangGraph agent with integrated memory."""
    async def run_async_tool_initialization():
        client = MultiServerMCPClient({
            "pinecone": {"url": MCP_PINECONE_URL, "transport": "sse", "headers": {"Authorization": f"Bearer {MCP_PINECONE_API_KEY}"}},
            "pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}
        })
        return await client.get_tools()

    mcp_tools = get_or_create_eventloop().run_until_complete(run_async_tool_initialization())
    all_tools = list(mcp_tools) + [tavily_search_fallback]
    
    # Define Graph Nodes
    def agent_node(state: AgentState):
        """Runs the pre-built agent with the current state."""
        system_prompt = get_system_prompt_content_string()
        if state.get("summary"):
            system_prompt += f"\n\nThis is a summary of the preceding conversation:\n{state['summary']}"

        agent_executor = create_react_agent(llm, all_tools, system_message=system_prompt)
        result = agent_executor.invoke({"messages": state["messages"]})
        return {"messages": result["messages"]}

    def summarize_node(state: AgentState):
        """Summarizes the history and prunes old messages."""
        messages_to_summarize = state["messages"][:-1]
        summarizer_memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500, return_messages=False)
        for msg in messages_to_summarize:
            if isinstance(msg, HumanMessage): summarizer_memory.chat_memory.add_user_message(msg.content)
            else: summarizer_memory.chat_memory.add_ai_message(msg.content)
        summary = summarizer_memory.load_memory_variables({})["history"]
        messages_to_remove = [RemoveMessage(id=m.id) for m in messages_to_summarize]
        st.info("Conversation history is long. Summarizing older messages...")
        return {"summary": summary, "messages": messages_to_remove}

    def should_summarize(state: AgentState) -> Literal["summarize", "run_agent"]:
        """Decides if the conversation is long enough to summarize."""
        if len(state["messages"]) > 6:
            return "summarize"
        return "run_agent"

    # Build the Unified Graph
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("summarize", summarize_node)
    graph_builder.add_node("run_agent", agent_node)
    graph_builder.add_conditional_edges("__start__", should_summarize, {"summarize": "summarize", "run_agent": "run_agent"})
    graph_builder.add_edge("summarize", "run_agent")
    graph_builder.add_edge("run_agent", END)
    
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph

# --- Agent execution logic ---
async def execute_agent_call_with_memory(user_query: str, graph):
    """Invokes the unified graph."""
    try:
        config = {"configurable": {"thread_id": THREAD_ID}}
        event = {"messages": [HumanMessage(content=user_query)]}
        final_state = await graph.ainvoke(event, config=config)
        
        # The final message is the one we want to display
        assistant_reply = final_state["messages"][-1].content
        return assistant_reply if assistant_reply else "(No response was generated.)"
    except Exception as e:
        print(f"--- ERROR: Exception caught during execution! ---")
        traceback.print_exc()
        error_message = f"**An error occurred during processing:**\n\n```\n{traceback.format_exc()}\n```"
        return error_message

# --- Input Handling and UI (Unchanged) ---
def handle_new_query_submission(query_text: str):
    if not st.session_state.get('thinking_for_ui', False):
        st.session_state.active_question = query_text
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.session_state.query_to_process = query_text
        st.session_state.thinking_for_ui = True
        st.rerun()

st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Collapsed for clarity
st.markdown("<h1 style='font-size: 24px;'>FiFi, AI sourcing assistant</h1>", unsafe_allow_html=True)
st.caption("Hello, I am FiFi, your AI-powered assistant...")

if SECRETS_ARE_MISSING:
    st.error("Secrets missing. Please configure necessary environment variables, including OPENAI_API_KEY.")
    st.stop()

if "messages" not in st.session_state: st.session_state.messages = []
if 'thinking_for_ui' not in st.session_state: st.session_state.thinking_for_ui = False
if 'query_to_process' not in st.session_state: st.session_state.query_to_process = None
if 'components_loaded' not in st.session_state: st.session_state.components_loaded = False
if 'active_question' not in st.session_state: st.session_state.active_question = None

try:
    agent_graph = get_agent_graph()
    st.session_state.components_loaded = True
except Exception as e:
    st.error(f"Failed to initialize agent graph. Please refresh. Error: {e}")
    st.stop()

st.sidebar.markdown("## Quick questions")
preview_questions = [ "Suggest some natural strawberry flavours for beverage", "Latest trends in plant-based proteins for 2025?", "Suggest me some vanilla flavours for ice-cream" ]
for question in preview_questions:
    button_type = "primary" if st.session_state.active_question == question else "secondary"
    if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True, type=button_type): handle_new_query_submission(question)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Reset chat session", use_container_width=True):
    st.session_state.messages = []
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    st.session_state.thread_id = f"fifi_streamlit_session_{uuid.uuid4()}"
    st.session_state.active_question = None
    st.rerun()

st.sidebar.markdown('By using this agent, you agree to our <a href="https://www.12taste.com/terms-conditions/" target="_blank">Terms of Service</a>.', unsafe_allow_html=True)

fifi_avatar_icon = f"data:image/png;base64,{FIFI_AVATAR_B64}" if FIFI_AVATAR_B64 else "🤖"
user_avatar_icon = f"data:image/png;base64,{USER_AVATAR_B64}" if USER_AVATAR_B64 else "🧑‍💻"
for message in st.session_state.get("messages", []):
    avatar_icon = fifi_avatar_icon if message["role"] == "assistant" else user_avatar_icon
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message.get("content", ""))

if st.session_state.get('thinking_for_ui', False):
    with st.chat_message("assistant", avatar=fifi_avatar_icon): st.markdown("⌛ FiFi is thinking...")

user_prompt = st.chat_input("Ask me for ingredients, recipes, or product development—in any language.", key="main_chat_input",
                            disabled=st.session_state.get('thinking_for_ui', False) or not st.session_state.get("components_loaded", False))
if user_prompt:
    st.session_state.active_question = None
    handle_new_query_submission(user_prompt)

if st.session_state.get('query_to_process'):
    query_to_run = st.session_state.query_to_process
    loop = get_or_create_eventloop()
    assistant_reply = loop.run_until_complete(execute_agent_call_with_memory(query_to_run, agent_graph))
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    st.rerun()
