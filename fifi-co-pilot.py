# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
import streamlit as st
import base64
from pathlib import Path
import asyncio
import os
import traceback
import uuid
from typing import Annotated, List, Literal, Dict
from typing_extensions import TypedDict

# --- LangGraph and LangChain Imports ---
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from tavily import TavilyClient
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message

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

FIFI_AVATAR_B64 = get_image_as_base64("assets/fifi-avatar.png")
USER_AVATAR_B64 = get_image_as_base64("assets/user-avatar.png")
st.set_page_config(page_title="FiFi", page_icon=f"data:image/png;base64,{FIFI_AVATAR_B64}" if FIFI_AVATAR_B64 else "🤖", layout="wide", initial_sidebar_state="auto")

def get_or_create_eventloop():
    """Gets the active asyncio event loop or creates a new one."""
    try: return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# This defines the state of our graph. It's a list of messages.
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Load environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ASSISTANT_NAME = os.environ.get("PINECONE_ASSISTANT_NAME")

SECRETS_ARE_MISSING = not all([OPENAI_API_KEY, TAVILY_API_KEY, PINECONE_API_KEY, PINECONE_ASSISTANT_NAME])

if not SECRETS_ARE_MISSING:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.2)
    if 'thread_id' not in st.session_state: st.session_state.thread_id = f"fifi_streamlit_session_{uuid.uuid4()}"
    THREAD_ID = st.session_state.thread_id

# --- TOOLS: `get_context` is primary, `tavily_search_fallback` is secondary ---

@tool
def get_context(query: str, conversation_history: list = None) -> str:
    """
    Retrieves answers from the 1-2-Taste Pinecone Assistant. This is the primary tool and should always be tried first.
    It returns 'SUCCESS: [answer]' if successful, and 'FAILURE: [reason]' if it fails or cannot answer.
    """
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        assistant = pc.assistant.Assistant(assistant_name=PINECONE_ASSISTANT_NAME)
        messages = []
        if conversation_history:
            for msg in conversation_history:
                messages.append(Message(role=msg["role"], content=msg["content"]))
        messages.append(Message(role="user", content=query))
        
        response = assistant.chat(messages=messages)
        content = str(response.message.content) if hasattr(response, 'message') else str(response)
        
        failure_keywords = ["unable to answer", "don't have information", "cannot provide"]
        if not content or any(keyword in content.lower() for keyword in failure_keywords):
            print("Pinecone tool returned a non-committal answer.")
            return "FAILURE: Pinecone Assistant could not provide a specific answer."
        
        return f"SUCCESS: {content}"
    except Exception as e:
        print(f"--- PINECONE TOOL ERROR --- \n{traceback.format_exc()}\n--------------------")
        return "FAILURE: An exception occurred while contacting Pinecone."

@tool
def tavily_search_fallback(query: str) -> str:
    """
    Performs a web search using Tavily. This tool is a fallback and should only be used when `get_context` fails.
    """
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(query=query, search_depth="advanced", max_results=5, include_answer=True)
        disclaimer = "> I could not find a specific answer within the 1-2-Taste knowledge base. The following information is from a general web search.\n\n"
        formatted_output = disclaimer
        if response.get('answer'): formatted_output += f"Web Search Summary: {response['answer']}\n\n"
        if response.get('results'):
            formatted_output += "Sources:\n"
            for source in response['results']:
                formatted_output += f"- **{source.get('title', 'No Title')}**: {source.get('url', 'No URL')}\n"
        return formatted_output
    except Exception as e:
        return f"Error during web search: {e}"

# --- AGENT AND GRAPH DEFINITION: This class defines the agent's logic flow ---

class Agent:
    def __init__(self, tools, system=""):
        self.system = system
        self.tools = tools
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", ToolNode(tools))
        graph.add_node("router", self.router)

        graph.set_entry_point("llm")
        graph.add_edge("action", "router")
        graph.add_conditional_edges("llm", self.should_call_tool, {True: "action", False: END})
        graph.add_conditional_edges("router", self.should_fallback, {"fallback": "llm", "continue": END})
        
        self.graph = graph.compile(checkpointer=MemorySaver())

    # --- THIS IS THE FIX ---
    def should_call_tool(self, state: AgentState) -> bool:
        """Determines if the LLM decided to call a tool."""
        # This now explicitly returns True or False
        return bool(isinstance(state['messages'][-1], AIMessage) and state['messages'][-1].tool_calls)

    def router(self, state: AgentState) -> str:
        """After a tool call, this router decides if we need to fall back to Tavily or if we can continue."""
        last_message = state['messages'][-1]
        if not isinstance(last_message, ToolMessage):
            return "continue"
        
        if last_message.name == "get_context" and last_message.content.startswith("FAILURE:"):
            return "fallback"
        return "continue"

    def should_fallback(self, state: AgentState) -> str:
        """This is a simple wrapper for the router to be used in conditional edges."""
        return self.router(state)

    def call_llm(self, state: AgentState):
        """The core logic of the agent. It calls the LLM with the current state."""
        messages = list(state['messages'])
        last_message = messages[-1]

        if isinstance(last_message, ToolMessage) and last_message.name == "get_context" and last_message.content.startswith("FAILURE:"):
            user_query = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            messages.append(
                SystemMessage(content=f"The primary `get_context` tool failed. You MUST now use the `tavily_search_fallback` tool to answer the user's original query: '{user_query}'")
            )
        elif isinstance(last_message, ToolMessage) and last_message.content.startswith("SUCCESS:"):
             last_message.content = last_message.content.replace("SUCCESS:", "").strip()

        prompt = ChatPromptTemplate.from_messages([("system", self.system), MessagesPlaceholder(variable_name="messages")])
        chain = prompt | llm.bind_tools(self.tools)
        result = chain.invoke({"messages": messages})
        return {'messages': [result]}

@st.cache_resource(ttl=3600)
def get_agent_graph():
    """Initializes the agent and its graph."""
    system_prompt = """You are FiFi, an expert AI assistant for 1-2-Taste.
- Your first step is ALWAYS to use the `get_context` tool to answer the user's question. For follow-up questions, you should pass the `conversation_history` to the tool.
- If the `get_context` tool call fails, the system will provide you with a new instruction. Follow that instruction exactly.
- When you have the final answer, either from `get_context` or `tavily_search_fallback`, present it clearly to the user. Do not mention internal markers like 'FAILURE' or 'SUCCESS' in your final response.
"""
    agent = Agent(tools=[get_context, tavily_search_fallback], system=system_prompt)
    return agent.graph

# --- Agent execution logic ---
async def execute_agent_call_with_memory(user_query: str, graph):
    """Runs the agent graph with the user's query and existing session history."""
    try:
        config = {"configurable": {"thread_id": THREAD_ID}}
        messages = [HumanMessage(content=user_query)]
        final_state = await graph.ainvoke({"messages": messages}, config=config)
        assistant_reply = final_state["messages"][-1].content
        
        return assistant_reply if assistant_reply else "(No response was generated.)"
    except Exception as e:
        print(f"--- ERROR: Exception caught during execution! ---")
        traceback.print_exc()
        if "RateLimitError" in str(e) or "429" in str(e):
             return "**The system is currently experiencing high load. Please try again in a moment.**"
        return f"**An error occurred during processing.** Please check the logs."

# --- UI Section ---
def handle_new_query_submission(query_text: str):
    if not st.session_state.get('thinking_for_ui', False):
        st.session_state.active_question = query_text
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.session_state.query_to_process = query_text
        st.session_state.thinking_for_ui = True
        st.rerun()

st.markdown("""<style>.st-emotion-cache-1629p8f{border:1px solid #ffffff;border-radius:7px;bottom:5px;position:fixed;width:100%;max-width:736px;left:50%;transform:translateX(-50%);z-index:101;}.st-emotion-cache-1629p8f:focus-within{border-color:#e6007e;}[data-testid=stCaptionContainer] p{font-size:1.3em !important;}[data-testid=stVerticalBlock]{padding-bottom:40px;}[data-testid=stChatMessage]{margin-top:0.1rem !important;margin-bottom:0.1rem !important;}.stApp{overflow-y:auto !important;}.st-scroll-to-bottom{display:none !important;}.st-emotion-cache-1fplawd{display:none !important;}</style>""", unsafe_allow_html=True)
st.markdown("<h1 style='font-size:24px;'>FiFi, AI sourcing assistant</h1>", unsafe_allow_html=True)
st.caption("Hello, I am FiFi, your AI-powered assistant, designed to support you across the sourcing and product development journey. Find the right ingredients, explore recipe ideas, technical data, and more.")

if SECRETS_ARE_MISSING:
    st.error("Secrets missing. Please configure necessary environment variables.")
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
preview_questions = ["Suggest some natural strawberry flavours for beverage", "Latest trends in plant-based proteins for 2025?", "Suggest me some vanilla flavours for ice-cream"]
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

user_prompt = st.chat_input("Ask me for ingredients, recipes, or product development—in any language.", key="main_chat_input", disabled=st.session_state.get('thinking_for_ui', False) or not st.session_state.get("components_loaded", False))
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
