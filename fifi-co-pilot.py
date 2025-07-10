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
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langgraph.prebuilt import ToolNode

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from tavily import TavilyClient

# Helper function to load and Base64-encode images
@st.cache_data
def get_image_as_base64(file_path):
    """Loads an image file and returns it as a Base64 encoded string."""
    try:
        path = Path(file_path)
        with path.open("rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

# Load images once using the helper function
FIFI_AVATAR_B64 = get_image_as_base64("assets/fifi-avatar.png")
USER_AVATAR_B64 = get_image_as_base64("assets/user-avatar.png")

# Use the Base64 string for the page_icon to avoid MediaFileStorageError
st.set_page_config(
    page_title="FiFi",
    page_icon=f"data:image/png;base64,{FIFI_AVATAR_B64}" if FIFI_AVATAR_B64 else "🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# Robust asyncio helper function that works in any environment
def get_or_create_eventloop():
    """Gets the active asyncio event loop or creates a new one."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# --- NEW: Define the state for our LangGraph agent ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str

# --- Load environment variables from secrets ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MCP_PINECONE_URL = os.environ.get("MCP_PINECONE_URL")
MCP_PINECONE_API_KEY = os.environ.get("MCP_PINECONE_API_KEY")
MCP_PIPEDREAM_URL = os.environ.get("MCP_PIPEDREAM_URL")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

SECRETS_ARE_MISSING = not all([OPENAI_API_KEY, MCP_PINECONE_URL, MCP_PINECONE_API_KEY, MCP_PIPEDREAM_URL, TAVILY_API_KEY])

if not SECRETS_ARE_MISSING:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.2)
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = f"fifi_streamlit_session_{uuid.uuid4()}"
    THREAD_ID = st.session_state.thread_id

# --- Tavily Search Tool with Exclusions (Preserved from your code) ---
DEFAULT_EXCLUDED_DOMAINS = [
    "ingredientsnetwork.com", "csmingredients.com", "batafood.com", "nccingredients.com", "prinovaglobal.com",
    "ingrizo.com", "solina.com", "opply.com", "brusco.co.uk", "lehmanningredients.co.uk", "i-ingredients.com",
    "fciltd.com", "lupafoods.com", "tradeingredients.com", "peterwhiting.co.uk", "globalgrains.co.uk",
    "tradeindia.com", "udaan.com", "ofbusiness.com", "indiamart.com", "symega.com", "meviveinternational.com",
    "amazon.com", "podfoods.co", "gocheetah.com", "foodmaven.com", "connect.kehe.com", "knowde.com",
    "ingredientsonline.com", "sourcegoodfood.com"
]

@tool
def tavily_search_fallback(query: str) -> str:
    """
    Search the web using Tavily. Use this for queries about broader, public-knowledge topics.
    This tool automatically excludes a predefined list of competitor and marketplace domains.
    """
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(
            query=query, search_depth="advanced", max_results=5, include_answer=True, include_raw_content=False,
            exclude_domains=DEFAULT_EXCLUDED_DOMAINS
        )
        if response.get('answer'):
            result = f"Web Search Results:\n\nSummary: {response['answer']}\n\nSources:\n"
        else:
            result = "Web Search Results:\n\nSources:\n"
        for i, source in enumerate(response.get('results', []), 1):
            result += f"{i}. {source['title']}\n   URL: {source['url']}\n   Content: {source['content']}\n\n"
        return result
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# --- System Prompt Definition (Preserved from your code) ---
def get_system_prompt_content_string():
    pinecone_tool = "functions.get_context"
    prompt = f"""<instructions>
<system_role>
You are FiFi, a helpful and expert AI assistant for 1-2-Taste. Your primary goal is to be helpful within your designated scope. Your role is to assist with product and service inquiries, flavours, industry trends, food science, and B2B support. Politely decline out-of-scope questions. You must follow the tool protocol exactly as written to gather information.
</system_role>

<core_mission_and_scope>
Your mission is to provide information and support on 1-2-Taste products, the food and beverage industry, food science, and related B2B support. Use the conversation history to understand the user's intent, especially for follow-up questions.
</core_mission_and_scope>

<tool_protocol>
Your process for gathering information is a mandatory, sequential procedure. Do not deviate.

1.  **Step 1: Primary Tool Execution.**
    *   For any user query, your first and only initial action is to call the `{pinecone_tool}`.
    *   **Parameters:** Unless specified by a different rule (like the Anti-Repetition Rule), you MUST use `top_k=5` and `snippet_size=1024`.

2.  **Step 2: Mandatory Result Analysis.**
    *   After the primary tool returns a result, you MUST analyze it against the failure conditions below.

3.  **Step 3: Conditional Fallback Execution.**
    *   **If** the primary tool fails (because the result is empty, irrelevant, or lacks a `sourceURL`/`productURL`), then your next and only action **MUST** be to call the `tavily_search_fallback` tool with the original user query.
    *   Do not stop or apologize after a primary tool failure. The fallback call is a required part of the procedure.

4.  **Step 4: Final Answer Formulation.**
    *   Formulate your answer based on the data from the one successful tool call (either the primary or the fallback).
    *   **Disclaimer Rule:** If your answer is based on results from `tavily_search_fallback`, you **MUST** begin your response with this exact disclaimer, enclosed in a markdown quote block:
        > I could not find specific results within the 1-2-Taste EU product database. The following information is from a general web search and may point to external sites not affiliated with 1-2-Taste.
    *   If both tools fail, only then should you state that you could not find the information.
</tool_protocol>

<formatting_rules>
- **Citations are Mandatory:** Always cite the URL from the tool you used. When using tavily_search_fallback, you MUST include every source URL provided in the search results.
- **Source Format:** Present sources as a numbered list with both title and URL for each result.
- **Complete Attribution - CRITICAL RULE:** You MUST display ALL sources returned by the tool. If the tool provides 5 sources, your response MUST reference all 5 sources. If the tool provides 3 sources, show all 3. NEVER omit any sources from your response. This is a mandatory requirement.
- **Source Display Requirements:** 
  * List every single source with its title and URL
  * Use the exact format: "1. **[Title]**: [URL]"
  * Do not summarize or condense the source list
  * Include all sources even if they seem similar or redundant
- **Product Rules:** Do not mention products without a URL. NEVER provide product prices; direct users to the product page or ask to contact Sales Team at: sales-eu@12taste.com
- **Anti-Repetition Rule:**
    *   When a user asks for "more," "other," or "different" suggestions on a topic you have already discussed, you MUST alter your search strategy.
    *   **Action:** Your next call to `{pinecone_tool}` for this topic MUST use a larger `top_k` parameter, for example, `top_k=10`. This is to ensure you get a wider selection of potential results.
    *   **Filtering:** Before presenting the new results, you MUST review the conversation history and filter out any products or `sourceURL`s that you have already suggested.
    *   **Response:** If you have new, unique products after filtering, present them. If the larger search returns only products you have already mentioned, you MUST inform the user that you have no new suggestions on this topic. Do not list the old products again.
</formatting_rules>

<final_instruction>
Adhering to your core mission and the mandatory tool protocol, provide a helpful and context-aware response to the user's query.
</final_instruction>
</instructions>"""
    return prompt

# --- NEW: Graph construction with integrated memory management ---
@st.cache_resource(ttl=3600)
def get_agent_graph():
    """
    Constructs the LangGraph agent graph with integrated memory summarization.
    """
    async def run_async_tool_initialization():
        client = MultiServerMCPClient({
            "pinecone": {"url": MCP_PINECONE_URL, "transport": "sse", "headers": {"Authorization": f"Bearer {MCP_PINECONE_API_KEY}"}},
            "pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}
        })
        return await client.get_tools()

    mcp_tools = get_or_create_eventloop().run_until_complete(run_async_tool_initialization())
    all_tools = list(mcp_tools) + [tavily_search_fallback]
    tool_node = ToolNode(all_tools)

    system_prompt = get_system_prompt_content_string()
    agent_prompt = hub.pull("hwchase17/xml-agent-convo").partial(system_message=system_prompt)
    agent_runnable = create_tool_calling_agent(llm, all_tools, agent_prompt)

    # --- MODIFIED: agent_node with the critical fix ---
    def agent_node(state: AgentState):
        """The 'think' node. Calls the LLM to decide the next action."""
        chat_history = state["messages"][:-1]
        input_text = state["messages"][-1].content
        
        if state.get("summary"):
            summary_message = SystemMessage(content=f"This is a summary of the preceding conversation:\n{state['summary']}")
            chat_history = [summary_message] + chat_history
        
        # --- THE FIX IS HERE ---
        # The agent runnable requires the 'intermediate_steps' key, even if it's empty.
        result = agent_runnable.invoke({
            "chat_history": chat_history,
            "input": input_text,
            "intermediate_steps": []
        })
        return {"messages": [result]}

    def summarize_node(state: AgentState):
        """The 'summarize' node. Creates a summary and prunes old messages."""
        print("@@@ MEMORY MGMT: Summarizing conversation...")
        messages_to_summarize = state["messages"][:-1]
        
        summarizer_memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500, return_messages=False)
        for msg in messages_to_summarize:
            if isinstance(msg, HumanMessage):
                summarizer_memory.chat_memory.add_user_message(msg.content)
            else:
                summarizer_memory.chat_memory.add_ai_message(msg.content)
        summary = summarizer_memory.load_memory_variables({})["history"]
        messages_to_remove = [RemoveMessage(id=m.id) for m in messages_to_summarize]
        st.info("Conversation history is long. Summarizing older messages...")
        return {"summary": summary, "messages": messages_to_remove}

    # Define Conditional Edges
    def should_continue_agent_loop(state: AgentState) -> Literal["tools", END]:
        if isinstance(state["messages"][-1], AIMessage) and not state["messages"][-1].tool_calls:
            return END
        return "tools"

    def should_summarize(state: AgentState) -> Literal["summarize", "agent"]:
        if len(state["messages"]) > 6:
            return "summarize"
        return "agent"

    # Build the Graph
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("summarize", summarize_node)
    graph_builder.add_conditional_edges("__start__", should_summarize, {"summarize": "summarize", "agent": "agent"})
    graph_builder.add_edge("summarize", "agent")
    graph_builder.add_conditional_edges("agent", should_continue_agent_loop, {"tools": "tools", "__end__": END})
    graph_builder.add_edge("tools", "agent")
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph

# --- Agent execution logic ---
async def execute_agent_call_with_memory(user_query: str, graph):
    """
    Runs the agent graph and returns the assistant's reply OR the full error traceback.
    """
    try:
        config = {"configurable": {"thread_id": THREAD_ID}}
        event = {"messages": [HumanMessage(content=user_query)]}
        final_state = await graph.ainvoke(event, config=config)
        assistant_reply = ""
        if final_state and "messages" in final_state and final_state["messages"]:
            last_message = final_state["messages"][-1]
            if isinstance(last_message, AIMessage):
                assistant_reply = last_message.content
        return assistant_reply if assistant_reply else "(An error occurred: No AI response was generated.)"
    except Exception as e:
        # This will now capture the full error and format it as a string
        # to be displayed directly in the chat window.
        print(f"--- ERROR: Exception caught during graph invocation! ---")
        traceback.print_exc()
        error_message = f"**An error occurred during processing:**\n\n```\n{traceback.format_exc()}\n```"
        return error_message

# --- Input Handling Function ---
def handle_new_query_submission(query_text: str):
    if not st.session_state.get('thinking_for_ui', False):
        st.session_state.active_question = query_text
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.session_state.query_to_process = query_text
        st.session_state.thinking_for_ui = True
        st.rerun()

# --- Streamlit App UI and Main Logic ---
st.markdown("""
<style>
    .st-emotion-cache-1629p8f { border: 1px solid #ffffff; border-radius: 7px; bottom: 5px; position: fixed; width: 100%; max-width: 736px; left: 50%; transform: translateX(-50%); z-index: 101; }
    .st-emotion-cache-1629p8f:focus-within { border-color: #e6007e; }
    [data-testid="stCaptionContainer"] p { font-size: 1.3em !important; }
    [data-testid="stVerticalBlock"] { padding-bottom: 40px; }
    [data-testid="stChatMessage"] { margin-top: 0.1rem !important; margin-bottom: 0.1rem !important; }
    .stApp { overflow-y: auto !important; }
    .st-scroll-to-bottom { display: none !important; }
    .st-emotion-cache-1fplawd { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 24px;'>FiFi, AI sourcing assistant</h1>", unsafe_allow_html=True)
st.caption("Hello, I am FiFi, your AI-powered assistant, designed to support you across the sourcing and product development journey. Find the right ingredients, explore recipe ideas, technical data, and more.")

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
preview_questions = [
    "Suggest some natural strawberry flavours for beverage",
    "Latest trends in plant-based proteins for 2025?",
    "Suggest me some vanilla flavours for ice-cream"
]
for question in preview_questions:
    button_type = "primary" if st.session_state.active_question == question else "secondary"
    if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True, type=button_type):
        handle_new_query_submission(question)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Reset chat session", use_container_width=True):
    st.session_state.messages = []
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    st.session_state.thread_id = f"fifi_streamlit_session_{uuid.uuid4()}"
    st.session_state.active_question = None
    print(f"@@@ New chat session started. Thread ID: {st.session_state.thread_id}")
    st.rerun()

st.sidebar.markdown('By using this agent, you agree to our <a href="https://www.12taste.com/terms-conditions/" target="_blank">Terms of Service</a>.', unsafe_allow_html=True)

fifi_avatar_icon = f"data:image/png;base64,{FIFI_AVATAR_B64}" if FIFI_AVATAR_B64 else "🤖"
user_avatar_icon = f"data:image/png;base64,{USER_AVATAR_B64}" if USER_AVATAR_B64 else "🧑‍💻"
for message in st.session_state.get("messages", []):
    avatar_icon = fifi_avatar_icon if message["role"] == "assistant" else user_avatar_icon
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message.get("content", ""))

if st.session_state.get('thinking_for_ui', False):
    with st.chat_message("assistant", avatar=fifi_avatar_icon):
        st.markdown("⌛ FiFi is thinking...")

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
