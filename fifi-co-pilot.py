# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
import streamlit as st
import base64
from pathlib import Path
import datetime
import asyncio
import os
import traceback
import uuid
from typing import Annotated, List, Literal, Dict
from typing_extensions import TypedDict

# --- LangGraph and LangChain Imports ---
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langgraph.prebuilt import ToolNode
from langchain_core.agents import AgentAction
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from tavily import TavilyClient
from pinecone import Pinecone
# This is the correct import from your provided code for the Message object
from pinecone_plugins.assistant.models.chat import Message

# Helper function to load and Base64-encode images
@st.cache_data
def get_image_as_base64(file_path):
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
    try: return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ASSISTANT_NAME = os.environ.get("PINECONE_ASSISTANT_NAME")

SECRETS_ARE_MISSING = not all([OPENAI_API_KEY, TAVILY_API_KEY, PINECONE_API_KEY, PINECONE_ASSISTANT_NAME])

if not SECRETS_ARE_MISSING:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.2)
    if 'thread_id' not in st.session_state: st.session_state.thread_id = f"fifi_streamlit_session_{uuid.uuid4()}"
    THREAD_ID = st.session_state.thread_id

# --- Robust Information Retrieval Tool ---

DEFAULT_EXCLUDED_DOMAINS = [
    "ingredientsnetwork.com", "csmingredients.com", "batafood.com", "nccingredients.com", "prinovaglobal.com", "ingrizo.com",
    "solina.com", "opply.com", "brusco.co.uk", "lehmanningredients.co.uk", "i-ingredients.com", "fciltd.com", "lupafoods.com",
    "tradeingredients.com", "peterwhiting.co.uk", "globalgrains.co.uk", "tradeindia.com", "udaan.com", "ofbusiness.com",
    "indiamart.com", "symega.com", "meviveinternational.com", "amazon.com", "podfoods.co", "gocheetah.com", "foodmaven.com",
    "connect.kehe.com", "knowde.com", "ingredientsonline.com", "sourcegoodfood.com"
]

@tool
def get_information(query: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """
    Retrieves information to answer a user's question, supporting multi-turn conversations.
    It first queries the internal 1-2-Taste Pinecone Assistant.
    If the assistant cannot answer or fails, it automatically performs a general web search as a fallback.
    This is the primary and only tool that should be used to find information.
    """
    pinecone_response = ""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        assistant = pc.assistant.Assistant(assistant_name=PINECONE_ASSISTANT_NAME)
        messages = []
        if conversation_history:
            for msg in conversation_history:
                messages.append(Message(role=msg["role"], content=msg["content"]))
        messages.append(Message(role="user", content=query))
        response = assistant.chat(messages=messages)
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            pinecone_response = response.message.content
        elif hasattr(response, 'content'):
            pinecone_response = response.content
        else:
            pinecone_response = str(response)
    except Exception as e:
        print(f"--- PINECONE TOOL CRITICAL ERROR --- \n{traceback.format_exc()}\n---------------------------")
        pinecone_response = f"Error communicating with Pinecone Assistant: {e}"

    failure_keywords = ["unable to answer", "don't have information", "cannot provide", "Error:"]
    if pinecone_response and not any(keyword in pinecone_response for keyword in failure_keywords):
        print("Pinecone Assistant provided a valid response.")
        return pinecone_response
    
    print(f"Pinecone failed or could not answer. Response: '{pinecone_response}'. Proceeding to Tavily fallback.")
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(
            query=query, search_depth="advanced", max_results=5, include_answer=True, include_raw_content=False,
            exclude_domains=DEFAULT_EXCLUDED_DOMAINS
        )
        disclaimer = "> I could not find a specific answer within the 1-2-Taste knowledge base. The following information is from a general web search and may point to external sites not affiliated with 1-2-Taste.\n\n"
        formatted_output = disclaimer
        if response.get('answer'):
            formatted_output += f"Web Search Summary: {response['answer']}\n\n"
        if response.get('results'):
            formatted_output += "Sources:\n"
            for i, source in enumerate(response['results'], 1):
                formatted_output += f"{i}. **{source.get('title', 'No Title')}**: {source.get('url', 'No URL')}\n"
        return formatted_output if len(formatted_output) > len(disclaimer) else "No information found from internal sources or web search."
    except Exception as e:
        return f"An error occurred during the web search: {str(e)}"

# --- Simplified System Prompt ---
def get_system_prompt_content_string():
    prompt = f"""<instructions>
<system_role>
You are FiFi, a helpful and expert AI assistant for 1-2-Taste. Your role is to assist with product and service inquiries, flavours, industry trends, food science, and B2B support.
Your ONLY way to gather information is by using the `get_information` tool. You must call this tool for any user query that requires information.
</system_role>
<formatting_rules>
- After getting a response from the tool, present the information clearly to the user.
- If the tool provides sources, you MUST list them at the end of your response.
- **Product Rules:** Do not mention products without a URL. NEVER provide product prices; direct users to the product page or ask them to contact the Sales Team at: sales-eu@12taste.com.
</formatting_rules>
<final_instruction>
Adhering to your core mission, provide a helpful and context-aware response to the user's query by calling the `get_information` tool. Pass the user's query directly to the tool. For follow-up questions, also pass the `conversation_history`.
</final_instruction>
</instructions>"""
    return prompt

# --- UNIFIED GRAPH ARCHITECTURE ---
@st.cache_resource(ttl=3600)
def get_agent_graph():
    all_tools = [get_information]
    tool_node = ToolNode(all_tools)
    system_prompt_content = get_system_prompt_content_string()
    agent_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_content),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad", optional=True),
        ]
    )
    agent_runnable = create_tool_calling_agent(llm, all_tools, agent_prompt_template)

    def agent_node(state: AgentState):
        current_messages = state["messages"]
        last_human_idx = -1
        for i in range(len(current_messages) - 1, -1, -1):
            if isinstance(current_messages[i], HumanMessage): last_human_idx = i; break
        if last_human_idx == -1: raise ValueError("HumanMessage not found in state.")
        
        current_input = current_messages[last_human_idx].content
        chat_history_for_tool = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in current_messages[:last_human_idx]
        ]
        
        agent_scratchpad_messages = current_messages[last_human_idx + 1:]
        intermediate_steps = []
        i = 0
        while i < len(agent_scratchpad_messages):
            if isinstance(agent_scratchpad_messages[i], AIMessage) and agent_scratchpad_messages[i].tool_calls:
                ai_msg = agent_scratchpad_messages[i]
                if (i + 1) < len(agent_scratchpad_messages) and isinstance(agent_scratchpad_messages[i+1], ToolMessage):
                    tool_msg = agent_scratchpad_messages[i + 1]
                    if tool_msg.tool_call_id == ai_msg.tool_calls[0]['id']:
                        action = AgentAction(tool=ai_msg.tool_calls[0]['name'], tool_input=ai_msg.tool_calls[0]['args'], log="", tool_call_id=ai_msg.tool_calls[0]['id'])
                        intermediate_steps.append((action, tool_msg)); i += 2; continue
                i += 1
            else: i += 1

        chat_history_for_runnable = current_messages[:last_human_idx]
        if state.get("summary"):
            chat_history_for_runnable = [SystemMessage(content=f"Conversation summary:\n{state['summary']}")] + chat_history_for_runnable
        
        agent_output_raw = agent_runnable.invoke({
            "chat_history": chat_history_for_runnable, "input": current_input, "intermediate_steps": intermediate_steps
        })
        
        if isinstance(agent_output_raw, AgentAction) and agent_output_raw.tool == 'get_information':
            agent_output_raw.tool_input['conversation_history'] = chat_history_for_tool

        processed_messages = []
        output_items = agent_output_raw if isinstance(agent_output_raw, list) else [agent_output_raw]
        for item in output_items:
            if isinstance(item, AgentAction):
                processed_messages.append(AIMessage(content="", tool_calls=[{"name": item.tool, "args": item.tool_input, "id": item.tool_call_id or str(uuid.uuid4())}]))
            elif hasattr(item, 'return_values'):
                processed_messages.append(AIMessage(content=item.return_values.get('output', '')))
            elif isinstance(item, BaseMessage):
                processed_messages.append(item)
            else: raise ValueError(f"Unexpected agent output type: {type(item)}")
        return {"messages": processed_messages}

    def summarize_node(state: AgentState):
        messages_to_summarize = state["messages"][:-1]
        summarizer_memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000, return_messages=False)
        for msg in messages_to_summarize:
            if isinstance(msg, HumanMessage): summarizer_memory.chat_memory.add_user_message(msg.content)
            else: summarizer_memory.chat_memory.add_ai_message(msg.content)
        summary = summarizer_memory.load_memory_variables({})["history"]
        return {"summary": summary, "messages": [RemoveMessage(id=m.id) for m in messages_to_summarize]}

    def should_continue_agent_loop(state: AgentState) -> Literal["tools", END]:
        return "tools" if isinstance(state["messages"][-1], AIMessage) and state["messages"][-1].tool_calls else END

    def should_summarize(state: AgentState) -> Literal["summarize", "agent"]:
        return "summarize" if len(state["messages"]) > 6 else "agent"

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("summarize", summarize_node); graph_builder.add_node("agent", agent_node); graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges("__start__", should_summarize, {"summarize": "summarize", "agent": "agent"})
    graph_builder.add_edge("summarize", "agent")
    graph_builder.add_conditional_edges("agent", should_continue_agent_loop, {"tools": "tools", END: END})
    graph_builder.add_edge("tools", "agent")
    return graph_builder.compile(checkpointer=MemorySaver())

async def execute_agent_call_with_memory(user_query: str, graph):
    try:
        config = {"configurable": {"thread_id": THREAD_ID}}
        event = {"messages": [HumanMessage(content=user_query)]}
        final_state = await graph.ainvoke(event, config=config)
        assistant_reply = final_state["messages"][-1].content
        return assistant_reply if assistant_reply else "(No response was generated.)"
    except Exception as e:
        print(f"--- ERROR: Exception caught during execution! ---"); traceback.print_exc()
        return f"**An error occurred during processing:**\n\n```\n{traceback.format_exc()}\n```"

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
# --- FIX: Using the more stable, targeted reset logic ---
if st.sidebar.button("🧹 Reset chat session", use_container_width=True):
    st.session_state.messages = []
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    st.session_state.thread_id = f"fifi_streamlit_session_{uuid.uuid4()}"
    st.session_state.active_question = None
    st.rerun()

st.sidebar.markdown('By using this agent, you agree to our <a href="https://www.12taste.com/terms-conditions/" target="_blank">Terms of Service</a>.', unsafe_allow_html=True)

# --- FIX: Re-defining the avatar variables before the loop ---
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
