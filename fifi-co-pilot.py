import streamlit as st
import datetime
import asyncio
import tiktoken
import os

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage, BaseMessage

# --- Constants for History Summarization ---
# Lowered for testing. You can set this back to 4000 for production.
SUMMARIZE_THRESHOLD_TOKENS = 500

# Number of recent user/assistant messages to keep raw (unsummarized).
MESSAGES_TO_KEEP_AFTER_SUMMARIZATION = 12

TOKEN_MODEL_ENCODING = "cl100k_base"

# --- Load environment variables from secrets ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MCP_PINECONE_URL = os.environ.get("MCP_PINECONE_URL")
MCP_PINECONE_API_KEY = os.environ.get("MCP_PINECONE_API_KEY")
MCP_PIPEDREAM_URL = os.environ.get("MCP_PIPEDREAM_URL")

if not all([OPENAI_API_KEY, MCP_PINECONE_URL, MCP_PINECONE_API_KEY, MCP_PIPEDREAM_URL]):
    st.error("One or more secrets are missing. Please configure them in Streamlit secrets.")
    st.stop()

llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.2)
THREAD_ID = "fifi_streamlit_session"

# --- System Prompt Definition ---
# This is defined outside run_async_initialization to be accessible for filtering
# and also passed directly when constructing the full message list for invocation.
def get_system_prompt_content_string(agent_components_for_prompt=None):
    # Default values for agent_components_for_prompt if not provided (e.g., initial call)
    if agent_components_for_prompt is None:
        agent_components_for_prompt = {
            'pinecone_tool_name': "functions.get_context",
            'all_tool_details_for_prompt': {"functions.get_context": "Retrieves relevant document snippets from the assistant knowledge base."}
        }

    pinecone_tool = agent_components_for_prompt['pinecone_tool_name']
    all_tool_details = agent_components_for_prompt['all_tool_details_for_prompt']
    prompt = f"""You are FiFi, an expert AI assistant for 1-2-Taste. Your **sole purpose** is to assist users with inquiries related to 1-2-Taste's products, the food and beverage ingredients industry, food science topics relevant to 1-2-Taste's offerings, B2B inquiries, recipe development support using 1-2-Taste ingredients, and specific e-commerce functions related to 1-2-Taste's WooCommerce platform.

**Core Mission:**
*   Provide accurate, **cited** information about 1-2-Taste's offerings using your product information capabilities.
*   Assist with relevant e-commerce tasks if explicitly requested by the user in a way that matches your e-commerce functions.
*   Politely decline to answer questions that are outside of your designated scope.

**Tool Usage Priority and Guidelines (Internal Instructions for You, the LLM):**

1.  **Primary Product & Industry Information Tool (Internally known as `{pinecone_tool}`):**
    *   For ANY query that could relate to 1-2-Taste product details, ingredients, flavors, availability, specifications, recipes, applications, food industry trends relevant to 1-2-Taste, or any information found within the 1-2-Taste catalog or relevant to its business, you **MUST ALWAYS PRIORITIZE** using this specialized tool (internally, its name is `{pinecone_tool}`). Its description is: "{all_tool_details.get(pinecone_tool, 'Retrieves relevant document snippets from the assistant knowledge base.')}" This is your main and most reliable knowledge source for product-related questions.
    *   If a query is ambiguous but might be product-related (e.g., "tell me about vanilla"), assume it is about 1-2-Taste's context and use this tool first.
    *   To manage token usage and control the amount of context returned, you MUST include the `top_k` and `snippet_size` parameters in your arguments. Use the following values:
        *   `top_k`: 16
        *   `snippet_size`: 2048
    *   For example, a correct tool call would look like: `get_context(query='some query about ingredients', top_k=10, snippet_size=1024)`

2.  **E-commerce and Order Management Tools (Internally, these are your WooCommerce tools like `functions.WOOCOMMERCE-GET-ORDER`, etc.):**
    *   You should **ONLY** use one of these e-commerce tools if the user's query EXPLICITLY mentions "WooCommerce", "orders", "my order", "customer accounts", "shipping status", "store management", "cart issues", or other clearly WooCommerce-specific administrative or e-commerce tasks relevant to 1-2-Taste that map to the specific functions of these tools.
    *   Do NOT use these e-commerce tools for general product information.

**Describing Your Capabilities to the User:**
*   **If a user asks what you can do or what tools you have, describe your functions in user-friendly terms. Do NOT reveal the internal or programmatic names of your tools (e.g., do not mention names like 'functions.get_context', 'functions.WOOCOMMERCE-...', or similar).**
*   Instead, explain your capabilities functionally. For example:
    *   "I can help you find detailed information about 1-2-Taste's products, ingredients, and flavors."
    *   "I can assist with inquiries about recipes and product applications using 1-2-Taste ingredients."
    *   "If you have questions about your orders on the 1-2-Taste platform or need help with e-commerce functions, I can try to assist with that."
    *   "My main role is to provide information and support related to 1-2-Taste and the food ingredients industry."

**Handling Out-of-Scope Queries:**
*   If a user's query is clearly unrelated to 1-2-Taste, its products, the food and beverage ingredients industry, relevant food science, B2B interactions in this context, or e-commerce tasks for 1-2-Taste (e.g., asking about celebrity gossip, historical events, general programming help, sports, etc.), you **MUST POLITELY DECLINE** to answer.
*   Example decline phrases:
    *   "My apologies, but my expertise is focused on 1-2-Taste and the food ingredients industry. I can't help with that topic."
    *   "I'm designed to assist with 1-2-Taste's products and related industry topics. Could we focus on that, or is there something else related to food ingredients I can help you with?"
    *   "That question is outside my scope of knowledge as an assistant for 1-2-Taste."

**General Knowledge (Strictly Limited Use - Internal Instruction for You, the LLM):**
    *   You should **AVOID** using your general knowledge.
    *   If, after attempting to use your primary product information tool (internally `{pinecone_tool}`) for a query that *seems potentially relevant* to 1-2-Taste or its industry, the tool yields no useful information or indicates the specific query is out of its own scope (but the topic is still vaguely related to food/ingredients), you *may* provide a very brief, general answer *if you are highly confident*.
    *   However, if there's any doubt, or if the query leans towards being off-topic even after failing with your product tool, it is better to politely decline as per the "Handling Out-of-Scope Queries" section.
    *   **Do NOT use general knowledge for topics clearly unrelated to 1-2-Taste's domain.**

**Response Guidelines & Output Format:**
*   **Strict Inclusion Policy:** You **MUST ONLY** include products in your answer that have a verifiable `productURL` or `source_url` in the tool's output. If a product appears in the tool's context but lacks a URL, you **MUST completely ignore that product** and it must not be mentioned in your response at all.
*   **Mandatory Citations for Product Information:** For every product you include in your answer (which, per the rule above, will always have a URL), you **MUST ALWAYS** cite the `productURL` or `source_url` provided in the tool's output.
*   Format citations clearly, for example: "You can find more details here: [Source URL]" or append "[Source: URL]" after the relevant sentence.
*   If multiple products are mentioned, cite each one appropriately if possible.
*   If a product is discontinued according to your product information tool, inform the user and, if possible, suggest alternatives found via the same tool (citing them as well).
*   **Do not provide product prices.** Instead, thank the user for asking and direct them to the product page on the 1-2-Taste website or to contact sales-eu@12taste.com.
*   If a product is marked as (QUOTE ONLY) and price is missing, ask them to visit: https://www.12taste.com/request-quote/.
*   Provide comprehensive and detailed answers, explaining your reasoning and including as much relevant information as possible from the context provided by your tools.

Answer the user's last query based on these instructions and the conversation history."""
    return prompt

def count_tokens(messages: list, model_encoding: str = TOKEN_MODEL_ENCODING) -> int:
    if not messages: return 0
    try: encoding = tiktoken.get_encoding(model_encoding)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4 # For role, start/end tokens (approximate)
        if isinstance(message, BaseMessage):
            content = message.content
        elif isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = str(message) # Fallback for unexpected types

        if content is not None:
            try: num_tokens += len(encoding.encode(str(content)))
            except (TypeError, AttributeError): pass
    num_tokens += 2 # Every reply is primed with <|start|>assistant<|message|>
    return num_tokens

# --- REVISED Function to summarize history if needed ---
async def summarize_history_if_needed(
    memory_instance: MemorySaver,
    thread_config: dict,
    main_system_prompt_content_str: str, # Pass the exact system prompt string for filtering
    summarize_threshold_tokens: int,
    keep_last_n_interactions: int,
    llm_for_summary: ChatOpenAI
):
    checkpoint = memory_instance.get(thread_config)
    current_stored_messages = checkpoint.get("messages", []) if checkpoint else []
    
    # 1. Filter out all occurrences of the MAIN system prompt from the stored history.
    #    This is the crucial step to prevent duplication.
    cleaned_messages = [
        m for m in current_stored_messages 
        if not (isinstance(m, SystemMessage) and m.content == main_system_prompt_content_str)
    ]
    
    # The 'cleaned_messages' now contain only conversational turns and explicit summary messages.
    conversational_messages_only = cleaned_messages

    current_token_count = count_tokens(conversational_messages_only)

    st.sidebar.markdown(f"**Conv. Tokens (w/ summaries):** `{current_token_count}` / `{summarize_threshold_tokens}`")
    st.sidebar.markdown(f"**Total Stored Messages (raw):** `{len(current_stored_messages)}`")
    st.sidebar.markdown(f"**Cleaned Conv. Messages:** `{len(conversational_messages_only)}`")

    if current_token_count > summarize_threshold_tokens:
        st.info(f"Summarization Triggered: Conversational history ({current_token_count} tokens) > threshold ({summarize_threshold_tokens}).")
        print(f"INFO: Summarization Triggered. History ({current_token_count}) > threshold ({summarize_threshold_tokens}).")

        if len(conversational_messages_only) <= keep_last_n_interactions:
            print("INFO: Not enough messages to summarize beyond the 'keep raw' count. Skipping summarization.")
            st.info(f"Skipping summarization: Not enough conversational messages ({len(conversational_messages_only)}) to summarize beyond the 'keep_last_n_interactions' ({keep_last_n_interactions}) count.")
            return False

        messages_to_summarize = conversational_messages_only[:-keep_last_n_interactions]
        messages_to_keep_raw = conversational_messages_only[-keep_last_n_interactions:]
        
        st.sidebar.markdown(f"**Messages to Summarize:** `{len(messages_to_summarize)}`")
        st.sidebar.markdown(f"**Messages to Keep Raw:** `{len(messages_to_keep_raw)}`")

        if messages_to_summarize:
            summarization_prompt_messages = [
                SystemMessage(content="Please summarize the following conversation history concisely..."),
                HumanMessage(content="\n".join([f"{m.type.capitalize()}: {m.content}" for m in messages_to_summarize]))
            ]
            
            try:
                summary_response = await llm_for_summary.ainvoke(summarization_prompt_messages)
                summary_content = summary_response.content
                st.info("Summary generated successfully and history updated.")
                print(f"DEBUG: Generated Summary: {summary_content[:150]}...")
                
                new_messages_for_checkpoint = [SystemMessage(content=f"Previous conversation summary: {summary_content}")] + messages_to_keep_raw
                
                if checkpoint is None:
                    checkpoint = {"messages": []}
                checkpoint["messages"] = new_messages_for_checkpoint
                memory_instance.put(thread_config, checkpoint)
                print(f"INFO: Memory checkpoint updated with summarized history. New stored tokens: {count_tokens(new_messages_for_checkpoint)}")
                return True

            except Exception as e:
                st.error(f"Failed to generate summary: {e}")
                print(f"ERROR: Failed to generate summary: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                return False
    return False

# --- Async handler for agent initialization ---
async def run_async_initialization():
    print("@@@ ASYNC run_async_initialization: Starting actual resource initialization...")
    client = MultiServerMCPClient({
        "pinecone": {"url": MCP_PINECONE_URL, "transport": "sse", "headers": {"Authorization": f"Bearer {MCP_PINECONE_API_KEY}"}},
        "pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}
    })
    tools = await client.get_tools()
    memory = MemorySaver()
    
    pinecone_tool_name = "functions.get_context"
    all_tool_details = {tool.name: tool.description for tool in tools}

    system_prompt_content_value = get_system_prompt_content_string({
        'pinecone_tool_name': pinecone_tool_name,
        'all_tool_details_for_prompt': all_tool_details
    })

    agent_executor = create_react_agent(llm, tools, checkpointer=memory)
    print("@@@ ASYNC run_async_initialization: Initialization complete.")
    
    return {
        "agent_executor": agent_executor,
        "memory_instance": memory,
        "llm_for_summary": llm,
        "main_system_prompt_content_str": system_prompt_content_value
    }

# --- Synchronous, cached function for Streamlit ---
@st.cache_resource(ttl=3600)
def get_agent_components():
    print("@@@ get_agent_components: Populating cache by running the async initialization...")
    return asyncio.run(run_async_initialization())

# --- Async handler for user queries ---
async def execute_agent_call_with_memory(user_query: str, agent_components: dict):
    assistant_reply = ""
    try:
        config = {"configurable": {"thread_id": THREAD_ID}}
        main_system_prompt_content_str = agent_components["main_system_prompt_content_str"]

        await summarize_history_if_needed(
            agent_components["memory_instance"], config,
            main_system_prompt_content_str,
            SUMMARIZE_THRESHOLD_TOKENS, MESSAGES_TO_KEEP_AFTER_SUMMARIZATION,
            agent_components["llm_for_summary"]
        )

        current_checkpoint = agent_components["memory_instance"].get(config)
        history_messages = current_checkpoint.get("messages", []) if current_checkpoint else []

        event_messages = [SystemMessage(content=main_system_prompt_content_str)] + history_messages + [HumanMessage(content=user_query)]

        event = {"messages": event_messages}
        result = await agent_components["agent_executor"].ainvoke(event, config=config)
        
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    assistant_reply = msg.content
                    break
            if not assistant_reply:
                assistant_reply = f"(Error: No AI message found in result for user query: '{user_query}')"
        else:
            assistant_reply = f"(Error: Unexpected agent response format: {type(result)} - {result})"
            st.error(f"Unexpected agent response: {result}")

    except Exception as e:
        import traceback
        st.error(f"Error during agent invocation: {e}\n{traceback.format_exc()}")
        assistant_reply = f"(Error: {e})"

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.session_state.thinking_for_ui = False
    st.rerun()

# --- Input Handling Function ---
def handle_new_query_submission(query_text: str):
    if not st.session_state.get('thinking_for_ui', False):
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.session_state.query_to_process = query_text
        st.session_state.thinking_for_ui = True
        st.rerun()

# --- Streamlit App Starts Here ---
st.title("FiFi Co-Pilot 🚀 (Auto-Summarizing Memory)")

if "messages" not in st.session_state: st.session_state.messages = []
if 'thinking_for_ui' not in st.session_state: st.session_state.thinking_for_ui = False
if 'query_to_process' not in st.session_state: st.session_state.query_to_process = None
if 'components_loaded' not in st.session_state: st.session_state.components_loaded = False

try:
    agent_components = get_agent_components()
    st.session_state.components_loaded = True
except Exception as e:
    st.error(f"Failed to initialize agent. Please refresh. Error: {e}")
    st.session_state.components_loaded = False
    st.stop()

# --- UI Rendering ---
st.sidebar.markdown("## Memory Debugger")
st.sidebar.markdown("---")
st.sidebar.markdown("## Quick Questions")
preview_questions = [
    "Help me with my recipe for a new juice drink",
    "Suggest me some strawberry flavours for beverage",
    "I need vanilla flavours for ice-cream"
]
for question in preview_questions:
    if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True):
        handle_new_query_submission(question)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    memory = agent_components.get("memory_instance")
    if memory:
        memory.put({"configurable": {"thread_id": THREAD_ID}}, {"messages": []})
    
    st.session_state.messages = []
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    print("@@@ Chat history cleared from UI and memory checkpoint.")
    st.rerun()

# --- Defensive access to st.session_state.messages ---
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(str(message.get("content", "")))

if st.session_state.get('thinking_for_ui', False):
    with st.chat_message("assistant"):
        st.markdown("⌛ FiFi is thinking...")

if st.session_state.get('query_to_process'):
    query_to_run = st.session_state.query_to_process
    st.session_state.query_to_process = None
    asyncio.run(execute_agent_call_with_memory(query_to_run, agent_components))

user_prompt = st.chat_input("Ask FiFi Co-Pilot...", key="main_chat_input",
                            disabled=st.session_state.get('thinking_for_ui', False) or \
                                     not st.session_state.get("components_loaded", False))
if user_prompt:
    handle_new_query_submission(user_prompt)
