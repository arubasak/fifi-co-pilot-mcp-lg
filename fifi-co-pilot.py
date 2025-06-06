import streamlit as st
import datetime
import asyncio
import tiktoken

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

# --- Constants for History Pruning ---
MAX_HISTORY_TOKENS = 90000
MESSAGES_TO_KEEP_AFTER_PRUNING = 6
TOKEN_MODEL_ENCODING = "cl100k_base"

# --- Load environment variables from secrets ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
MCP_PINECONE_URL = st.secrets.get("MCP_PINECONE_URL")
MCP_PINECONE_API_KEY = st.secrets.get("MCP_PINECONE_API_KEY")
MCP_PIPEDREAM_URL = st.secrets.get("MCP_PIPEDREAM_URL")

if not all([OPENAI_API_KEY, MCP_PINECONE_URL, MCP_PINECONE_API_KEY, MCP_PIPEDREAM_URL]):
    st.error("One or more secrets are missing. Please configure them in Streamlit secrets.")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.2)
THREAD_ID = "fifi_streamlit_session"

def count_tokens(messages: list, model_encoding: str = TOKEN_MODEL_ENCODING) -> int:
    if not messages: return 0
    try: encoding = tiktoken.get_encoding(model_encoding)
    except Exception:
        print(f"Warning: Encoding {model_encoding} not found. Using 'cl100k_base'.")
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            if value is not None:
                try: num_tokens += len(encoding.encode(str(value)))
                except TypeError: print(f"Warning: Could not encode value of type {type(value)}.")
    num_tokens += 2
    return num_tokens

def prune_history_if_needed(
    memory_instance: MemorySaver, thread_config: dict, current_system_prompt_content: str, 
    max_tokens: int, keep_last_n_interactions: int 
):
    checkpoint_value = memory_instance.get(thread_config)
    if not checkpoint_value or "messages" not in checkpoint_value or \
       not isinstance(checkpoint_value.get("messages"), list):
        return False 
    current_messages_in_history = checkpoint_value["messages"]
    if not current_messages_in_history: return False
    current_token_count = count_tokens(current_messages_in_history)
    if current_token_count > max_tokens:
        print(f"INFO (prune): History token count ({current_token_count}) > max ({max_tokens}). Pruning...")
        user_assistant_messages = [m for m in current_messages_in_history if m.get("role") != "system"]
        pruned_user_assistant_messages = user_assistant_messages[-keep_last_n_interactions:]
        new_history_messages = [{"role": "system", "content": current_system_prompt_content}]
        new_history_messages.extend(pruned_user_assistant_messages)
        new_checkpoint_value_to_put = {"messages": new_history_messages}
        memory_instance.put(thread_config, new_checkpoint_value_to_put)
        pruned_token_count = count_tokens(new_history_messages)
        print(f"INFO (prune): History pruned. New token count: {pruned_token_count}.")
        return True
    return False

@st.cache_resource # This decorator handles the "run once and cache result"
async def get_agent_with_memory_tuple_cached(): 
    print("INVOKING ASYNC @st.cache_resource: Initializing agent with memory and fetching tools...")
    # This function will be awaited by Streamlit's caching mechanism ONCE.
    # Its RESULT (the tuple) will be cached.
    client = MultiServerMCPClient(
        {
            "pinecone": {"url": MCP_PINECONE_URL, "transport": "sse", "headers": {"Authorization": f"Bearer {MCP_PINECONE_API_KEY}"}},
            "pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}
        }
    )
    tools = await client.get_tools()
    
    pinecone_tool_name = "functions.get_context"
    woocommerce_tool_names = []
    all_tool_details = {}

    for tool in tools:
        all_tool_details[tool.name] = tool.description
        if tool.name == "functions.get_context": pinecone_tool_name = tool.name
        elif "woocommerce" in tool.name.lower(): woocommerce_tool_names.append(tool.name)
    
    # Set these directly into session_state from the cached function ONCE it runs.
    st.session_state.pinecone_tool_name = pinecone_tool_name
    st.session_state.woocommerce_tool_names = woocommerce_tool_names
    st.session_state.all_tool_details_for_prompt = all_tool_details

    print(f"Confirmed Pinecone tool (cached exec): {st.session_state.pinecone_tool_name}")
    if not st.session_state.woocommerce_tool_names: print("Warning (cached exec): No WooCommerce tools identified.")
    else: print(f"Identified WooCommerce tools (cached exec): {st.session_state.woocommerce_tool_names}")
    
    memory = MemorySaver()
    agent_executor = create_react_agent(llm, tools, checkpointer=memory)
    print("Agent with memory initialized successfully (cached exec).")
    return agent_executor, memory

# Initialize session states with defaults first
if "messages" not in st.session_state: st.session_state.messages = []
if 'thinking_for_ui' not in st.session_state: st.session_state.thinking_for_ui = False
if 'query_to_process' not in st.session_state: st.session_state.query_to_process = None
if 'pinecone_tool_name' not in st.session_state: st.session_state.pinecone_tool_name = "functions.get_context"
if 'woocommerce_tool_names' not in st.session_state: st.session_state.woocommerce_tool_names = []
if 'all_tool_details_for_prompt' not in st.session_state: st.session_state.all_tool_details_for_prompt = {}
if 'last_known_history_token_count' not in st.session_state: st.session_state.last_known_history_token_count = 0
# This session state will hold the *actual result tuple* after the cached async function has been run by Streamlit.
if 'app_agent_tuple' not in st.session_state: 
    st.session_state.app_agent_tuple = None


def get_system_prompt():
    pinecone_tool = st.session_state.get('pinecone_tool_name', "functions.get_context") 
    prompt = f"""You are FiFi, an expert AI assistant for 1-2-Taste. Your **sole purpose** is to assist users with inquiries related to 1-2-Taste's products, the food and beverage ingredients industry, food science topics relevant to 1-2-Taste's offerings, B2B inquiries, recipe development support using 1-2-Taste ingredients, and specific e-commerce functions related to 1-2-Taste's WooCommerce platform.

**Core Mission:**
*   Provide accurate information about 1-2-Taste's offerings using your product information capabilities.
*   Assist with relevant e-commerce tasks if explicitly requested by the user in a way that matches your e-commerce functions.
*   Politely decline to answer questions that are outside of your designated scope.

**Tool Usage Priority and Guidelines (Internal Instructions for You, the LLM):**

1.  **Primary Product & Industry Information Tool (Internally known as `{pinecone_tool}`):**
    *   For ANY query that could relate to 1-2-Taste product details, ingredients, flavors, availability, specifications, recipes, applications, food industry trends relevant to 1-2-Taste, or any information found within the 1-2-Taste catalog or relevant to its business, you **MUST ALWAYS PRIORITIZE** using this specialized tool (internally, its name is `{pinecone_tool}`). Its description is: "{st.session_state.all_tool_details_for_prompt.get(pinecone_tool, 'Retrieves relevant document snippets from the assistant knowledge base.')}" This is your main and most reliable knowledge source for product-related questions.
    *   If a query is ambiguous but might be product-related (e.g., "tell me about vanilla"), assume it is about 1-2-Taste's context and use this tool first.

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

**Response Guidelines:**
*   Always cite your sources when providing product information obtained from your product information tool (internally `{pinecone_tool}`), if citation information is available from the tool.
*   If a product is discontinued according to your product information tool, inform the user and, if possible, suggest alternatives found via the same tool.
*   **Do not provide product prices.** Instead, thank the user for asking and direct them to the product page on the 1-2-Taste website or to contact sales-eu@12taste.com.
*   If a product is marked as (QUOTE ONLY) and price is missing, ask them to visit: https://www.12taste.com/request-quote/.
*   Keep answers concise and to the point.

Answer the user's last query based on these instructions and the conversation history.
"""
    return prompt

async def execute_agent_call_with_memory(user_query: str):
    assistant_reply = ""
    try:
        # Directly await the cached async function. Streamlit's cache mechanism
        # should return the already computed result if available, or run and cache it.
        agent_executor, memory_instance = await get_agent_with_memory_tuple_cached()
        # Store/update it in session_state if fetched, so UI token count can use it
        st.session_state.app_agent_tuple = (agent_executor, memory_instance)


        if agent_executor is None or memory_instance is None: # Should not happen if cache works
            st.error("Agent or Memory could not be initialized after caching.")
            assistant_reply = "(Error: Agent/Memory not initialized)"
        else:
            config = {"configurable": {"thread_id": THREAD_ID}}
            system_prompt_content = get_system_prompt()
            
            was_pruned = prune_history_if_needed(
                memory_instance, config, system_prompt_content,
                MAX_HISTORY_TOKENS, MESSAGES_TO_KEEP_AFTER_PRUNING
            )
            if was_pruned:
                checkpoint_after_prune = memory_instance.get(config)
                st.session_state.last_known_history_token_count = count_tokens(
                    checkpoint_after_prune.get("messages", []) if checkpoint_after_prune else []
                )

            current_turn_messages = [
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": user_query}
            ]
            event = {"messages": current_turn_messages}
            result = await agent_executor.ainvoke(event, config=config)
            
            checkpoint_after_ainvoke = memory_instance.get(config) 
            # print(f"DEBUG (execute): Checkpoint value AFTER AINVOKE: {checkpoint_after_ainvoke}")


            if isinstance(result, dict) and "messages" in result and result["messages"]:
                assistant_reply = result["messages"][-1].content
            else:
                assistant_reply = f"(Error: Unexpected agent response format: {type(result)} - {result})"
                st.error(f"Unexpected agent response: {result}")

    except Exception as e:
        import traceback
        st.error(f"Error during agent invocation: {e}\n{traceback.format_exc()}")
        assistant_reply = f"(Error: {e})"

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.session_state.thinking_for_ui = False
    
    if st.session_state.app_agent_tuple: # Check if it was set
        _, memory_instance_for_ui = st.session_state.app_agent_tuple
        if memory_instance_for_ui:
            config_for_ui = {"configurable": {"thread_id": THREAD_ID}}
            checkpoint_val = memory_instance_for_ui.get(config_for_ui)
            if checkpoint_val and "messages" in checkpoint_val:
                st.session_state.last_known_history_token_count = count_tokens(checkpoint_val["messages"])
            else:
                st.session_state.last_known_history_token_count = 0
    st.rerun()

def handle_new_query_submission(query_text: str):
    if not st.session_state.get('thinking_for_ui', False):
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.session_state.query_to_process = query_text
        st.session_state.thinking_for_ui = True
        st.rerun()

# --- Main App Logic ---
st.title("FiFi Co-Pilot 🚀 (LangGraph MCP Agent with Auto-Pruning Memory)")

# Top-level initialization:
# We need to ensure the async cached function runs once to populate session_state for tool names
# and to make the agent available.
# The challenge is calling an async function from the top-level sync script.
# The most robust way with st.cache_resource is to simply call it when it's first needed
# from an async context, and Streamlit handles the caching.
# For ensuring it's ready for the first UI paint (for token counter & system prompt access),
# we can try an initial explicit call if not already populated.

if st.session_state.app_agent_tuple is None:
    print("MAIN SCRIPT: app_agent_tuple is None. Attempting to initialize via asyncio.run().")
    try:
        # This will block and run the async function, caching its result.
        # @st.cache_resource should ensure the inner async work only happens once.
        st.session_state.app_agent_tuple = asyncio.run(get_agent_with_memory_tuple_cached())
        print("MAIN SCRIPT: app_agent_tuple initialized via asyncio.run().")
        
        # Initialize token count for UI after agent is ready
        if st.session_state.app_agent_tuple:
            _, memory_instance_for_init_ui = st.session_state.app_agent_tuple
            if memory_instance_for_init_ui:
                config_for_init_ui = {"configurable": {"thread_id": THREAD_ID}}
                checkpoint_val_init = memory_instance_for_init_ui.get(config_for_init_ui)
                if checkpoint_val_init and "messages" in checkpoint_val_init:
                    st.session_state.last_known_history_token_count = count_tokens(checkpoint_val_init["messages"])
                else:
                    st.session_state.last_known_history_token_count = 0
                print(f"DEBUG (UI init after run): UI Token count initialized: {st.session_state.last_known_history_token_count}")

    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e) or \
           "cannot enter context with task" in str(e): # Another variant
            # This happens if Streamlit Cloud (or local) is already managing an outer event loop.
            # In this scenario, we can't use asyncio.run().
            # We'll have to rely on the first call from execute_agent_call_with_memory (which is async)
            # to populate st.session_state.app_agent_tuple.
            # The initial token count might show "N/A" or 0 until the first interaction.
            print(f"INFO (UI init): Could not run initial agent setup using asyncio.run() due to existing event loop: {e}")
            st.session_state.last_known_history_token_count = "N/A (Initializing...)"
            # We can't set app_agent_tuple here if asyncio.run fails.
            # It will be set upon the first call to execute_agent_call_with_memory.
        else:
            st.error(f"Critical error during initial agent setup: {e}")
            st.stop() # Stop the app if a critical non-async related error occurs


# Sidebar
st.sidebar.markdown("## Quick Questions")
preview_questions = ["Help me with my recipe for a new juice drink", "Suggest me some strawberry flavours for beverage", "I need vanilla flavours for ice-cream"]
for question in preview_questions:
    if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True):
        handle_new_query_submission(question)

st.sidebar.markdown("---")
debug_token_val_sidebar = st.session_state.get('last_known_history_token_count', "N/A")
# st.sidebar.write(f"DEBUG Sidebar sees token: {debug_token_val_sidebar} (Type: {type(debug_token_val_sidebar)})") # Optional debug
current_history_tokens_display = st.session_state.get('last_known_history_token_count', 0)
if not isinstance(current_history_tokens_display, int): current_history_tokens_display = 0 

st.sidebar.metric(label="Approx. History Tokens (MemorySaver)", value=f"{current_history_tokens_display} / {MAX_HISTORY_TOKENS}")
if isinstance(current_history_tokens_display, int) and current_history_tokens_display > MAX_HISTORY_TOKENS:
    st.sidebar.warning("History near/over pruning threshold!")

if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    st.session_state.app_agent_tuple = None 
    
    if 'pinecone_tool_name' in st.session_state: del st.session_state.pinecone_tool_name
    if 'woocommerce_tool_names' in st.session_state: del st.session_state.woocommerce_tool_names
    if 'all_tool_details_for_prompt' in st.session_state: del st.session_state.all_tool_details_for_prompt
    st.session_state.last_known_history_token_count = 0
    
    get_agent_with_memory_tuple_cached.clear() 
    st.rerun()

if st.session_state.messages:
    chat_export_data_txt = "\n\n".join([f"{msg.get('role', 'Unknown').capitalize()}: {msg.get('content', '')}" for msg in st.session_state.messages])
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.sidebar.download_button(label="📥 Download Chat (TXT)", data=chat_export_data_txt, file_name=f"fifi_mcp_chat_{current_time}.txt", mime="text/plain", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("💡 FiFi uses guided memory with auto-pruning and tool prioritization!")

# Main chat display and input
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(str(message.get("content", "")))
if st.session_state.get('thinking_for_ui', False):
    with st.chat_message("assistant"): st.markdown("⌛ FiFi is thinking...")
if st.session_state.get('thinking_for_ui', False) and st.session_state.get('query_to_process') is not None:
    query_to_run = st.session_state.query_to_process
    st.session_state.query_to_process = None
    # This is an entry point for asyncio where an event loop is created if not present
    asyncio.run(execute_agent_call_with_memory(query_to_run)) 
user_prompt = st.chat_input("Ask FiFi Co-Pilot...", key="main_chat_input", disabled=st.session_state.get('thinking_for_ui', False))
if user_prompt: handle_new_query_submission(user_prompt)
