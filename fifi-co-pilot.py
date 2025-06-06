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
    # print(f"DEBUG (prune): Current history token count: {current_token_count}") 
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
async def get_agent_with_memory_tuple_cached(): # Renamed for clarity of its role
    print("INVOKING ASYNC: Initializing agent with memory and fetching tools (cached)...")
    client = MultiServerMCPClient(
        {
            "pinecone": {"url": MCP_PINECONE_URL, "transport": "sse", "headers": {"Authorization": f"Bearer {MCP_PINECONE_API_KEY}"}},
            "pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}
        }
    )
    tools = await client.get_tools()
    
    # Initialize these here as they depend on tools being available
    # Although pinecone_tool_name is hardcoded, this ensures consistency if it were dynamic
    pinecone_tool_name = "functions.get_context"
    woocommerce_tool_names = []
    all_tool_details = {}

    print("--- Identifying Tools based on provided list (inside cached function) ---")
    for tool in tools:
        all_tool_details[tool.name] = tool.description
        if tool.name == "functions.get_context":
            pinecone_tool_name = tool.name # Confirm
        elif "woocommerce" in tool.name.lower():
            woocommerce_tool_names.append(tool.name)
    
    # Store identified names directly for use by get_system_prompt
    st.session_state.pinecone_tool_name = pinecone_tool_name
    st.session_state.woocommerce_tool_names = woocommerce_tool_names
    st.session_state.all_tool_details_for_prompt = all_tool_details

    print(f"Confirmed Pinecone tool (cached): {st.session_state.pinecone_tool_name}")
    if not st.session_state.woocommerce_tool_names: print("Warning (cached): No WooCommerce tools identified.")
    else: print(f"Identified WooCommerce tools (cached): {st.session_state.woocommerce_tool_names}")
    print("-------------------------------------------------")

    memory = MemorySaver()
    agent_executor = create_react_agent(llm, tools, checkpointer=memory)
    print("Agent with memory initialized successfully (cached).")
    return agent_executor, memory

# This function will now be synchronous and rely on st.cache_resource to run the async part.
def ensure_agent_is_initialized_and_get_tuple():
    # st.cache_resource handles the awaiting of the async function internally.
    # We call the cached async function directly.
    # The result (the tuple) is what will be stored in session state.
    if 'agent_with_memory_tuple' not in st.session_state or st.session_state.agent_with_memory_tuple is None:
        # Since get_agent_with_memory_tuple_cached is async and cached by st.cache_resource,
        # Streamlit needs to run it in an event loop if one isn't present.
        # However, st.cache_resource should manage this.
        # The problem arises if we try to asyncio.run() a function that itself calls
        # another async function that is @st.cache_resource decorated *and* we try to manage the result
        # in a complex way.
        # The simplest way with @st.cache_resource on an async function is to just call it.
        # Streamlit will run it if needed.
        # BUT, we can't call an async function directly from sync code without asyncio.run() or await.
        # This means the top-level call to initialize needs to be async OR
        # the cached function itself needs to be a synchronous wrapper that internally runs the async code.
        
        # Let's try making this outer function the one that runs the asyncio.run
        # and the cached function is purely async.
        # This seems to be what's causing the loop issue.

        # The @st.cache_resource on an async function means Streamlit will await it.
        # We just need to call it.
        # The issue is calling this from the top-level script.
        # Let's assume the top-level asyncio.run() is what gets the cached resource.
        # This function's primary job is to ensure session state is populated.
        # It shouldn't re-await. It should just get the result from the cache if available,
        # or trigger the first call.

        # This function is called by asyncio.run() at the script top level.
        # It means it's already in an event loop context provided by that asyncio.run().
        # So, we can await here.
        # The function being awaited is the @st.cache_resource one.
        # The RuntimeError: cannot reuse already awaited coroutine happens if
        # what get_agent_with_memory_tuple_cached() returns *after the first time*
        # is the *original coroutine object* instead of its *result*.
        # This indicates a misunderstanding of how st.cache_resource handles async functions
        # or how it's being called.

        # Let's simplify: The cached function IS the source of truth.
        # We ensure it's called, and its result is what we use.
        # The @st.cache_resource decorator itself should return the RESULT of the
        # awaited coroutine, not the coroutine itself, on subsequent calls.
        
        # This function will be called via asyncio.run() at the top.
        # So it's okay to await the cached resource here.
        # The error implies that st.cache_resource is returning the coroutine object itself on cache hit.
        # This is unusual. Let's assume for a moment st.cache_resource handles the await.
        # The pattern recommended by Streamlit for caching async resources
        # is often to have the @st.cache_resource function be the one called directly.
        
        # The error is likely because ensure_agent_is_initialized itself is an async function
        # being called by asyncio.run(), and *within it*, we again await the cached resource.
        # If st.cache_resource is already handling the await, this is redundant and problematic.

        # The solution is usually to have a synchronous function that gets the resource,
        # and if that resource needs to be created by an async function, the sync function
        # does the asyncio.run() once. But here @st.cache_resource is on the async function.

        # If `get_agent_with_memory_tuple_cached` is decorated with `@st.cache_resource`,
        # Streamlit itself will await this function when it's called for the first time
        # from a synchronous context (or if called with await from an async context).
        # The key is that `get_agent_with_memory_tuple_cached()` will return the *actual tuple result*
        # after the first await, not the coroutine object.

        # This function is async, and is called via asyncio.run()
        # So we can `await` the cached function call here.
        st.session_state.agent_with_memory_tuple = asyncio.run(get_agent_with_memory_tuple_cached())
        # No, this is wrong. If ensure_agent_is_initialized is already async, we await:
        # st.session_state.agent_with_memory_tuple = await get_agent_with_memory_tuple_cached()

    # Ensure other session state vars (tool names) are populated,
    # as they are set within get_agent_with_memory_tuple_cached
    if 'pinecone_tool_name' not in st.session_state: 
        # This indicates get_agent_with_memory_tuple_cached hasn't run or set them
        # This should not happen if agent_with_memory_tuple is populated
        print("WARNING: pinecone_tool_name not in session_state after ensuring agent init.")
        st.session_state.pinecone_tool_name = "functions.get_context"
    if 'woocommerce_tool_names' not in st.session_state: 
        st.session_state.woocommerce_tool_names = []
    if 'all_tool_details_for_prompt' not in st.session_state: 
        st.session_state.all_tool_details_for_prompt = {}
        
    return st.session_state.agent_with_memory_tuple


# --- Re-thinking initialization for Streamlit's @st.cache_resource with async ---
# The @st.cache_resource decorated function is the one that should be called.
# Streamlit handles its execution.

# Initialize these session states with defaults first
if "messages" not in st.session_state: st.session_state.messages = []
if 'thinking_for_ui' not in st.session_state: st.session_state.thinking_for_ui = False
if 'query_to_process' not in st.session_state: st.session_state.query_to_process = None
if 'pinecone_tool_name' not in st.session_state: st.session_state.pinecone_tool_name = "functions.get_context"
if 'woocommerce_tool_names' not in st.session_state: st.session_state.woocommerce_tool_names = []
if 'all_tool_details_for_prompt' not in st.session_state: st.session_state.all_tool_details_for_prompt = {}
if 'last_known_history_token_count' not in st.session_state: st.session_state.last_known_history_token_count = 0
# This session state will hold the *result* of the cached async function
if 'cached_agent_tuple' not in st.session_state:
    st.session_state.cached_agent_tuple = None


def get_system_prompt():
    # This function now relies on tool names being in st.session_state
    # which are set by get_agent_with_memory_tuple_cached() when it runs.
    pinecone_tool = st.session_state.get('pinecone_tool_name', "functions.get_context") # Fallback
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
        # Get the cached agent tuple. Streamlit's @st.cache_resource handles awaiting the async function.
        # This call will return the cached result if available, or run the async function and cache its result.
        # We need to run this in an event loop if called from sync code.
        # Since execute_agent_call_with_memory is async, we can await it.
        agent_executor, memory_instance = await get_agent_with_memory_tuple_cached()
        st.session_state.cached_agent_tuple = (agent_executor, memory_instance) # Store it if fetched

        if agent_executor is None or memory_instance is None:
            st.error("Agent or Memory could not be initialized.")
            assistant_reply = "(Error: Agent/Memory not initialized)"
        else:
            config = {"configurable": {"thread_id": THREAD_ID}}
            system_prompt_content = get_system_prompt()
            
            # Debug: Check MemorySaver content before pruning
            checkpoint_before_prune = memory_instance.get(config)
            # print(f"DEBUG (execute): Checkpoint value BEFORE PRUNING: {checkpoint_before_prune}")


            was_pruned = prune_history_if_needed(
                memory_instance, config, system_prompt_content,
                MAX_HISTORY_TOKENS, MESSAGES_TO_KEEP_AFTER_PRUNING
            )
            if was_pruned:
                checkpoint_after_prune = memory_instance.get(config)
                st.session_state.last_known_history_token_count = count_tokens(
                    checkpoint_after_prune.get("messages", []) if checkpoint_after_prune else []
                )
                # print(f"DEBUG (execute): UI Token count updated AFTER PRUNE: {st.session_state.last_known_history_token_count}")

            current_turn_messages = [
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": user_query}
            ]
            event = {"messages": current_turn_messages}
            
            # Debug: Check MemorySaver content before ainvoke
            # checkpoint_before_ainvoke = memory_instance.get(config)
            # print(f"DEBUG (execute): Checkpoint value BEFORE AINVOKE (after potential prune): {checkpoint_before_ainvoke}")


            result = await agent_executor.ainvoke(event, config=config)

            # Debug: Check MemorySaver content after ainvoke
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
    
    # Update UI token count
    if st.session_state.cached_agent_tuple:
        _, memory_instance_for_ui = st.session_state.cached_agent_tuple
        if memory_instance_for_ui:
            config_for_ui = {"configurable": {"thread_id": THREAD_ID}}
            checkpoint_val = memory_instance_for_ui.get(config_for_ui)
            if checkpoint_val and "messages" in checkpoint_val:
                st.session_state.last_known_history_token_count = count_tokens(checkpoint_val["messages"])
                # print(f"DEBUG (execute): UI Token count updated at END: {st.session_state.last_known_history_token_count}")
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

# This top-level call to get the agent will be handled by Streamlit's main thread.
# @st.cache_resource on an async function means Streamlit will run it in an event loop.
# We then store this result in session_state to avoid re-running the async init logic unnecessarily
# from within other async functions like execute_agent_call_with_memory.
if st.session_state.cached_agent_tuple is None:
    print("Main script: cached_agent_tuple is None. Attempting to initialize.")
    # This is tricky. Calling an async function from the top level of a Streamlit script
    # usually requires asyncio.run().
    # However, @st.cache_resource is meant to simplify this.
    # Let's try calling it and letting Streamlit manage the await.
    # If this still causes issues, the pattern is to have a sync wrapper that calls asyncio.run().
    try:
        # The recommended way is to call the cached function directly.
        # Streamlit should handle running it.
        # However, we can't assign the result of an async function directly without await or asyncio.run
        # This indicates a structural issue if we need its result immediately here.
        # The solution is often to ensure it's called, and other parts of the app
        # that need it will then get the cached result.

        # Let's ensure it's at least *called* if not already cached.
        # The actual result will be retrieved within execute_agent_call_with_memory
        # For the initial token count, we need to ensure it has run.
        # This becomes a bit of a chicken-and-egg if we need the result right here synchronously.
        
        # Simpler approach: Initialize the agent tuple using asyncio.run here ONCE
        # and store it. execute_agent_call_with_memory will then just use this.
        # The @st.cache_resource will ensure get_agent_with_memory_tuple_cached itself
        # only does its work once.
        st.session_state.cached_agent_tuple = asyncio.run(get_agent_with_memory_tuple_cached())
        print("Main script: cached_agent_tuple initialized.")

        # Initialize token count for UI after agent is ready
        if st.session_state.cached_agent_tuple:
            _, memory_instance_for_init_ui = st.session_state.cached_agent_tuple
            if memory_instance_for_init_ui:
                config_for_init_ui = {"configurable": {"thread_id": THREAD_ID}}
                checkpoint_val_init = memory_instance_for_init_ui.get(config_for_init_ui)
                if checkpoint_val_init and "messages" in checkpoint_val_init:
                    st.session_state.last_known_history_token_count = count_tokens(checkpoint_val_init["messages"])
                else:
                    st.session_state.last_known_history_token_count = 0
                print(f"DEBUG (UI init): UI Token count initialized: {st.session_state.last_known_history_token_count}")
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # This can happen if Streamlit is already running an event loop (e.g. for websockets)
            # In this case, we can't use asyncio.run().
            # We have to rely on execute_agent_call_with_memory to fetch it via await.
            # The initial token count might be delayed or show N/A until first interaction.
            print(f"INFO: Could not run initial agent setup due to existing event loop: {e}")
            st.session_state.cached_agent_tuple = "PENDING_INIT_IN_ASYNC_CALL" # Placeholder
            st.session_state.last_known_history_token_count = "N/A (Initializing...)"
        else:
            raise # Re-raise other RuntimeErrors

# Sidebar
st.sidebar.markdown("## Quick Questions")
preview_questions = ["Help me with my recipe for a new juice drink", "Suggest me some strawberry flavours for beverage", "I need vanilla flavours for ice-cream"]
for question in preview_questions:
    if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True):
        handle_new_query_submission(question)

st.sidebar.markdown("---")
debug_token_val_sidebar = st.session_state.get('last_known_history_token_count', "N/A")
st.sidebar.write(f"DEBUG Sidebar sees token: {debug_token_val_sidebar} (Type: {type(debug_token_val_sidebar)})")
current_history_tokens_display = st.session_state.get('last_known_history_token_count', 0)
if not isinstance(current_history_tokens_display, int): current_history_tokens_display = 0 # Ensure it's a number for metric

st.sidebar.metric(label="Approx. History Tokens (MemorySaver)", value=f"{current_history_tokens_display} / {MAX_HISTORY_TOKENS}")
if isinstance(current_history_tokens_display, int) and current_history_tokens_display > MAX_HISTORY_TOKENS:
    st.sidebar.warning("History near/over pruning threshold!")

if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    st.session_state.cached_agent_tuple = None # Clear our specific session state holder
    
    # Clear session state for tool names
    if 'pinecone_tool_name' in st.session_state: del st.session_state.pinecone_tool_name
    if 'woocommerce_tool_names' in st.session_state: del st.session_state.woocommerce_tool_names
    if 'all_tool_details_for_prompt' in st.session_state: del st.session_state.all_tool_details_for_prompt
    st.session_state.last_known_history_token_count = 0
    
    get_agent_with_memory_tuple_cached.clear() # Clear the st.cache_resource
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
    asyncio.run(execute_agent_call_with_memory(query_to_run)) # This is an entry point for asyncio
user_prompt = st.chat_input("Ask FiFi Co-Pilot...", key="main_chat_input", disabled=st.session_state.get('thinking_for_ui', False))
if user_prompt: handle_new_query_submission(user_prompt)
