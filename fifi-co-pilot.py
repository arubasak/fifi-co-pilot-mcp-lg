import streamlit as st
import datetime
import asyncio
import tiktoken  # Import tiktoken

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

# --- Constants for History Pruning ---
# GPT-4.1-mini's context is 200k tokens as per 
# This is for the HISTORY, not a single request's TPM limit.
# TPM is about throughput, this is about context window size for a single call.
MAX_HISTORY_TOKENS = 80000  # Prune if history exceeds this. Adjust based on model (4o-mini might need lower)
MESSAGES_TO_KEEP_AFTER_PRUNING = 6  # Keep ~last 3 user/assistant turns. System prompt is re-added.
TOKEN_MODEL_ENCODING = "cl100k_base"  # Encoding for gpt-4, gpt-3.5-turbo, text-embedding-ada-002, and gpt-4o models

# --- Load environment variables from secrets ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
MCP_PINECONE_URL = st.secrets.get("MCP_PINECONE_URL")
MCP_PINECONE_API_KEY = st.secrets.get("MCP_PINECONE_API_KEY")
MCP_PIPEDREAM_URL = st.secrets.get("MCP_PIPEDREAM_URL")

if not all([OPENAI_API_KEY, MCP_PINECONE_URL, MCP_PINECONE_API_KEY, MCP_PIPEDREAM_URL]):
    st.error("One or more secrets are missing. Please configure them in Streamlit secrets.")
    st.stop()

# --- LangChain LLM (OpenAI GPT-4.1-mini) ---
llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY, temperature=0.2) # Corrected to gpt-4.1-mini

# Define a constant for our conversation thread ID
THREAD_ID = "fifi_streamlit_session"


# --- Token Counting Helper ---
def count_tokens(messages: list, model_encoding: str = TOKEN_MODEL_ENCODING) -> int:
    if not messages:
        return 0
    try:
        encoding = tiktoken.get_encoding(model_encoding)
    except Exception:
        print(f"Warning: Encoding {model_encoding} not found or tiktoken issue. Using 'cl100k_base' as fallback.")
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            if value is not None:
                try:
                    num_tokens += len(encoding.encode(str(value)))
                except TypeError:
                    print(
                        f"Warning: Could not encode value of type {type(value)}. Skipping token count for this value.")
    num_tokens += 2
    return num_tokens


# --- History Pruning Helper (Corrected Version) ---
def prune_history_if_needed(
        memory_instance: MemorySaver,
        thread_config: dict,
        current_system_prompt_content: str,
        max_tokens: int,
        keep_last_n_interactions: int
):
    checkpoint_value = memory_instance.get(thread_config)

    if not checkpoint_value or "messages" not in checkpoint_value or \
            not isinstance(checkpoint_value.get("messages"), list):
        return False

    current_messages_in_history = checkpoint_value["messages"]
    if not current_messages_in_history:
        return False

    current_token_count = count_tokens(current_messages_in_history)
    # print(f"DEBUG (prune): Current history token count: {current_token_count}") 


    if current_token_count > max_tokens:
        print(f"INFO: History token count ({current_token_count}) > max ({max_tokens}). Pruning...")

        user_assistant_messages = [m for m in current_messages_in_history if m.get("role") != "system"]
        pruned_user_assistant_messages = user_assistant_messages[-keep_last_n_interactions:]

        new_history_messages = []
        new_history_messages.append({"role": "system", "content": current_system_prompt_content})
        new_history_messages.extend(pruned_user_assistant_messages)

        new_checkpoint_value_to_put = {"messages": new_history_messages}

        memory_instance.put(thread_config, new_checkpoint_value_to_put)
        pruned_token_count = count_tokens(new_history_messages)
        print(
            f"INFO: History pruned. New token count: {pruned_token_count}. Kept {len(pruned_user_assistant_messages)} user/assistant messages.")
        return True
    return False


# --- Async function to initialize LangChain Agent with Memory ---
@st.cache_resource
async def get_agent_with_memory_tuple():  # Renamed to reflect it returns a tuple
    print("Initializing agent with memory and fetching tools...")
    client = MultiServerMCPClient(
        {
            "pinecone": {"url": MCP_PINECONE_URL, "transport": "sse",
                         "headers": {"Authorization": f"Bearer {MCP_PINECONE_API_KEY}"}},
            "pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}
        }
    )
    tools = await client.get_tools()

    st.session_state.pinecone_tool_name = "functions.get_context"
    st.session_state.woocommerce_tool_names = []
    st.session_state.all_tool_details_for_prompt = {}

    print("--- Identifying Tools based on provided list ---")
    for tool in tools:
        st.session_state.all_tool_details_for_prompt[tool.name] = tool.description
        if tool.name == "functions.get_context":
            st.session_state.pinecone_tool_name = tool.name
            # print(f"Confirmed Pinecone/get_context tool: {tool.name}") # Less verbose
        elif "woocommerce" in tool.name.lower():
            st.session_state.woocommerce_tool_names.append(tool.name)
    # print(f"Confirmed Pinecone tool: {st.session_state.pinecone_tool_name}") # Less verbose
    if not st.session_state.woocommerce_tool_names:
        print("Warning: No WooCommerce tools were identified based on 'woocommerce' in name.")
    # else: # Less verbose
        # print(f"Identified WooCommerce tools: {st.session_state.woocommerce_tool_names}")
    print("-------------------------------------------------")

    memory = MemorySaver()
    agent_executor = create_react_agent(llm, tools, checkpointer=memory)
    print("Agent with memory initialized successfully.")
    return agent_executor, memory


async def ensure_agent_is_initialized():
    if 'agent_with_memory_tuple' not in st.session_state or st.session_state.agent_with_memory_tuple is None:
        st.session_state.agent_with_memory_tuple = await get_agent_with_memory_tuple()

    if 'pinecone_tool_name' not in st.session_state: st.session_state.pinecone_tool_name = "functions.get_context"
    if 'woocommerce_tool_names' not in st.session_state: st.session_state.woocommerce_tool_names = []
    if 'all_tool_details_for_prompt' not in st.session_state: st.session_state.all_tool_details_for_prompt = {}

    return st.session_state.agent_with_memory_tuple


# --- Initialize session state ---
if "messages" not in st.session_state: st.session_state.messages = []
if 'thinking_for_ui' not in st.session_state: st.session_state.thinking_for_ui = False
if 'query_to_process' not in st.session_state: st.session_state.query_to_process = None
if 'pinecone_tool_name' not in st.session_state: st.session_state.pinecone_tool_name = "functions.get_context"
if 'woocommerce_tool_names' not in st.session_state: st.session_state.woocommerce_tool_names = []
if 'all_tool_details_for_prompt' not in st.session_state: st.session_state.all_tool_details_for_prompt = {}


# --- System Prompt Definition ---
# --- System Prompt Definition ---
def get_system_prompt():
    pinecone_tool = st.session_state.get('pinecone_tool_name', "functions.get_context")
    # The description of the pinecone_tool is fetched and used directly, no need for a separate variable here if only used once.
    # woocommerce_tools_exist = bool(st.session_state.get('woocommerce_tool_names')) # Not directly used in this version of prompt

    prompt = f"""You are FiFi, an expert AI assistant for 1-2-Taste. Your **sole purpose** is to assist with inquiries about 1-2-Taste's products, the food/beverage ingredients industry, relevant food science, B2B inquiries, recipes using 1-2-Taste ingredients, and 1-2-Taste WooCommerce e-commerce functions.

**Core Directives:**
*   Provide accurate, **cited** information on 1-2-Taste offerings via your product information tool.
*   Assist with relevant 1-2-Taste e-commerce tasks if explicitly requested.
*   Politely decline queries clearly outside your designated scope.

**Tool Usage (Internal LLM Instructions):**

1.  **Product/Industry Info (Tool: `{pinecone_tool}`):**
    *   **ALWAYS PRIORITIZE** this tool for ANY query potentially related to 1-2-Taste (products, ingredients, flavors, availability, specs, recipes, applications, industry trends, catalog info).
    *   Description: "{st.session_state.all_tool_details_for_prompt.get(pinecone_tool, 'Retrieves relevant document snippets from the assistant knowledge base.')}"
    *   For ambiguous queries (e.g., "vanilla"), assume 1-2-Taste context and use this tool first.

2.  **E-commerce (WooCommerce Tools like `functions.WOOCOMMERCE-GET-ORDER`):**
    *   **ONLY** use these if the query EXPLICITLY mentions "WooCommerce", "orders", "my order", "customer accounts", "shipping", "store management", "cart", or similar specific e-commerce administrative tasks for 1-2-Taste.
    *   NOT for general product info.

**User Communication:**
*   **Capabilities:** If asked what you can do, describe your functions simply (e.g., "I provide info on 1-2-Taste products and ingredients," "I can help with recipe ideas," "I can assist with 1-2-Taste order inquiries."). **NEVER reveal internal tool names** like `functions.get_context`.
*   **Out-of-Scope Queries:** For clearly unrelated topics (celebrity news, history, sports, general programming), **MUST POLITELY DECLINE**. Examples: "My apologies, I specialize in 1-2-Taste and food ingredients topics." or "That's outside my area of expertise for 1-2-Taste."
*   **General Knowledge:** **AVOID.** As a last resort, if the `{pinecone_tool}` fails on a *potentially relevant* food/ingredient query, a very brief, confident general answer *may* be given. If unsure, decline. NEVER for clearly unrelated topics.

**Response Format & Guidelines:**
*   **Mandatory Citations:** ALWAYS cite sources for product info from `{pinecone_tool}` if the tool provides a URL. Format: "[Source: URL]" or similar. If no URL is provided by the tool for a specific fact, state info is from the 1-2-Taste catalog. Cite alternatives for discontinued products.
*   **No Prices:** Thank users for asking about price and direct them to the product page or sales-eu@12taste.com.
*   **Quote Only:** For "(QUOTE ONLY)" products with missing prices, direct to https://www.12taste.com/request-quote/.
*   **Conciseness:** Keep answers brief and to the point.

Answer the user's last query based on these instructions and the conversation history.
"""
    return prompt


# --- Async handler for user queries, using agent with memory ---
async def execute_agent_call_with_memory(user_query: str):
    assistant_reply = ""
    try:
        agent_executor, memory_instance = await ensure_agent_is_initialized()

        if agent_executor is None or memory_instance is None:
            st.error("Agent or Memory could not be initialized.")
            assistant_reply = "(Error: Agent/Memory not initialized)"
        else:
            config = {"configurable": {"thread_id": THREAD_ID}}
            system_prompt_content = get_system_prompt()

            was_pruned = prune_history_if_needed(
                memory_instance,
                config,
                system_prompt_content,
                MAX_HISTORY_TOKENS,
                MESSAGES_TO_KEEP_AFTER_PRUNING
            )
            # if was_pruned: # No need to print this in the non-debug version
                # print(f"INFO: History was pruned for thread {THREAD_ID}")

            current_turn_messages = [
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": user_query}
            ]
            event = {"messages": current_turn_messages}

            result = await agent_executor.ainvoke(event, config=config)

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
    st.rerun()


# --- Input Handling Function ---
def handle_new_query_submission(query_text: str):
    if not st.session_state.get('thinking_for_ui', False):
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.session_state.query_to_process = query_text
        st.session_state.thinking_for_ui = True
        st.rerun()


# --- Streamlit UI ---
st.title("FiFi Co-Pilot 🚀 (LangGraph MCP Agent with Auto-Pruning Memory)")

if 'agent_with_memory_tuple' not in st.session_state:
    asyncio.run(ensure_agent_is_initialized())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(str(message.get("content", "")))

if st.session_state.get('thinking_for_ui', False):
    with st.chat_message("assistant"):
        st.markdown("⌛ FiFi is thinking...")

if st.session_state.get('thinking_for_ui', False) and st.session_state.get('query_to_process') is not None:
    query_to_run = st.session_state.query_to_process
    st.session_state.query_to_process = None
    asyncio.run(execute_agent_call_with_memory(query_to_run))

st.sidebar.markdown("## Quick Questions")
preview_questions = [
    "Help me with my recipe for a new juice drink",
    "Suggest me some strawberry flavours for beverage",
    "I need vanilla flavours for ice-cream"
]
for question in preview_questions:
    if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True):
        handle_new_query_submission(question)

user_prompt = st.chat_input("Ask FiFi Co-Pilot...", key="main_chat_input",
                            disabled=st.session_state.get('thinking_for_ui', False))
if user_prompt:
    handle_new_query_submission(user_prompt)

if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    if 'agent_with_memory_tuple' in st.session_state:
        del st.session_state.agent_with_memory_tuple

    if 'pinecone_tool_name' in st.session_state: del st.session_state.pinecone_tool_name
    if 'woocommerce_tool_names' in st.session_state: del st.session_state.woocommerce_tool_names
    if 'all_tool_details_for_prompt' in st.session_state: del st.session_state.all_tool_details_for_prompt
    # No UI token counter state to clear here.
    get_agent_with_memory_tuple.clear()
    st.rerun()

if st.session_state.messages:
    chat_export_data_txt = "\n\n".join(
        [f"{str(msg.get('role', 'Unknown')).capitalize()}: {str(msg.get('content', ''))}" for msg in
         st.session_state.messages]
    )
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.sidebar.download_button(
        label="📥 Download Chat (TXT)",
        data=chat_export_data_txt,
        file_name=f"fifi_mcp_chat_{current_time}.txt",
        mime="text/plain",
        use_container_width=True
    )

st.sidebar.markdown("---")
st.sidebar.info("💡 FiFi uses guided memory with auto-pruning and tool prioritization!")
