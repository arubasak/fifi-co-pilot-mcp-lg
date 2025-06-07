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
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            if value is not None:
                try: num_tokens += len(encoding.encode(str(value)))
                except TypeError: pass
    num_tokens += 2
    return num_tokens

def prune_history_if_needed(memory_instance: MemorySaver, thread_config: dict, current_system_prompt_content: str, max_tokens: int, keep_last_n_interactions: int):
    checkpoint_value = memory_instance.get(thread_config)
    if not checkpoint_value or "messages" not in checkpoint_value or not isinstance(checkpoint_value.get("messages"), list):
        return False
    current_messages_in_history = checkpoint_value["messages"]
    if not current_messages_in_history: return False
    current_token_count = count_tokens(current_messages_in_history)
    if current_token_count > max_tokens:
        print(f"INFO: History token count ({current_token_count}) > max ({max_tokens}). Pruning...")
        user_assistant_messages = [m for m in current_messages_in_history if m.get("role") != "system"]
        pruned_user_assistant_messages = user_assistant_messages[-keep_last_n_interactions:]
        new_history_messages = [{"role": "system", "content": current_system_prompt_content}]
        new_history_messages.extend(pruned_user_assistant_messages)
        memory_instance.put(thread_config, {"messages": new_history_messages})
        return True
    return False

@st.cache_resource(ttl=3600)
async def initialize_agent_components():
    client = MultiServerMCPClient({
        "pinecone": {"url": MCP_PINECONE_URL, "transport": "sse", "headers": {"Authorization": f"Bearer {MCP_PINECONE_API_KEY}"}},
        "pipedream": {"url": MCP_PIPEDREAM_URL, "transport": "sse"}
    })
    tools = await client.get_tools()

    pinecone_tool_name = "functions.get_context"
    woocommerce_tool_names = []
    all_tool_details = {}

    for tool in tools:
        all_tool_details[tool.name] = tool.description
        if "woocommerce" in tool.name.lower():
            woocommerce_tool_names.append(tool.name)

    memory = MemorySaver()
    agent_executor = create_react_agent(llm, tools, checkpointer=memory)
    return {
        "agent_executor": agent_executor,
        "memory_instance": memory,
        "pinecone_tool_name": pinecone_tool_name,
        "woocommerce_tool_names": woocommerce_tool_names,
        "all_tool_details_for_prompt": all_tool_details
    }

# ✅ Updated load_components() to avoid asyncio.run()
def load_components():
    if "components_loaded" not in st.session_state or not st.session_state.components_loaded:
        print("@@@ load_components: components_loaded flag not set or false. Calling initialize_agent_components() directly.")
        try:
            components_future = initialize_agent_components()
            if hasattr(components_future, "__await__"):
                st.warning("Initializing agent components, please wait a moment...")
                st.stop()
            else:
                components = components_future
                st.session_state.agent_executor = components["agent_executor"]
                st.session_state.memory_instance = components["memory_instance"]
                st.session_state.pinecone_tool_name = components["pinecone_tool_name"]
                st.session_state.woocommerce_tool_names = components["woocommerce_tool_names"]
                st.session_state.all_tool_details_for_prompt = components["all_tool_details_for_prompt"]
                st.session_state.components_loaded = True
                print("@@@ load_components: Components loaded into session state successfully.")
        except Exception as e:
            st.error(f"Critical error during agent initialization: {e}")
            st.session_state.components_loaded = False
            raise
    else:
        print("@@@ load_components: components_loaded flag is true. Components should be available.")

# The rest of your script (system prompt, execute_agent_call_with_memory, handle_new_query_submission, UI) remains unchanged.
# Use load_components() as you have been.

# ... (rest of your existing Streamlit app and UI logic remains untouched)

# --- Initialize session state (basic flags) ---
if "messages" not in st.session_state: st.session_state.messages = []
if 'thinking_for_ui' not in st.session_state: st.session_state.thinking_for_ui = False
if 'query_to_process' not in st.session_state: st.session_state.query_to_process = None
# Tool names and other components will be set by load_components()

# --- System Prompt Definition ---
def get_system_prompt():
    pinecone_tool = st.session_state.get('pinecone_tool_name', "functions.get_context") 
    all_tool_details = st.session_state.get('all_tool_details_for_prompt', {})
    prompt = f"""You are FiFi, an expert AI assistant for 1-2-Taste. Your **sole purpose** is to assist users with inquiries related to 1-2-Taste's products, the food and beverage ingredients industry, food science topics relevant to 1-2-Taste's offerings, B2B inquiries, recipe development support using 1-2-Taste ingredients, and specific e-commerce functions related to 1-2-Taste's WooCommerce platform.

**Core Mission:**
*   Provide accurate, **cited** information about 1-2-Taste's offerings using your product information capabilities.
*   Assist with relevant e-commerce tasks if explicitly requested by the user in a way that matches your e-commerce functions.
*   Politely decline to answer questions that are outside of your designated scope.

**Tool Usage Priority and Guidelines (Internal Instructions for You, the LLM):**

1.  **Primary Product & Industry Information Tool (Internally known as `{pinecone_tool}`):**
    *   For ANY query that could relate to 1-2-Taste product details, ingredients, flavors, availability, specifications, recipes, applications, food industry trends relevant to 1-2-Taste, or any information found within the 1-2-Taste catalog or relevant to its business, you **MUST ALWAYS PRIORITIZE** using this specialized tool (internally, its name is `{pinecone_tool}`). Its description is: "{all_tool_details.get(pinecone_tool, 'Retrieves relevant document snippets from the assistant knowledge base.')}" This is your main and most reliable knowledge source for product-related questions.
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

**Response Guidelines & Output Format:**
*   **Mandatory Citations for Product Information:** When providing any information obtained from your primary product information tool (internally `{pinecone_tool}`), you **MUST ALWAYS** include a citation to the source URL or product page link if that information is available from the tool's output.
    *   Format citations clearly, for example: "You can find more details here: [Source URL]" or append "[Source: URL]" after the relevant sentence.
    *   If multiple products are mentioned, cite each one appropriately if possible.
    *   **If the tool provides product information but no specific source URL for a piece of that information, state that the information is from the 1-2-Taste catalog without providing a broken link.**
*   If a product is discontinued according to your product information tool, inform the user and, if possible, suggest alternatives found via the same tool (citing them as well).
*   **Do not provide product prices.** Instead, thank the user for asking and direct them to the product page on the 1-2-Taste website or to contact sales-eu@12taste.com.
*   If a product is marked as (QUOTE ONLY) and price is missing, ask them to visit: https://www.12taste.com/request-quote/.
*   Keep answers concise and to the point.

Answer the user's last query based on these instructions and the conversation history.
"""
    return prompt

# --- Async handler for user queries, using agent with memory ---
async def execute_agent_call_with_memory(user_query: str):
    assistant_reply = ""
    try:
        # Retrieve components from session state, ensure they are loaded
        if not st.session_state.get("components_loaded"):
            print("@@@ execute_agent_call: FATAL - components not loaded before execution attempt.")
            st.error("Agent is not ready. Please try refreshing the page or wait a moment.")
            st.session_state.messages.append({"role": "assistant", "content": "(Critical Error: Agent not ready. Please refresh.)"})
            st.session_state.thinking_for_ui = False
            st.rerun()
            return

        agent_executor = st.session_state.get("agent_executor")
        memory_instance = st.session_state.get("memory_instance")

        if agent_executor is None or memory_instance is None:
            st.error("Agent or Memory instance is missing from session state. This should not happen.")
            assistant_reply = "(Error: Agent/Memory components missing)"
        else:
            config = {"configurable": {"thread_id": THREAD_ID}}
            system_prompt_content = get_system_prompt()

            was_pruned = prune_history_if_needed(
                memory_instance, config, system_prompt_content,
                MAX_HISTORY_TOKENS, MESSAGES_TO_KEEP_AFTER_PRUNING
            )
            # if was_pruned: print(f"INFO: History was pruned for thread {THREAD_ID}")

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

# --- Streamlit App Starts Here ---
st.title("FiFi Co-Pilot 🚀 (LangGraph MCP Agent with Auto-Pruning Memory)")

# --- Initialize agent resources ONCE at the beginning of the script ---
load_components() # Call the synchronous wrapper function

# --- UI Rendering ---
# Sidebar
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
    st.session_state.messages = []
    st.session_state.thinking_for_ui = False
    st.session_state.query_to_process = None
    
    for key in ["agent_executor", "memory_instance", 
                "pinecone_tool_name", "woocommerce_tool_names", 
                "all_tool_details_for_prompt", "components_loaded"]:
        if key in st.session_state:
            del st.session_state[key]
            
    initialize_agent_components.clear() # Clear the cache 
    print("@@@ Chat history cleared, cache cleared, session state for agent components cleared.")
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

# Main chat area
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(str(message.get("content", "")))

if st.session_state.get('thinking_for_ui', False):
    with st.chat_message("assistant"):
        st.markdown("⌛ FiFi is thinking...")

if st.session_state.get('thinking_for_ui', False) and st.session_state.get('query_to_process') is not None:
    # Ensure components are loaded before processing a query
    if not st.session_state.get("components_loaded"):
        print("@@@ Query processing: Components not loaded. This indicates an issue with initial load.")
        st.error("Agent is not ready. Please refresh the page.")
        # Optionally clear thinking_for_ui and query_to_process to prevent loop
        st.session_state.thinking_for_ui = False
        st.session_state.query_to_process = None
        st.rerun()
    else:
        query_to_run = st.session_state.query_to_process
        st.session_state.query_to_process = None
        asyncio.run(execute_agent_call_with_memory(query_to_run))


user_prompt = st.chat_input("Ask FiFi Co-Pilot...", key="main_chat_input",
                            disabled=st.session_state.get('thinking_for_ui', False))
if user_prompt:
    handle_new_query_submission(user_prompt)
