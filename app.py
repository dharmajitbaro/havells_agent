import streamlit as st
import pandas as pd
from typing import Annotated, Literal, TypedDict
import operator

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# ==========================================
# 1. UI and API Setup
# ==========================================
st.set_page_config(page_title="Havells Voice Intel", page_icon="🔌")
st.title("Customer Voice Intelligence Agent")

api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not api_key:
    st.info("Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

# ==========================================
# 2. Mock Data 
# ==========================================
@st.cache_data
def load_data():
    data = {
        "product": ["Air Purifier", "Water Heater", "Mixer Grinder", "Air Purifier", "Water Heater"],
        "review_text": [
            "Filters are way too expensive to replace.",
            "Takes 30 mins to heat up, terrible in winter.",
            "Loud but grinds well.",
            "Filter replacement light is broken, annoying.",
            "Rusting after just 6 months of use!"
        ],
        "date": ["2023-11-01", "2023-12-15", "2024-01-05", "2024-02-10", "2024-03-20"]
    }
    return pd.DataFrame(data)

df = load_data()

# ==========================================
# 3. Define Tools
# ==========================================
def get_product_reviews(product_name: str) -> str:
    """Gets raw reviews for a specific product."""
    reviews = df[df['product'].str.lower() == product_name.lower()]['review_text'].tolist()
    if not reviews:
        return f"No reviews found for {product_name}."
    return "\n".join([f"- {r}" for r in reviews])

def get_recurring_complaints(product_name: str) -> str:
    """Analyzes recent complaints for a product based on reviews."""
    reviews = df[df['product'].str.lower() == product_name.lower()]
    if reviews.empty:
        return "No data available to determine complaints."
    return "Raw complaints data: " + " | ".join(reviews['review_text'].tolist())

tools = [get_product_reviews, get_recurring_complaints]

# ==========================================
# 4. Define Graph Components (Cached)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

@st.cache_resource
def get_compiled_graph(_api_key):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=_api_key)
    llm_with_tools = llm.bind_tools(tools)

    def reasoning_node(state: AgentState):
        sys_msg = SystemMessage(content="""You are an analytical assistant for product managers.
        Answer questions based ONLY on the data provided by your tools. 
        If the data doesn't back up a theme or a number, state that clearly rather than inventing one.""")
        messages = [sys_msg] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    action_node = ToolNode(tools)

    def route(state: AgentState) -> Literal["action", "end"]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "action"
        return "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("action", action_node)
    workflow.add_edge(START, "reasoning")
    workflow.add_conditional_edges("reasoning", route, {"action": "action", "end": END})
    workflow.add_edge("action", "reasoning")
    
    return workflow.compile()

app = get_compiled_graph(api_key)

# ==========================================
# 5. Streamlit Chat Interface
# ==========================================
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    # Only display human and final AI responses, hide internal tool calls from the UI
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage) and not message.tool_calls:
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Accept user input
if prompt := st.chat_input("Ask about product reviews..."):
    # Add user message to chat history and display it
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream the response from the agent
    with st.chat_message("assistant"):
        st_placeholder = st.empty()
        
        # We pass the entire conversation history into the graph state
        inputs = {"messages": st.session_state.messages}
        
        with st.spinner("Analyzing reviews..."):
            final_response_content = ""
            # Run the graph and update session state with the new messages
            for output in app.stream(inputs):
                for key, value in output.items():
                    # Append the generated messages to our session state to keep history intact
                    st.session_state.messages.extend(value["messages"])
                    
                    if key == "reasoning":
                        msg = value["messages"][0]
                        if not msg.tool_calls:
                            final_response_content = msg.content
                            st_placeholder.markdown(final_response_content)
