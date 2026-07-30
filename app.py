import streamlit as st
import pandas as pd
from typing import Annotated, Literal, TypedDict
import operator

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# ==========================================
# 1. UI and API Setup
# ==========================================
st.set_page_config(page_title="Havells Voice Intel", page_icon="🔌")
st.title("Customer Voice Intelligence Agent")

api_key = st.sidebar.text_input("Groq API Key", type="password") or st.secrets.get("GROQ_API_KEY", "")
if not api_key:
    st.info("Please enter your Groq API key in the sidebar to continue.")
    st.stop()

# ==========================================
# 2. Synthetic Review Dataset
# (Real system: replace with a scraped/open dataset. Kept synthetic here so
# trends over time are clearly visible for demo purposes.)
# ==========================================
@st.cache_data
def load_data():
    data = [
        # Air Purifier
        ("Air Purifier", "R001", "2024-01-05", "Filters are way too expensive to replace."),
        ("Air Purifier", "R002", "2024-02-10", "Filter replacement light is broken, annoying."),
        ("Air Purifier", "R003", "2024-02-22", "Filter cost is really high compared to competitors."),
        ("Air Purifier", "R004", "2024-03-15", "Great air quality but filters need replacing too often."),
        ("Air Purifier", "R005", "2024-04-02", "App connectivity keeps dropping, very frustrating."),
        ("Air Purifier", "R006", "2024-04-20", "Bluetooth pairing fails constantly, app is unreliable."),
        ("Air Purifier", "R007", "2024-05-11", "Replacement filters are overpriced, feels like a scam."),
        ("Air Purifier", "R008", "2024-05-28", "App crashes every time I try to check filter status."),
        ("Air Purifier", "R009", "2024-06-14", "Filter costs are the only downside, otherwise decent."),
        # Water Heater
        ("Water Heater", "R010", "2024-01-08", "Takes 30 mins to heat up, terrible in winter."),
        ("Water Heater", "R011", "2024-01-25", "Rusting after just 6 months of use!"),
        ("Water Heater", "R012", "2024-02-18", "Heating is slow, not ideal for morning rush."),
        ("Water Heater", "R013", "2024-03-09", "Body started rusting near the base, disappointing."),
        ("Water Heater", "R014", "2024-04-05", "Rust spots appearing on the outer casing already."),
        ("Water Heater", "R015", "2024-04-30", "Heats up faster than my old one, happy with speed."),
        ("Water Heater", "R016", "2024-05-19", "Rusting issue seems common, saw others complain too."),
        ("Water Heater", "R017", "2024-06-02", "No rust so far, but it's only been a month."),
        # Mixer Grinder
        ("Mixer Grinder", "R018", "2024-01-14", "Loud but grinds well."),
        ("Mixer Grinder", "R019", "2024-02-06", "Very noisy, wakes up the whole house."),
        ("Mixer Grinder", "R020", "2024-03-01", "Noise level is high but performance is solid."),
        ("Mixer Grinder", "R021", "2024-04-11", "Motor sound is a bit much but grinding is smooth."),
        ("Mixer Grinder", "R022", "2024-05-07", "Quieter than I expected, decent purchase."),
        ("Mixer Grinder", "R023", "2024-06-01", "Grinding quality is great, no complaints."),
        # Lighting
        ("Lighting", "R024", "2024-01-20", "Bulb stopped working within a month."),
        ("Lighting", "R025", "2024-02-14", "Flickering issue right out of the box."),
        ("Lighting", "R026", "2024-03-19", "Another bulb died early, second one this year."),
        ("Lighting", "R027", "2024-04-22", "Flickering got worse over time, had to replace."),
        ("Lighting", "R028", "2024-05-15", "No issues so far, good brightness."),
        ("Lighting", "R029", "2024-06-09", "Lasted longer than expected, happy with it."),
    ]
    df = pd.DataFrame(data, columns=["product", "review_id", "date", "review_text"])
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df

df = load_data()

# ==========================================
# 3. Theme Extraction (rule-based, transparent, cached)
# Real system: replace keyword matching with an LLM-based clustering pass
# run once per ingestion batch, not per chat turn.
# ==========================================
THEME_KEYWORDS = {
    "Filter/Consumable Cost": ["filter", "expensive", "replace", "cost", "overpriced", "scam"],
    "App/Connectivity Issues": ["app", "bluetooth", "connectivity", "crash", "pairing"],
    "Slow Heating": ["slow", "30 mins", "heat up", "heating"],
    "Rusting/Build Quality": ["rust", "rusting", "casing"],
    "Noise Level": ["loud", "noisy", "noise", "sound"],
    "Early Failure/Durability": ["stopped working", "died", "early", "flicker"],
}

@st.cache_data
def extract_themes(_df):
    """Tags each review with matching themes based on keywords. Returns a
    per-product, per-theme breakdown with monthly counts for trend calc."""
    rows = []
    for _, row in _df.iterrows():
        text_lower = row["review_text"].lower()
        matched = [t for t, kws in THEME_KEYWORDS.items() if any(k in text_lower for k in kws)]
        for theme in matched:
            rows.append({
                "product": row["product"],
                "theme": theme,
                "month": row["month"],
                "review_id": row["review_id"],
                "review_text": row["review_text"],
            })
    return pd.DataFrame(rows)

themed_df = extract_themes(df)

def compute_trend(monthly_counts: list) -> str:
    """Simple trend heuristic: compares first half vs second half of the
    time series average count."""
    if len(monthly_counts) < 2:
        return "insufficient data"
    mid = len(monthly_counts) // 2
    first_half_avg = sum(monthly_counts[:mid]) / mid
    second_half_avg = sum(monthly_counts[mid:]) / (len(monthly_counts) - mid)
    if second_half_avg > first_half_avg * 1.2:
        return "worsening"
    elif second_half_avg < first_half_avg * 0.8:
        return "improving"
    return "stable"

# ==========================================
# 4. Tools — return structured, grounded data (not free-text summaries)
# ==========================================
def get_product_themes(product_name: str) -> str:
    """Returns recurring complaint themes for a product, with review counts,
    trend direction, and example review IDs for citation."""
    subset = themed_df[themed_df["product"].str.lower() == product_name.lower()]
    if subset.empty:
        return f"No themed complaint data found for '{product_name}'. Do not invent themes for this product."

    lines = [f"Themes for {product_name}:"]
    for theme, group in subset.groupby("theme"):
        monthly = group.groupby("month").size().sort_index()
        trend = compute_trend(monthly.tolist())
        example_ids = group["review_id"].head(3).tolist()
        lines.append(
            f"- {theme}: {len(group)} mentions across {group['month'].nunique()} months, "
            f"trend = {trend}, example review IDs = {example_ids}"
        )
    return "\n".join(lines)

def get_review_by_id(review_id: str) -> str:
    """Returns the raw review text for a given review ID, for citation/verification."""
    match = df[df["review_id"] == review_id]
    if match.empty:
        return f"No review found with ID {review_id}."
    r = match.iloc[0]
    return f"[{r['review_id']}] ({r['date'].date()}) {r['review_text']}"

def list_products() -> str:
    """Returns the list of products with available review data."""
    return ", ".join(sorted(df["product"].unique()))

tools = [get_product_themes, get_review_by_id, list_products]

# ==========================================
# 5. Graph
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

@st.cache_resource
def get_compiled_graph(_api_key):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=_api_key)
    llm_with_tools = llm.bind_tools(tools)

    def reasoning_node(state: AgentState):
        sys_msg = SystemMessage(content="""You are an analytical assistant for Havells product managers.

RULES:
1. Only use information returned by your tools. Never state a theme, count, or trend that isn't backed by tool output.
2. When you state a theme or number, cite the supporting review IDs it came from.
3. If a product has no data, or the data doesn't support a pattern the user is asking about, say so explicitly rather than guessing.
4. Use get_review_by_id if the user wants to see the actual review text behind a theme.
5. Use list_products if you're unsure which products have data available.""")
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
# 6. Streamlit Chat Interface
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar.expander("Available products"):
    st.write(list_products())

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage) and not message.tool_calls:
        with st.chat_message("assistant"):
            st.markdown(message.content)

if prompt := st.chat_input("Ask about product reviews..."):
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st_placeholder = st.empty()
        inputs = {"messages": st.session_state.messages}

        with st.spinner("Analyzing reviews..."):
            final_response_content = ""
            for output in app.stream(inputs):
                for key, value in output.items():
                    st.session_state.messages.extend(value["messages"])
                    if key == "reasoning":
                        msg = value["messages"][0]
                        if not msg.tool_calls:
                            final_response_content = msg.content
                            st_placeholder.markdown(final_response_content)
