# =============================================================
# Real Estate Financial Assistant — Streamlit UI
# =============================================================

import streamlit as st
import boto3
import json
import psycopg2
import asyncio
import os
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
import vertexai

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate Financial Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid #2d3548;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #4ade80;
        font-family: 'Courier New', monospace;
    }
    
    .metric-label {
        font-size: 13px;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    
    .chat-message-user {
        background: #1e2433;
        border-left: 3px solid #4ade80;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e2e8f0;
    }
    
    .chat-message-assistant {
        background: #161b27;
        border-left: 3px solid #60a5fa;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e2e8f0;
    }
    
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #e2e8f0;
        border-bottom: 1px solid #2d3548;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    .prediction-card {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid #2d3548;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }

    .prediction-value {
        font-size: 36px;
        font-weight: 700;
        color: #4ade80;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "database": "realestate_db",
    "user": os.getenv("DB_USER", "alanantony"),
    "password": os.getenv("DB_PASSWORD", "")
}

REGRESSION_ENDPOINT     = "realestate-regression"
CLASSIFICATION_ENDPOINT = "realestate-classification"
AWS_REGION              = "us-east-1"

# ── Initialize Vertex AI ──────────────────────────────────────
@st.cache_resource
def init_vertex():
    vertexai.init(project="realestate-assistant-499113", location="us-central1")
    return True

# ── Database helpers ──────────────────────────────────────────
from mock_data import MOCK_PROPERTIES, MOCK_FINANCIAL_SUMMARY

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def query_properties(metro_area=None, property_type=None):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = """
            SELECT p.property_id, p.address, p.metro_area, p.sq_footage,
                   p.property_type, f.revenue, f.net_income, f.expenses
            FROM properties p
            JOIN financials f ON p.property_id = f.property_id
            WHERE 1=1
        """
        params = []
        if metro_area and metro_area != "All":
            query += " AND LOWER(p.metro_area) = LOWER(%s)"
            params.append(metro_area)
        if property_type and property_type != "All":
            query += " AND LOWER(p.property_type) = LOWER(%s)"
            params.append(property_type)
        query += " ORDER BY f.revenue DESC"
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception:
        # Fall back to mock data
        rows = MOCK_PROPERTIES
        if metro_area and metro_area != "All":
            rows = [r for r in rows if r[2].lower() == metro_area.lower()]
        if property_type and property_type != "All":
            rows = [r for r in rows if r[4].lower() == property_type.lower()]
        return sorted(rows, key=lambda x: x[5], reverse=True)

def get_financial_summary():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT p.metro_area,
                   COUNT(*) as count,
                   SUM(f.revenue) as total_revenue,
                   SUM(f.net_income) as total_net_income,
                   SUM(f.expenses) as total_expenses
            FROM properties p
            JOIN financials f ON p.property_id = f.property_id
            GROUP BY p.metro_area
            ORDER BY total_revenue DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception:
        return MOCK_FINANCIAL_SUMMARY
    
# ── SageMaker helpers ─────────────────────────────────────────
def predict_regression(features):
    try:
        runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
        payload = ",".join(str(f) for f in features)
        response = runtime.invoke_endpoint(
            EndpointName=REGRESSION_ENDPOINT,
            ContentType="text/csv",
            Body=payload
        )
        result = response["Body"].read().decode("utf-8")
        return float(result) * 100_000
    except Exception as e:
        return f"Error: {e}"

def predict_classification(features):
    try:
        runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
        payload = ",".join(str(f) for f in features)
        response = runtime.invoke_endpoint(
            EndpointName=CLASSIFICATION_ENDPOINT,
            ContentType="text/csv",
            Body=payload
        )
        result = response["Body"].read().decode("utf-8")
        label, prob = result.strip().split(",")
        return label.strip(), float(prob.strip())
    except Exception as e:
        return f"Error: {e}", 0.0

# ── Chatbot tools ─────────────────────────────────────────────
def chat_query_properties(metro_area: str = None, property_type: str = None) -> str:
    rows = query_properties(metro_area, property_type)
    if not rows:
        return json.dumps({"message": "No properties found"})
    results = [{"property_id": r[0], "address": r[1], "metro_area": r[2],
                "sq_footage": r[3], "property_type": r[4],
                "revenue": float(r[5]), "net_income": float(r[6]),
                "expenses": float(r[7])} for r in rows]
    return json.dumps(results, indent=2)

def chat_get_financial_summary(metric: str = "revenue") -> str:
    rows = get_financial_summary()
    if not rows:
        return json.dumps({"message": "No data found"})
    results = [{"metro_area": r[0], "count": r[1],
                "total_revenue": float(r[2]), "total_net_income": float(r[3]),
                "total_expenses": float(r[4])} for r in rows]
    return json.dumps(results, indent=2)

def chat_search_press_releases(query: str = None, release_type: str = None) -> str:
    try:
        pr_path = os.path.join(os.path.dirname(__file__), "press_releases.json")
        with open(pr_path) as f:
            press_releases = json.load(f)
        results = []
        for pr in press_releases:
            match = True
            if release_type and pr.get("type", "").lower() != release_type.lower():
                match = False
            if query:
                searchable = (pr.get("title","") + pr.get("summary","") +
                             " ".join(pr.get("tags",[]))).lower()
                if query.lower() not in searchable:
                    match = False
            if match:
                results.append(pr)
        return json.dumps(results if results else {"message": "No results"}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

def chat_get_sec_financials(metric: str = "revenue") -> str:
    try:
        edgar_path = os.path.join(os.path.dirname(__file__), "edgar_data.json")
        if os.path.exists(edgar_path):
            with open(edgar_path) as f:
                data = json.load(f)
            if metric in data:
                return json.dumps({"metric": metric, "company": "Prologis (PLD)",
                                   "source": "SEC EDGAR", "data": data[metric]}, indent=2)
        return json.dumps({"message": "EDGAR data not available"})
    except Exception as e:
        return json.dumps({"error": str(e)})

# ── Chatbot agent ─────────────────────────────────────────────
@st.cache_resource
def create_chatbot_agent():
    init_vertex()
    tools = [
        FunctionTool(chat_query_properties),
        FunctionTool(chat_get_financial_summary),
        FunctionTool(chat_search_press_releases),
        FunctionTool(chat_get_sec_financials),
    ]
    agent = Agent(
        model="gemini-2.5-flash",
        name="realestate_financial_assistant",
        description="Real estate financial assistant",
        instruction="""You are a financial assistant for a real estate company.
You have access to property data, financials, press releases, and SEC filings.
Answer questions concisely. Format dollar amounts clearly (e.g. $1.2M).
Use the appropriate tool for each query.""",
        tools=tools
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="realestate_assistant",
                    session_service=session_service)
    return runner, session_service

async def ask_chatbot(runner, session_service, user_input, session_id):
    message = Content(role="user", parts=[Part(text=user_input)])
    response_text = ""
    async for event in runner.run_async(
        user_id="user_1", session_id=session_id, new_message=message
    ):
        if event.is_final_response():
            if hasattr(event, "content") and event.content:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        response_text += part.text
    return response_text

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 Real Estate AI")
    st.markdown("---")
    page = st.radio("Navigation", [
        "💬 Chatbot",
        "🏘️ Property Data",
        "📊 Financial Summary",
        "📰 Press Releases",
        "🤖 ML Predictions"
    ])
    st.markdown("---")
    st.markdown("**Data Sources**")
    st.markdown("• PostgreSQL Database")
    st.markdown("• SEC EDGAR (Prologis)")
    st.markdown("• Press Releases")
    st.markdown("• SageMaker Endpoints")
    st.markdown("• Vertex AI ADK")

# ── Page: Chatbot ─────────────────────────────────────────────
if page == "💬 Chatbot":
    st.title("💬 Financial Assistant Chatbot")
    st.markdown("Ask questions about properties, financials, press releases, or SEC filings.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    # Initialize session
    if st.session_state.session_id is None:
        try:
            runner, session_service = create_chatbot_agent()
            loop = asyncio.new_event_loop()
            session = loop.run_until_complete(
                session_service.create_session(
                    app_name="realestate_assistant", user_id="user_1"
                )
            )
            loop.close()
            st.session_state.session_id = session.id
        except Exception as e:
            st.error(f"Failed to initialize chatbot: {e}")

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message-user">👤 {msg["content"]}</div>',
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-assistant">🤖 {msg["content"]}</div>',
                       unsafe_allow_html=True)

    # Example queries
    st.markdown("**Example queries:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏭 Industrial in Chicago"):
            st.session_state.prefill = "Show me industrial properties in Chicago"
    with col2:
        if st.button("📢 Recent acquisitions"):
            st.session_state.prefill = "Did the company announce any acquisitions recently?"
    with col3:
        if st.button("💰 Revenue summary"):
            st.session_state.prefill = "What are total revenues across all metro areas?"

    # Chat input
    prefill = st.session_state.get("prefill", "")
    user_input = st.chat_input("Ask a question...")

    if prefill and not user_input:
        user_input = prefill
        st.session_state.prefill = ""

    if user_input and st.session_state.session_id:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            try:
                runner, session_service = create_chatbot_agent()
                loop = asyncio.new_event_loop()
                response = loop.run_until_complete(
                    ask_chatbot(runner, session_service, user_input,
                               st.session_state.session_id)
                )
                loop.close()
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
                st.rerun()
            except Exception as e:
                st.error(f"Chatbot error: {e}")

# ── Page: Property Data ───────────────────────────────────────
elif page == "🏘️ Property Data":
    st.title("🏘️ Property Database")

    col1, col2 = st.columns(2)
    with col1:
        metro_filter = st.selectbox("Metro Area",
            ["All", "Chicago", "New York", "Dallas", "Seattle", "Atlanta"])
    with col2:
        type_filter = st.selectbox("Property Type",
            ["All", "Office", "Retail", "Industrial", "Residential"])

    rows = query_properties(
        metro_filter if metro_filter != "All" else None,
        type_filter if type_filter != "All" else None
    )

    if rows:
        # Summary metrics
        total_revenue = sum(float(r[5]) for r in rows)
        total_net_income = sum(float(r[6]) for r in rows)
        avg_sqft = sum(r[3] for r in rows) / len(rows)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Properties", len(rows))
        with c2:
            st.metric("Total Revenue", f"${total_revenue/1e6:.1f}M")
        with c3:
            st.metric("Total Net Income", f"${total_net_income/1e6:.1f}M")
        with c4:
            st.metric("Avg Sq Footage", f"{avg_sqft:,.0f}")

        st.markdown("---")

        # Table
        import pandas as pd
        df = pd.DataFrame(rows, columns=[
            "ID", "Address", "Metro", "Sq Ft", "Type",
            "Revenue", "Net Income", "Expenses"
        ])
        df["Revenue"] = df["Revenue"].apply(lambda x: f"${float(x):,.0f}")
        df["Net Income"] = df["Net Income"].apply(lambda x: f"${float(x):,.0f}")
        df["Expenses"] = df["Expenses"].apply(lambda x: f"${float(x):,.0f}")
        df["Sq Ft"] = df["Sq Ft"].apply(lambda x: f"{x:,}")
        st.dataframe(df.drop("ID", axis=1), use_container_width=True)
    else:
        st.info("No properties found.")

# ── Page: Financial Summary ───────────────────────────────────
elif page == "📊 Financial Summary":
    st.title("📊 Financial Summary by Metro Area")

    rows = get_financial_summary()
    if rows:
        import pandas as pd

        total_rev = sum(float(r[2]) for r in rows)
        total_inc = sum(float(r[3]) for r in rows)
        total_exp = sum(float(r[4]) for r in rows)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Portfolio Revenue", f"${total_rev/1e6:.1f}M")
        with c2:
            st.metric("Total Net Income", f"${total_inc/1e6:.1f}M")
        with c3:
            st.metric("Total Expenses", f"${total_exp/1e6:.1f}M")

        st.markdown("---")

        df = pd.DataFrame(rows, columns=[
            "Metro Area", "Properties", "Total Revenue",
            "Total Net Income", "Total Expenses"
        ])

        # Bar chart
        st.bar_chart(df.set_index("Metro Area")[["Total Revenue", "Total Net Income"]])

        # Formatted table
        df["Total Revenue"] = df["Total Revenue"].apply(lambda x: f"${float(x)/1e6:.2f}M")
        df["Total Net Income"] = df["Total Net Income"].apply(lambda x: f"${float(x)/1e6:.2f}M")
        df["Total Expenses"] = df["Total Expenses"].apply(lambda x: f"${float(x)/1e6:.2f}M")
        st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.markdown("### SEC EDGAR — Prologis (PLD)")

    try:
        edgar_path = os.path.join(os.path.dirname(__file__), "edgar_data.json")
        if os.path.exists(edgar_path):
            with open(edgar_path) as f:
                edgar_data = json.load(f)
            for metric, entries in edgar_data.items():
                if isinstance(entries, list) and entries:
                    st.markdown(f"**{metric.replace('_', ' ').title()}**")
                    cols = st.columns(min(len(entries), 3))
                    for i, entry in enumerate(entries[-3:]):
                        with cols[i]:
                            val = entry.get("val", 0)
                            end = entry.get("end", "")
                            form = entry.get("form", "")
                            st.metric(f"{form} {end}", f"${val/1e9:.2f}B")
        else:
            st.info("Run sec_edgar.py first to generate edgar_data.json")
    except Exception as e:
        st.error(f"Error loading EDGAR data: {e}")

# ── Page: Press Releases ──────────────────────────────────────
elif page == "📰 Press Releases":
    st.title("📰 Press Releases")

    try:
        pr_path = os.path.join(os.path.dirname(__file__), "press_releases.json")
        with open(pr_path) as f:
            press_releases = json.load(f)

        type_filter = st.selectbox("Filter by type",
            ["All", "earnings", "acquisition", "expansion",
             "merger", "financing", "technology"])

        filtered = press_releases if type_filter == "All" else \
            [pr for pr in press_releases if pr.get("type") == type_filter]

        for pr in filtered:
            pr_type = pr.get("type", "").upper()
            color = {"earnings": "#4ade80", "acquisition": "#60a5fa",
                    "expansion": "#f59e0b", "merger": "#a78bfa",
                    "financing": "#34d399", "technology": "#fb7185"}.get(
                        pr.get("type", ""), "#8892a4")

            with st.expander(f"**{pr['title']}** — {pr['date']}"):
                st.markdown(f"**Type:** :{pr.get('type', 'unknown')}:")
                st.markdown(f"**Summary:** {pr['summary']}")
                st.markdown("**Highlights:**")
                for h in pr.get("highlights", []):
                    st.markdown(f"• {h}")
                st.markdown(f"**Tags:** {', '.join(pr.get('tags', []))}")
    except Exception as e:
        st.error(f"Error loading press releases: {e}")

# ── Page: ML Predictions ──────────────────────────────────────
elif page == "🤖 ML Predictions":
    st.title("🤖 ML Predictions — SageMaker Endpoints")

    tab1, tab2 = st.tabs(["🏠 Housing Value (Regression)", "📊 Subscription (Classification)"])

    with tab1:
        st.markdown("### California Housing Value Predictor")
        st.markdown("Predict median house value using Random Forest Regressor")

        col1, col2 = st.columns(2)
        with col1:
            med_inc     = st.slider("Median Income ($10k)", 0.5, 15.0, 5.0, 0.1)
            house_age   = st.slider("House Age (years)", 1, 52, 20)
            ave_rooms   = st.slider("Avg Rooms", 1.0, 15.0, 6.0, 0.1)
            ave_bedrms  = st.slider("Avg Bedrooms", 0.5, 5.0, 1.0, 0.1)
        with col2:
            population  = st.slider("Population", 3, 35682, 1000)
            ave_occup   = st.slider("Avg Occupancy", 1.0, 20.0, 3.0, 0.1)
            latitude    = st.slider("Latitude", 32.0, 42.0, 37.0, 0.1)
            longitude   = st.slider("Longitude", -124.0, -114.0, -120.0, 0.1)

        if st.button("🔮 Predict Housing Value", type="primary"):
            features = [med_inc, house_age, ave_rooms, ave_bedrms,
                       population, ave_occup, latitude, longitude]
            with st.spinner("Calling SageMaker endpoint..."):
                result = predict_regression(features)
            if isinstance(result, float):
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="metric-label">Predicted Median House Value</div>
                    <div class="prediction-value">${result:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(result)

    with tab2:
        st.markdown("### Bank Marketing Subscription Predictor")
        st.markdown("Predict if a customer will subscribe using Logistic Regression")

        col1, col2 = st.columns(2)
        with col1:
            age         = st.slider("Age", 18, 95, 35)
            job         = st.selectbox("Job", ["management", "technician", "blue-collar",
                                               "admin.", "services", "retired",
                                               "self-employed", "entrepreneur",
                                               "housemaid", "student", "unemployed"])
            marital     = st.selectbox("Marital Status", ["married", "single", "divorced"])
            education   = st.selectbox("Education", ["university.degree", "high.school",
                                                     "basic.9y", "basic.6y",
                                                     "professional.course",
                                                     "basic.4y", "illiterate"])
            default     = st.selectbox("Has Credit Default?", ["no", "yes", "unknown"])
            housing     = st.selectbox("Has Housing Loan?", ["yes", "no", "unknown"])
            loan        = st.selectbox("Has Personal Loan?", ["no", "yes", "unknown"])
        with col2:
            contact     = st.selectbox("Contact Type", ["cellular", "telephone"])
            month       = st.selectbox("Last Contact Month",
                                      ["may", "jul", "aug", "jun", "nov",
                                       "apr", "oct", "sep", "mar", "dec"])
            dow         = st.selectbox("Day of Week",
                                      ["mon", "tue", "wed", "thu", "fri"])
            duration    = st.slider("Call Duration (seconds)", 0, 5000, 200)
            campaign    = st.slider("Contacts This Campaign", 1, 50, 2)
            pdays       = st.slider("Days Since Last Contact", 0, 999, 999)
            previous    = st.slider("Previous Contacts", 0, 10, 0)
            poutcome    = st.selectbox("Previous Outcome",
                                      ["nonexistent", "failure", "success"])

        if st.button("🔮 Predict Subscription", type="primary"):
            features = [
                age, job, marital, education, default, housing, loan,
                contact, month, dow, duration, campaign, pdays, previous,
                poutcome, -1.8, 92.893, -46.2, 1.299, 5099.1
            ]
            with st.spinner("Calling SageMaker endpoint..."):
                label, prob = predict_classification(features)

            if isinstance(label, str) and label in ["yes", "no"]:
                color = "#4ade80" if label == "yes" else "#fb7185"
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="metric-label">Subscription Prediction</div>
                    <div class="prediction-value" style="color:{color}">
                        {label.upper()}
                    </div>
                    <div class="metric-label" style="margin-top:8px">
                        Probability: {prob:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(label)
