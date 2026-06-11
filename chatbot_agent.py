# =============================================================
# Vertex AI ADK Chatbot Agent
# Real Estate Financial Assistant
# =============================================================

import os
import json
import psycopg2
import requests
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
import vertexai
vertexai.init(project="realestate-assistant-499113", location="us-central1")
# ── Database config ───────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "database": "realestate_db",
    "user": "alanantony",
    "password": ""
}

# ── Tool 1: Query Properties ──────────────────────────────────
def query_properties(metro_area: str = None, property_type: str = None) -> str:
    """
    Query the properties and financials database.
    Args:
        metro_area: Filter by city e.g. 'Chicago', 'New York', 'Dallas'
        property_type: Filter by type e.g. 'Office', 'Retail', 'Industrial', 'Residential'
    Returns:
        JSON string of matching properties with financial data
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        query = """
            SELECT p.property_id, p.address, p.metro_area, p.sq_footage,
                   p.property_type, f.revenue, f.net_income, f.expenses
            FROM properties p
            JOIN financials f ON p.property_id = f.property_id
            WHERE 1=1
        """
        params = []

        if metro_area:
            query += " AND LOWER(p.metro_area) = LOWER(%s)"
            params.append(metro_area)

        if property_type:
            query += " AND LOWER(p.property_type) = LOWER(%s)"
            params.append(property_type)

        query += " ORDER BY f.revenue DESC LIMIT 10"

        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return json.dumps({"message": "No properties found matching criteria"})

        results = []
        for row in rows:
            results.append({
                "property_id": row[0],
                "address": row[1],
                "metro_area": row[2],
                "sq_footage": row[3],
                "property_type": row[4],
                "revenue": float(row[5]),
                "net_income": float(row[6]),
                "expenses": float(row[7])
            })

        return json.dumps(results, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Tool 2: Get Financial Summary ─────────────────────────────
def get_financial_summary(metric: str = "revenue") -> str:
    """
    Get aggregated financial summary across all properties.
    Args:
        metric: One of 'revenue', 'net_income', 'expenses'
    Returns:
        JSON string with financial summary by metro area
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        valid_metrics = ["revenue", "net_income", "expenses"]
        if metric not in valid_metrics:
            metric = "revenue"

        query = f"""
            SELECT p.metro_area,
                   COUNT(*) as property_count,
                   SUM(f.{metric}) as total_{metric},
                   AVG(f.{metric}) as avg_{metric},
                   MAX(f.{metric}) as max_{metric}
            FROM properties p
            JOIN financials f ON p.property_id = f.property_id
            GROUP BY p.metro_area
            ORDER BY total_{metric} DESC
        """

        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        results = []
        for row in rows:
            results.append({
                "metro_area": row[0],
                "property_count": row[1],
                f"total_{metric}": float(row[2]),
                f"avg_{metric}": float(row[3]),
                f"max_{metric}": float(row[4])
            })

        return json.dumps(results, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Tool 3: Search Press Releases ─────────────────────────────
def search_press_releases(query: str = None, release_type: str = None) -> str:
    """
    Search press releases for company news and announcements.
    Args:
        query: Keyword to search for e.g. 'acquisition', 'earnings', 'Chicago'
        release_type: Filter by type e.g. 'acquisition', 'earnings', 'expansion', 'merger'
    Returns:
        JSON string of matching press releases
    """
    try:
        # Load press releases from JSON file
        pr_path = os.path.join(os.path.dirname(__file__), "press_releases.json")
        with open(pr_path, "r") as f:
            press_releases = json.load(f)

        results = []
        for pr in press_releases:
            match = True

            if release_type and pr.get("type", "").lower() != release_type.lower():
                match = False

            if query:
                query_lower = query.lower()
                searchable = (
                    pr.get("title", "") +
                    pr.get("summary", "") +
                    " ".join(pr.get("tags", []))
                ).lower()
                if query_lower not in searchable:
                    match = False

            if match:
                results.append(pr)

        if not results:
            return json.dumps({"message": "No press releases found matching criteria"})

        return json.dumps(results, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})

# ── Tool 4: Get SEC EDGAR Financials ──────────────────────────
def get_sec_financials(metric: str = "revenue") -> str:
    """
    Get Prologis SEC EDGAR financial data.
    Args:
        metric: One of 'revenue', 'net_income', 'operating_expenses', 'total_assets'
    Returns:
        JSON string with historical financial data from SEC filings
    """
    try:
        edgar_path = os.path.join(os.path.dirname(__file__), "edgar_data.json")

        if os.path.exists(edgar_path):
            with open(edgar_path, "r") as f:
                data = json.load(f)

            if metric in data:
                return json.dumps({
                    "metric": metric,
                    "source": "SEC EDGAR",
                    "company": "Prologis (PLD)",
                    "data": data[metric]
                }, indent=2)
            else:
                return json.dumps({
                    "available_metrics": list(data.keys()),
                    "message": f"Metric '{metric}' not found"
                })
        else:
            # Fetch live from EDGAR if file doesn't exist
            CIK = "0001045609"
            facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"
            headers = {"User-Agent": "realestate-assistant@email.com"}
            response = requests.get(facts_url, headers=headers, timeout=10)
            data = response.json()

            us_gaap = data.get("facts", {}).get("us-gaap", {})
            revenue_keys = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]

            for key in revenue_keys:
                if key in us_gaap:
                    units = us_gaap[key].get("units", {}).get("USD", [])
                    annual = [x for x in units if x.get("form") == "10-K"][-3:]
                    return json.dumps({
                        "metric": "revenue",
                        "company": "Prologis (PLD)",
                        "source": "SEC EDGAR Live",
                        "data": annual
                    }, indent=2)

            return json.dumps({"message": "Could not fetch EDGAR data"})

    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Build Agent ───────────────────────────────────────────────
def create_agent():
    tools = [
        FunctionTool(query_properties),
        FunctionTool(get_financial_summary),
        FunctionTool(search_press_releases),
        FunctionTool(get_sec_financials),
    ]

    agent = Agent(
        model="gemini-2.5-flash",
        name="realestate_financial_assistant",
        description="A financial assistant for a real estate company that can query property data, financial metrics, press releases, and SEC filings.",
        instruction="""You are a financial assistant for a real estate company.
        
You have access to:
1. A property database with financial data (revenue, expenses, net income) across Chicago, New York, Dallas, Seattle, and Atlanta
2. Press releases covering acquisitions, earnings, expansions, and other company news
3. SEC EDGAR financial filings for Prologis (PLD)

When answering questions:
- For property queries, use query_properties tool
- For financial summaries across regions, use get_financial_summary tool  
- For news, acquisitions, or announcements, use search_press_releases tool
- For SEC filing data, use get_sec_financials tool
- Combine multiple tools when needed for comprehensive answers
- Always provide dollar amounts in a readable format (e.g. $1.2M, $450K)
- Be concise but thorough in your responses

Example queries you can handle:
- "What was the net income reported last quarter?"
- "Show industrial properties in Chicago with revenue details"
- "Did the company announce any acquisitions recently?"
- "What are the total revenues across all metro areas?"
- "What is the highest revenue property?"
""",
        tools=tools
    )
    return agent


# ── Run chatbot interactively ─────────────────────────────────
async def run_chatbot():
    import asyncio
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    agent = create_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="realestate_assistant",
        session_service=session_service
    )

    session = await session_service.create_session(
        app_name="realestate_assistant",
        user_id="user_1"
    )

    print("=" * 50)
    print("Real Estate Financial Assistant")
    print("Type 'quit' to exit")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        if not user_input:
            continue

        message = Content(role="user", parts=[Part(text=user_input)])

        async for event in runner.run_async(
            user_id="user_1",
            session_id=session.id,
            new_message=message
        ):
            if event.is_final_response():
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            print(f"\nAssistant: {part.text}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_chatbot())
