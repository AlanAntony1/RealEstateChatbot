import requests
import json

HEADERS = {"User-Agent": "alanantony@email.com"}  # EDGAR requires a user-agent

def get_filings(ticker="PLD"):
    # Step 1: get the CIK number for the company
    search_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2023-01-01&enddt=2024-12-31&forms=10-K,10-Q"
    
    # Use the company facts API instead - more reliable
    cik_url = "https://www.sec.gov/cgi-bin/browse-edgar?company=prologis&CIK=&type=10-K&dateb=&owner=include&count=10&search_text=&action=getcompany&output=atom"
    
    # Prologis CIK is 0001045609 - hardcoded for reliability
    cik = "0001045609"
    
    # Step 2: get company facts (financial data)
    facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(facts_url, headers=HEADERS)
    data = response.json()
    
    # Step 3: extract key metrics
    us_gaap = data.get("facts", {}).get("us-gaap", {})
    
    metrics = {}
    
    # Revenue
    for key in ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "RealEstateRevenueNet"]:
        if key in us_gaap:
            units = us_gaap[key].get("units", {}).get("USD", [])
            annual = [x for x in units if x.get("form") == "10-K"]
            if annual:
                metrics["revenue"] = annual[-3:]  # last 3 annual reports
            break
    
    # Net Income
    if "NetIncomeLoss" in us_gaap:
        units = us_gaap["NetIncomeLoss"].get("units", {}).get("USD", [])
        annual = [x for x in units if x.get("form") == "10-K"]
        metrics["net_income"] = annual[-3:]
    
    # Operating Expenses
    for key in ["OperatingExpenses", "CostsAndExpenses"]:
        if key in us_gaap:
            units = us_gaap[key].get("units", {}).get("USD", [])
            annual = [x for x in units if x.get("form") == "10-K"]
            if annual:
                metrics["operating_expenses"] = annual[-3:]
            break
    
    # Total Assets
    if "Assets" in us_gaap:
        units = us_gaap["Assets"].get("units", {}).get("USD", [])
        annual = [x for x in units if x.get("form") == "10-K"]
        metrics["total_assets"] = annual[-3:]
    
    return metrics

def format_metrics(metrics):
    print("\n=== Prologis (PLD) - SEC EDGAR Financial Data ===\n")
    for metric_name, entries in metrics.items():
        print(f"{metric_name.upper()}:")
        for entry in entries:
            val = entry.get("val", 0)
            end = entry.get("end", "")
            form = entry.get("form", "")
            print(f"  {form} ending {end}: ${val:,.0f}")
        print()

if __name__ == "__main__":
    metrics = get_filings()
    format_metrics(metrics)
    
    # Save to JSON for use in your app
    with open("edgar_data.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved to edgar_data.json")
