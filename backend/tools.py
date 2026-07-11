import os
from datetime import date, timedelta
from dotenv import load_dotenv
from simple_salesforce import Salesforce
from langchain_core.tools import tool

load_dotenv()

# ── Salesforce connection ─────────────────────────────────────────────────────
# Fault-tolerant: if creds are missing, wrong, or the dev org has expired,
# fall back to built-in mock data so the demo never dies mid-presentation.
# Force mock mode with MOCK_SF=true in .env.

def _connect() -> Salesforce | None:
    if os.getenv("MOCK_SF", "").lower() == "true":
        print("[salesforce] MOCK_SF=true — using built-in mock Salesforce data")
        return None
    try:
        return Salesforce(
            username=os.getenv("SALESFORCE_USERNAME"),
            password=os.getenv("SALESFORCE_PASSWORD"),
            security_token=os.getenv("SALESFORCE_SECURITY_TOKEN"),
            domain=os.getenv("SALESFORCE_DOMAIN"),
        )
    except Exception as e:
        print(f"[salesforce] Unavailable ({e}) — falling back to mock data")
        return None

sf = _connect()
SF_MODE = "live" if sf else "mock"

# Single source of truth for opportunity stages — the tool docstring,
# the agent prompt, and the knowledge base doc all reference this list.
VALID_STAGES = ["Prospecting", "Qualification", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
VALID_PRIORITIES = ["Low", "Medium", "High"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _soql_escape(value: str) -> str:
    """Escape quotes and backslashes so user text can't break out of a SOQL string literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _money(value) -> str:
    """Format a currency amount, tolerating null fields from Salesforce."""
    return f"${value:,.0f}" if value is not None else "N/A"


# ── Mock data (mirrors seed_data.py) ─────────────────────────────────────────

_today = date.today()
_MOCK = {
    "accounts": [
        {"Id": "MOCK001", "Name": "Acme Corp",        "Industry": "Technology",    "AnnualRevenue": 5000000,  "Phone": "415-555-0101"},
        {"Id": "MOCK002", "Name": "Globex Inc",       "Industry": "Manufacturing", "AnnualRevenue": 12000000, "Phone": "415-555-0102"},
        {"Id": "MOCK003", "Name": "Initech Ltd",      "Industry": "Finance",       "AnnualRevenue": 3000000,  "Phone": "415-555-0103"},
        {"Id": "MOCK004", "Name": "Umbrella Co",      "Industry": "Healthcare",    "AnnualRevenue": 8000000,  "Phone": "415-555-0104"},
        {"Id": "MOCK005", "Name": "Stark Industries", "Industry": "Technology",    "AnnualRevenue": 50000000, "Phone": "415-555-0105"},
        {"Id": "MOCK006", "Name": "Dunder Mifflin",   "Industry": "Retail",        "AnnualRevenue": 1500000,  "Phone": "415-555-0106"},
        {"Id": "MOCK007", "Name": "Hooli",            "Industry": "Technology",    "AnnualRevenue": 20000000, "Phone": "415-555-0107"},
        {"Id": "MOCK008", "Name": "Pied Piper",       "Industry": "Technology",    "AnnualRevenue": 500000,   "Phone": "415-555-0108"},
    ],
    "cases": [
        {"Id": "MCASE01", "AccountId": "MOCK001", "Subject": "Login issues after update",    "Status": "New",         "Priority": "High",   "Description": "Users cannot log in since the latest platform update deployed yesterday.", "CreatedDate": str(_today)},
        {"Id": "MCASE02", "AccountId": "MOCK001", "Subject": "Data export failing silently", "Status": "In Progress", "Priority": "High",   "Description": "CSV export returns 200 OK but the file is empty. Affecting all users.",    "CreatedDate": str(_today)},
        {"Id": "MCASE03", "AccountId": "MOCK002", "Subject": "Slow dashboard load times",    "Status": "New",         "Priority": "Medium", "Description": "Dashboard takes 45+ seconds to load. Was fine last week.",                 "CreatedDate": str(_today)},
        {"Id": "MCASE04", "AccountId": "MOCK003", "Subject": "Billing discrepancy Q4",       "Status": "New",         "Priority": "Medium", "Description": "Invoice amount does not match the contract terms signed in October.",      "CreatedDate": str(_today)},
        {"Id": "MCASE05", "AccountId": "MOCK004", "Subject": "API rate limit errors",        "Status": "In Progress", "Priority": "High",   "Description": "Getting 429 errors from the API despite being within our plan limits.",    "CreatedDate": str(_today)},
        {"Id": "MCASE06", "AccountId": "MOCK005", "Subject": "SSO configuration broken",     "Status": "New",         "Priority": "High",   "Description": "SSO stopped working after their IT team updated their identity provider.", "CreatedDate": str(_today)},
        {"Id": "MCASE07", "AccountId": "MOCK006", "Subject": "Feature request: bulk import", "Status": "New",         "Priority": "Low",    "Description": "Customer wants to import 10k records at once. Current limit is 500.",      "CreatedDate": str(_today)},
        {"Id": "MCASE08", "AccountId": "MOCK008", "Subject": "Onboarding questions",         "Status": "New",         "Priority": "Low",    "Description": "New customer needs help setting up their first workspace.",                "CreatedDate": str(_today)},
    ],
    "opportunities": [
        {"Id": "MOPP01", "AccountId": "MOCK001", "Name": "Acme Corp - Enterprise License",   "StageName": "Negotiation",   "Amount": 120000, "CloseDate": str(_today + timedelta(days=25))},
        {"Id": "MOPP02", "AccountId": "MOCK002", "Name": "Globex Inc - Platform Renewal",    "StageName": "Proposal",      "Amount": 85000,  "CloseDate": str(_today + timedelta(days=25))},
        {"Id": "MOPP03", "AccountId": "MOCK003", "Name": "Initech Ltd - Starter Pack",       "StageName": "Qualification", "Amount": 15000,  "CloseDate": str(_today + timedelta(days=90))},
        {"Id": "MOPP04", "AccountId": "MOCK004", "Name": "Umbrella Co - Healthcare Suite",   "StageName": "Negotiation",   "Amount": 200000, "CloseDate": str(_today + timedelta(days=25))},
        {"Id": "MOPP05", "AccountId": "MOCK005", "Name": "Stark Industries - Full Platform", "StageName": "Proposal",      "Amount": 500000, "CloseDate": str(_today + timedelta(days=90))},
        {"Id": "MOPP06", "AccountId": "MOCK006", "Name": "Dunder Mifflin - Basic Plan",      "StageName": "Closed Won",    "Amount": 12000,  "CloseDate": str(_today - timedelta(days=10))},
        {"Id": "MOPP07", "AccountId": "MOCK007", "Name": "Hooli - Pilot Program",            "StageName": "Prospecting",   "Amount": 75000,  "CloseDate": str(_today + timedelta(days=90))},
        {"Id": "MOPP08", "AccountId": "MOCK008", "Name": "Pied Piper - Seed Deal",           "StageName": "Qualification", "Amount": 25000,  "CloseDate": str(_today + timedelta(days=90))},
    ],
}
_mock_case_counter = 100


def _mock_find_account(name: str) -> dict | None:
    n = name.lower()
    return next((a for a in _MOCK["accounts"] if n in a["Name"].lower()), None)


def _mock_find_opportunity(name: str) -> dict | None:
    n = name.lower()
    return next((o for o in _MOCK["opportunities"] if n in o["Name"].lower()), None)


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def get_account_health(account_name: str) -> str:
    """Get the health summary of a customer account including open cases and opportunity status.
    Use this when asked about a customer's situation, status, or health."""

    if sf is None:
        acc = _mock_find_account(account_name)
        if not acc:
            return f"No account found matching '{account_name}'"
        cases = {"totalSize": 0, "records": [c for c in _MOCK["cases"] if c["AccountId"] == acc["Id"] and c["Status"] != "Closed"]}
        cases["totalSize"] = len(cases["records"])
        opps = {"totalSize": 0, "records": [o for o in _MOCK["opportunities"] if o["AccountId"] == acc["Id"]]}
        opps["totalSize"] = len(opps["records"])
    else:
        safe_name = _soql_escape(account_name)
        result = sf.query(f"""
            SELECT Id, Name, Type, Industry, AnnualRevenue, Phone
            FROM Account
            WHERE Name LIKE '%{safe_name}%'
            LIMIT 1
        """)

        if result["totalSize"] == 0:
            return f"No account found matching '{account_name}'"

        acc = result["records"][0]
        account_id = acc["Id"]

        cases = sf.query(f"""
            SELECT Id, Subject, Status, Priority
            FROM Case
            WHERE AccountId = '{account_id}'
            AND Status != 'Closed'
        """)

        opps = sf.query(f"""
            SELECT Id, Name, StageName, Amount, CloseDate
            FROM Opportunity
            WHERE AccountId = '{account_id}'
        """)

    summary = f"""
    ACCOUNT: {acc['Name']}
    Industry: {acc['Industry']}
    Annual Revenue: {_money(acc['AnnualRevenue'])}
    Phone: {acc['Phone'] or 'N/A'}
    OPEN CASES ({cases['totalSize']}):
    """
    for case in cases['records']:
        summary += f"\n [{case['Priority']}] {case['Subject']} - {case['Status']}"

    if cases["totalSize"] == 0:
        summary += "\n None"

    summary += f"\n\nOPPORTUNITIES ({opps['totalSize']}):"
    for opp in opps['records']:
        summary += f"\n {opp['Name']} - {opp['StageName']} - {_money(opp['Amount'])} - closes {opp['CloseDate']}"

    if opps["totalSize"] == 0:
        summary += "\n None"

    return summary


@tool
def list_open_cases(account_name: str) -> str:
    """List all open support cases for a customer account.
    Use this when asked about open support issues, problems or tickets for a specific customer."""

    if sf is None:
        acc = _mock_find_account(account_name)
        if not acc:
            return f"No Account Found Matching '{account_name}'"
        records = [c for c in _MOCK["cases"] if c["AccountId"] == acc["Id"] and c["Status"] != "Closed"]
        account_label = acc["Name"]
    else:
        safe_name = _soql_escape(account_name)
        account = sf.query(f"""
            SELECT Id, Name FROM Account
            WHERE Name LIKE '%{safe_name}%'
            LIMIT 1
            """)

        if account['totalSize'] == 0:
            return f"No Account Found Matching '{account_name}'"

        account_id = account['records'][0]['Id']
        account_label = account['records'][0]['Name']

        cases = sf.query(f"""
            SELECT Id, Subject, Status, Priority, Description, CreatedDate
            FROM Case
            WHERE AccountId ='{account_id}'
            AND Status != 'Closed'
            ORDER BY CreatedDate DESC
        """)
        records = cases['records']

    if not records:
        return f"No open cases found for {account_label}."

    output = f"Open cases for {account_label}:\n"
    for case in records:
        output += f"""
        Case ID: {case['Id']}
        Subject: {case['Subject']}
        Status: {case['Status']}
        Priority: {case['Priority']}
        Created: {case['CreatedDate']}
        Description: {case['Description']}
        ---"""

    return output


@tool
def update_opportunity_stage(opportunity_name: str, new_stage: str) -> str:
    """Update the stage of an opportunity.
    Valid stages are: Prospecting, Qualification, Proposal, Negotiation, Closed Won, Closed Lost.
    Use this when asked to update, move, or change a deal stage."""

    if new_stage not in VALID_STAGES:
        return f"Invalid stage '{new_stage}'. Must be one of: {', '.join(VALID_STAGES)}"

    if sf is None:
        record = _mock_find_opportunity(opportunity_name)
        if not record:
            return f"No opportunity found matching '{opportunity_name}'"
        old_stage = record["StageName"]
        record["StageName"] = new_stage
    else:
        safe_name = _soql_escape(opportunity_name)
        opp = sf.query(f"""
            SELECT Id, Name, StageName, Amount, AccountId
            FROM Opportunity
            WHERE Name LIKE '%{safe_name}%'
            LIMIT 1
            """)

        if opp['totalSize'] == 0:
            return f"No opportunity found matching '{opportunity_name}'"

        record = opp["records"][0]
        old_stage = record["StageName"]

        sf.Opportunity.update(record["Id"], {"StageName": new_stage})

    return f"""
    Opportunity updated:
        Name: {record['Name']}
        Old stage: {old_stage}
        New stage: {new_stage}
        Amount: {_money(record['Amount'])}
    """


@tool
def create_support_case(account_name: str, subject: str, description: str, priority: str = "Medium") -> str:
    """Create a new support case for a customer account.
    Priority must be Low, Medium, or High.
    Use this when a customer reports a new problem that needs to be tracked."""

    if priority not in VALID_PRIORITIES:
        return f"Invalid priority '{priority}'. Must be one of: {', '.join(VALID_PRIORITIES)}"

    if sf is None:
        global _mock_case_counter
        acc = _mock_find_account(account_name)
        if not acc:
            return f"No account found matching '{account_name}'"
        _mock_case_counter += 1
        case_id = f"MCASE{_mock_case_counter}"
        _MOCK["cases"].append({
            "Id": case_id, "AccountId": acc["Id"], "Subject": subject,
            "Description": description, "Priority": priority,
            "Status": "New", "CreatedDate": str(date.today()),
        })
        account_label = acc["Name"]
    else:
        safe_name = _soql_escape(account_name)
        account = sf.query(f"""
            SELECT Id, Name FROM Account
            WHERE Name LIKE '%{safe_name}%'
            LIMIT 1
        """)

        if account['totalSize'] == 0:
            return f"No account found matching '{account_name}'"

        account_id = account["records"][0]["Id"]
        account_label = account["records"][0]["Name"]

        result = sf.Case.create({
            "AccountId": account_id,
            "Subject": subject,
            "Description": description,
            "Priority": priority,
            "Status": "New"
        })
        case_id = result['id']

    return f"""
    New case created:
        Case ID: {case_id}
        Account: {account_label}
        Subject: {subject}
        Priority: {priority}
        Status: New
    """
