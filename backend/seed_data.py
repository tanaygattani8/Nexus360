import os
from dotenv import load_dotenv
from simple_salesforce import Salesforce

load_dotenv()

sf = Salesforce(
    username=os.getenv("SALESFORCE_USERNAME"),
    password=os.getenv("SALESFORCE_PASSWORD"),
    security_token=os.getenv("SALESFORCE_SECURITY_TOKEN"),
    domain=os.getenv("SALESFORCE_DOMAIN")
)

# ── Accounts ──────────────────────────────────────────────
accounts = [
    {"Name": "Acme Corp",        "Type": "Customer",  "Industry": "Technology",    "AnnualRevenue": 5000000,  "Phone": "415-555-0101"},
    {"Name": "Globex Inc",       "Type": "Customer",  "Industry": "Manufacturing", "AnnualRevenue": 12000000, "Phone": "415-555-0102"},
    {"Name": "Initech Ltd",      "Type": "Customer",  "Industry": "Finance",       "AnnualRevenue": 3000000,  "Phone": "415-555-0103"},
    {"Name": "Umbrella Co",      "Type": "Customer",  "Industry": "Healthcare",    "AnnualRevenue": 8000000,  "Phone": "415-555-0104"},
    {"Name": "Stark Industries", "Type": "Customer",  "Industry": "Technology",    "AnnualRevenue": 50000000, "Phone": "415-555-0105"},
    {"Name": "Dunder Mifflin",   "Type": "Customer",  "Industry": "Retail",        "AnnualRevenue": 1500000,  "Phone": "415-555-0106"},
    {"Name": "Hooli",            "Type": "Prospect",  "Industry": "Technology",    "AnnualRevenue": 20000000, "Phone": "415-555-0107"},
    {"Name": "Pied Piper",       "Type": "Prospect",  "Industry": "Technology",    "AnnualRevenue": 500000,   "Phone": "415-555-0108"},
]

print("Creating accounts...")
account_ids = {}
for acc in accounts:
    result = sf.Account.create(acc)
    account_ids[acc["Name"]] = result["id"]
    print(f"  ✓ {acc['Name']} → {result['id']}")

# ── Opportunities ──────────────────────────────────────────
import datetime
today = datetime.date.today()
close_soon = (today + datetime.timedelta(days=25)).isoformat()
close_later = (today + datetime.timedelta(days=90)).isoformat()
close_past = (today - datetime.timedelta(days=10)).isoformat()

opportunities = [
    {"Name": "Acme Corp - Enterprise License",    "AccountId": account_ids["Acme Corp"],        "StageName": "Negotiation",   "Amount": 120000,  "CloseDate": close_soon},
    {"Name": "Globex Inc - Platform Renewal",     "AccountId": account_ids["Globex Inc"],       "StageName": "Proposal",      "Amount": 85000,   "CloseDate": close_soon},
    {"Name": "Initech Ltd - Starter Pack",        "AccountId": account_ids["Initech Ltd"],      "StageName": "Qualification", "Amount": 15000,   "CloseDate": close_later},
    {"Name": "Umbrella Co - Healthcare Suite",    "AccountId": account_ids["Umbrella Co"],      "StageName": "Negotiation",   "Amount": 200000,  "CloseDate": close_soon},
    {"Name": "Stark Industries - Full Platform",  "AccountId": account_ids["Stark Industries"], "StageName": "Proposal",      "Amount": 500000,  "CloseDate": close_later},
    {"Name": "Dunder Mifflin - Basic Plan",       "AccountId": account_ids["Dunder Mifflin"],   "StageName": "Closed Won",    "Amount": 12000,   "CloseDate": close_past},
    {"Name": "Hooli - Pilot Program",             "AccountId": account_ids["Hooli"],            "StageName": "Prospecting",   "Amount": 75000,   "CloseDate": close_later},
    {"Name": "Pied Piper - Seed Deal",            "AccountId": account_ids["Pied Piper"],       "StageName": "Qualification", "Amount": 25000,   "CloseDate": close_later},
]

print("\nCreating opportunities...")
opportunity_ids = {}
for opp in opportunities:
    result = sf.Opportunity.create(opp)
    opportunity_ids[opp["Name"]] = result["id"]
    print(f"  ✓ {opp['Name']} → {result['id']}")

# ── Cases ──────────────────────────────────────────────────
cases = [
    {"Subject": "Login issues after update",        "AccountId": account_ids["Acme Corp"],        "Status": "New",         "Priority": "High",   "Description": "Users cannot log in since the latest platform update deployed yesterday."},
    {"Subject": "Data export failing silently",     "AccountId": account_ids["Acme Corp"],        "Status": "In Progress", "Priority": "High",   "Description": "CSV export returns 200 OK but the file is empty. Affecting all users."},
    {"Subject": "Slow dashboard load times",        "AccountId": account_ids["Globex Inc"],       "Status": "New",         "Priority": "Medium", "Description": "Dashboard takes 45+ seconds to load. Was fine last week."},
    {"Subject": "Billing discrepancy Q4",           "AccountId": account_ids["Initech Ltd"],      "Status": "New",         "Priority": "Medium", "Description": "Invoice amount does not match the contract terms signed in October."},
    {"Subject": "API rate limit errors",            "AccountId": account_ids["Umbrella Co"],      "Status": "In Progress", "Priority": "High",   "Description": "Getting 429 errors from the API despite being within our plan limits."},
    {"Subject": "SSO configuration broken",        "AccountId": account_ids["Stark Industries"], "Status": "New",         "Priority": "High",   "Description": "SSO stopped working after their IT team updated their identity provider."},
    {"Subject": "Feature request: bulk import",    "AccountId": account_ids["Dunder Mifflin"],   "Status": "New",         "Priority": "Low",    "Description": "Customer wants to import 10k records at once. Current limit is 500."},
    {"Subject": "Onboarding questions",            "AccountId": account_ids["Pied Piper"],        "Status": "New",         "Priority": "Low",    "Description": "New customer needs help setting up their first workspace."},
]

print("\nCreating cases...")
for case in cases:
    result = sf.Case.create(case)
    print(f"  ✓ {case['Subject']} → {result['id']}")

print("\n✓ Seed data complete.")
print(f"  {len(accounts)} accounts")
print(f"  {len(opportunities)} opportunities")
print(f"  {len(cases)} cases")