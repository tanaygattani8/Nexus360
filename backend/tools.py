import os
from dotenv import load_dotenv
from simple_salesforce import Salesforce
from langchain_core.tools import tool

load_dotenv()

sf = Salesforce(
    username=os.getenv("SALESFORCE_USERNAME"),
    password=os.getenv("SALESFORCE_PASSWORD"),
    security_token=os.getenv("SALESFORCE_SECURITY_TOKEN"),
    domain=os.getenv("SALESFORCE_DOMAIN")
)

@tool
def get_account_health(account_name: str) -> str:
    """ Get the health summary of a customer account including open cases and opportunity status.
    Use this when asked about a customer's situation, staatus, or health."""

    # Finding account by name
    result = sf.query(f"""
        SELECT Id, Name, Type, Industry, AnnualRevenue, Phone
        FROM Account
        WHERE Name LIKE '{account_name}'
        LIMIT 1
    """)
    
    if result["totalSize"] == 0:
        return f"No account found matching '{account_name}'"

    acc = result["records"][0]
    account_id = acc["Id"]

    # Get open cases
    cases = sf.query(f"""
        SELECT Id, Subject, Status, Priority
        FROM Case
        WHERE AccountId = '{account_id}'
        AND Status != 'Closed'
    """)
    
    # Get opportunities 
    opps = sf.query(f"""
        SELECT Id, Name, StageName, Amount, CloseDate
        FROM Opportunity
        WHERE AccountId = '{account_id}'
    """)

    # Build summary
    summary = f""" 
    ACCOUNT: {acc['Name']} 
    Industry: {acc['Industry']} 
    Annual Revenue: ${acc['AnnualRevenue']:,.0f} 
    Phone: {acc['Phone']} 
    OPEN CASES ({cases['totalSize']}):
    """
    for case in cases['records']:
        summary += f"\n [{case['Priority']}] {case['Subject']} - {case['Status']}"

    if cases["totalSize"] == 0:
        summary += "\n None"

    summary += f"\n\nOPPORTUNITIES ({opps['totalSize']}):"
    for opp in opps['records']:
        summary += f"\n {opp['Name']} - {opp['StageName']} - ${opp['Amount']:,.0f} - closes {opp['CloseDate']}"
 
    if opps["totalSize"] == 0:
        summary += "\n None"

    return summary
    

@tool
def list_open_cases(account_name: str) -> str:
    """ List all open suppoert cases for a customer account. 
    Use this when asked about open support issues, problems or tickets for aspecific customer."""

    account = sf.query(f"""
        SELECT Id, Name FROM Account
        WHERE name LIKE '%{account_name}%'
        LIMIT 1
        """)
    
    if account['totalSize'] == 0:
        return f"No Account Found Matching '{account_name}'"

    account_id = account['records'][0]['Id']

    cases = sf.query(f"""
        SELECT Id, Subject, Status, Priority, Description, CreatedDate
        FROM Case 
        WHERE AccountId ='{account_id}'
        AND Status != 'Closed'
        ORDER BY CreatedDate DESC
    """)

    if cases["totalSize"] == 0:
        return f"No open cases found for {account_name}."

    output = f"Open cases for {account['records'][0]['Name']}:\n"
    for case in cases['records']:
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

    valid_stages = ["Prospecting","Qualification", "Proposal","Negotiation","Closed Won","Closed Lost"]

    if new_stage not in valid_stages:
        return f"Invalid stage '{new_stage}'. Must be one of: {', '.join(valid_stages)}"

    opp = sf.query(f"""
        SELECT Id, Name, StageName, Amount, AccountId
        FROM Opportunity
        WHERE Name LIKE '%{opportunity_name}%'
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
        Amount: ${record['Amount']:,.0f}
    """
  
@tool 
def create_support_case(account_name: str, subject: str, description: str, priority: str = "Medium") -> str:
    """ Create a new support case for a customer account.
    Priority must be Low, Medium, or High.
    Use this when a customer reports a new problem that needsd to be tracked."""

    valid_priorities = ["Low", "Medium", "High"]
    if priority not in valid_priorities:
        return f"Invalid priority '{priority}'. Must be one of: {', '.join(valid_priorities)}"
    
    account = sf.query(f"""
        SELECT Id, Name FROM Account
        WHERE Name LIKE '%{account_name}%'
        LIMIT 1
    """)

    if account['totalSize'] == 0:
        return f"No account found matching '{account_name}'"

    account_id = account["records"][0]["Id"]

    result = sf.Case.create({
        "AccountId": account_id,
        "Subject": subject,
        "Description": description,
        "Priority": priority,
        "Status": "New"
    })

    return f"""
    New case created: 
        Case ID: {result['id']}
        Account: {account['records'][0]['Name']}
        Subject: {subject}
        Priority: {priority}
        Status: New
    """
    