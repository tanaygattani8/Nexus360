from tools import (get_account_health, list_open_cases, 
                    update_opportunity_stage, create_support_case)

print("=== Testing Account Health ===")
print(get_account_health.invoke({"account_name": "Acme Corp"}))

print("\n=== Testing List Open Cases ===")
print(list_open_cases.invoke({"account_name": "Umbrella Co"}))

print("\n=== Testing Update Opportunity Stage ===")
print(update_opportunity_stage.invoke({"opportunity_name": "Initech", "new_stage": "Proposal"}))

print("\n=== Testing Create Support Case ===")
print(create_support_case.invoke({"account_name": "Pied Piper", "subject": "Cannot access analytics dashboard", 
                               "description": "Getting a 403 error when navigating to the analytics section.", "priority": "High"}))