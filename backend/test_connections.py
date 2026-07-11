import os
from dotenv import load_dotenv
from simple_salesforce import Salesforce
from groq import Groq

load_dotenv()
# Testing Salesforce
print("Testing Salesforce connection...")
sf = Salesforce(
    username=os.getenv("SALESFORCE_USERNAME"),
    password=os.getenv("SALESFORCE_PASSWORD"),
    security_token=os.getenv("SALESFORCE_SECURITY_TOKEN"),
    domain=os.getenv("SALESFORCE_DOMAIN")
)
result = sf.query("SELECT Id, Name, Type FROM Account LIMIT 5")
print(f"✓ Salesforce connected. Found {result['totalSize']} accounts.")

# Test Groq
print("\nTesting Groq connection...")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)
print(f"✓ Groq connected. Response: {response.choices[0].message.content}")