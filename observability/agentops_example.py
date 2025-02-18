import json
import os
import agentops
from agentops import ActionEvent
from autogen import ConversableAgent, config_list_from_json
from dotenv import load_dotenv

load_dotenv()


def calculator(a: int, b: int, operator: str) -> int:
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return int(a / b)
    else:
        raise ValueError("Invalid operator")


session = agentops.init(api_key=os.getenv("AGENTOPS_API_KEY"))
# Initialize AgentOps session
session.record(ActionEvent("llms"))
session.record(ActionEvent("tools"))

# Configure the assistant agent
assistant = ConversableAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant.",
    llm_config={"config_list": config_list_from_json("OAI_CONFIG_LIST")},
)

# Register the calculator tool
assistant.register_for_llm(name="calculator", description="Performs basic arithmetic")(
    calculator
)

# Use the assistant
response = assistant.generate_reply(
    messages=[{"content": "What is 5+3?", "role": "user"}]
)
print(json.dumps(response, indent=4))

# End AgentOps session
session.end_session(end_state="Success")
