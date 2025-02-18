from autogen import (
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    config_list_from_json,
    GroupChatManager,
)


llm_config = {"config_list": config_list_from_json("OAI_CONFIG_LIST")}
# Create agents

user_proxy = UserProxyAgent(name="User")

assistant = AssistantAgent(name="Assistant", llm_config=llm_config)

researcher = AssistantAgent(name="Researcher", llm_config=llm_config)

writer = AssistantAgent(name="Writer", llm_config=llm_config)

# Create a group chat

agents = [user_proxy, assistant, researcher, writer]

groupchat = GroupChat(agents=agents, messages=[], max_round=12)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
# Start the conversation

user_proxy.initiate_chat(
    manager, message="Write a comprehensive report on the impact of AI on job markets."
)
