from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

def create_agent(retrieve_tool,memory = None):
    llm = ChatOpenAI(temperature=0)
    return initialize_agent(
        tools=[retrieve_tool],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        verbose=True
    )
