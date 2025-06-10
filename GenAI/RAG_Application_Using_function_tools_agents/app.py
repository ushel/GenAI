from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from tools import retrieve_docs

llm = ChatOpenAI(temperature=0)
tools = [retrieve_docs]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

query = "What are the key contributions of paper1 and paper2?"
response = agent.run(query)
print("🧠 Response:", response)
