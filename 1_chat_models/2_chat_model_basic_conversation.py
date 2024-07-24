from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4")

"""
    SystemMessage : Large context message to the LLM and instruction to AI
    HumanMessage : Message from human to AI model
    AIMessage : LLM's response
"""

messages = [
    SystemMessage(content="Sovle the following math problem"),
    HumanMessage(content="What is 5 + 7?")
]

result = model.invoke(messages)

print("Answer from AI :", result.content)
