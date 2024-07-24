from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4")

chat_history = []

system_messages = SystemMessage(content="You are a helpul AI assistant.")
chat_history.append(system_messages)

print(chat_history)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    print(f"Bot: {response}")


print("---------- Message History ----------")
print(chat_history)