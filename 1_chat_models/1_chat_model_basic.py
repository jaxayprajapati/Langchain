from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4")

result = model.invoke("What is 81 divide by 9?")
print(result)