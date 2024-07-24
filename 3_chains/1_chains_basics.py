from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("user", "Tell me {joke_count} jokes")
    ]
)

print("Prompt Template: ", prompt_template)

# Create the combine chain using Langchain Expressions Language
chain = prompt_template | model

# print("Chain ===>", chain.__dict__)

result = chain.invoke({"topic":"lawyers", "joke_count":3})
print("Without string parser")
print(result)
print("*"*100)

chain_01 = prompt_template | model | StrOutputParser()
result_01 = chain_01.invoke({"topic":"lawyers", "joke_count":3})
print("With string parser")
print(result_01)
print("*"*100)
