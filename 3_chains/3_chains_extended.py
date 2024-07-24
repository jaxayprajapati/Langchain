from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

load_dotenv()

model = ChatOpenAI(model="gpt-4")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("user", "Tell me {joke_count} jokes")
    ]
)

print("Prompt Template: ", prompt_template)

# Define additional process step using runnable
uppercase_ouput = RunnableLambda(lambda x:x.upper())
count_words = RunnableLambda(lambda x: f"Word Count: {len(x.split())}\n{x}")


# Create chain
chain = prompt_template | model | StrOutputParser() | uppercase_ouput | count_words

# Run the chain
response = chain.invoke({
    "topic":"lawyers", "joke_count":3
})

print(response)