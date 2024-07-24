from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence

load_dotenv()

model = ChatOpenAI(model="gpt-4")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("user", "Tell me {joke_count} jokes")
    ]
)

print("Prompt Template: ", prompt_template)

# Create individual runnable (task) (step in chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
response = chain.invoke({
    "topic":"lawyers", "joke_count":3
})

print(response)