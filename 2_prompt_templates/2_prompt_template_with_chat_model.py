from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4")

template = "Tell me joke about {animal}"
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({'animal':'cat'})
print(f"Single Placeholder Prompt: {prompt}")
# result = model.invoke(prompt)
# print(result.content)

# Multiple Placeholder Prompt
messages = [
    ("system","You are an help full {subject} tutor and you task is to help me understand the topic"),
    ("user", "Explain me {concept} of {subject}")
]

multiple_placeholder_prompt_template = ChatPromptTemplate.from_messages(messages)
# 1 way
multiple_placeholder_prompt = multiple_placeholder_prompt_template.invoke({
    "subject":"Artificial Intelligence",
    "concept":"Large language Models"
    })

# 2 way
multiple_placeholder_prompt_01 = multiple_placeholder_prompt_template.format_prompt(subject="Data Science", concept="Hypothesis testing")
print(f"Multiple Placeholder Prompt: {multiple_placeholder_prompt}")
print(f"Multiple Placeholder Prompt_01: {multiple_placeholder_prompt_01}")

# multiple_placeholder_result = model.invoke(multiple_placeholder_prompt)
# print(multiple_placeholder_result.content)