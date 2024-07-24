from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# Prompt with single placeholder
template = "Tell me a joke about {topic}"

prompt_template = ChatPromptTemplate.from_template(template)

print(prompt_template)
# print(prompt_template.input_variables)
# print(prompt_template.input_schema)
# print(prompt_template.input_types)

print("---- Prompt from Template ----")
prompt = prompt_template.invoke({"topic":"cats"})
print(prompt)


# Prompt with multiple placeholder
template_multiple = """
    You are a helpful assistant.
    Human: Tell me a {adjective} story about {animal}.
    Bot: 
"""
prompt_template_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt_multiple = prompt_template_multiple.invoke({"adjective":"funny", "animal":"panda"})
print("---- Prompt from Template (multiple variables) ----")
print(prompt_multiple)


# prompt with AIMessage and HumanMessage
messages =[
    ("system","You are a comedian who tells jokes about {topic}"),
    ("human", "Tell me {joke_count} jokes.")
]

prompt_with_ai_and_human_messages_template = ChatPromptTemplate.from_messages(messages)
prompt_with_ai_and_human_messages = prompt_with_ai_and_human_messages_template.invoke({
    "topic":"lawyers",
    "joke_count":3
})

print("---- Prompt from Template (with AI and Human messages) ----")
print(prompt_with_ai_and_human_messages)