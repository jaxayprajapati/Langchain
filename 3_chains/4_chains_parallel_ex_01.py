from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser


load_dotenv()

model = ChatOpenAI(model="gpt-4")

prompt_template = ChatPromptTemplate.from_messages([
    ('system',"You are a maths teacher"),
    ('user', 'Your task is to calculate the sum of {number_1} and {number_2}')
])


def is_prime(number):
    prime_template = ChatPromptTemplate.from_messages([
        ('system','You are a maths teacher'),
        ('user', f'Determine if the number {number} is prime or not and do not provide explaination')
    ])
    return prime_template.format_prompt(number=number)


def is_natural_number(number):
    natural_template = ChatPromptTemplate.from_messages([
        ('system','You are a maths teacher'),
        ('user', f'Determine if the number {number} is a natural number or not and do not provide explaination')
    ])
    return natural_template.format_prompt(number=number)


prime_chain = (
    RunnableLambda(lambda x: is_prime(x)) | model | StrOutputParser()
)

is_natural_chain = (
    RunnableLambda(lambda x: is_natural_number(x)) | model | StrOutputParser()
)


def get_final_result(x, y):
    return (x, y)

final_chain = (
    prompt_template
    |model
    |StrOutputParser()
    |RunnableParallel(
        branches={
            "is_prime": prime_chain,
            "is_natural": is_natural_chain
        }
    )
    |RunnableLambda(lambda x: get_final_result(x['branches']['is_prime'], x['branches']['is_natural']))
)

result = final_chain.invoke({
    "number_1":10,
    "number_2":20
})

print(result)