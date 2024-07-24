from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatOpenAI(model="gpt-4")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer"),
        ("human", "List the main features of the product {product_name}.")
    ]
)

print("Prompt Template: ", prompt_template) 


# Define pros analysis step
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", f"Given these features: {features}, list the pros of these features")
        ]
    )
    print("pros Template: ", pros_template)
    return pros_template.format_prompt(features=features)


# Define cons analysis step
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", f"Given these features: {features}, list the cons of these features")
        ]
    )
    print("cons Template: ", cons_template)
    return cons_template.format_prompt(features=features)


# Conbime pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f'Pros:\n{pros}\n\nCons:\n{cons}'


# Simplify branches with LCEL
pros_branch_chain = (
    RunnableLambda(lambda x:analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x:analyze_cons(x)) | model | StrOutputParser()
)

print("pros_branch_chain", pros_branch_chain)
print("cons_branch_chain", cons_branch_chain)

# Create the combined chain using LCEL

# chain  = prompt_template | model | StrOutputParser()
chain = (
    prompt_template
    |model
    |StrOutputParser()
    |RunnableParallel(
        branches={
            "pros":pros_branch_chain,
            "cons":cons_branch_chain
        }
    )
    |RunnableLambda(lambda x:print("Final Output", x) or combine_pros_cons(x['branches']['pros'], x['branches']['cons']))
)

result = chain.invoke({"product_name":"MacBook Pro"})
print(result)
