from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnableBranch

load_dotenv()

model = ChatOpenAI(model="gpt-4")
 
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system','You are a helpful assistant'),
        ('human','Generate a thank you note for this positive feedback: {feedback}'),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system','You are a helpful assistant'),
        ('human','Generate a response addressing this negative feedback: {feedback}'),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system','You are a helpful assistant'),
        ('human','Generate a request for more details for this neutral feedback: {feedback}'),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system','You are a helpful assistant'),
        ('human','Generate a message to escalate this feedback to a human agent : {feedback}'),
    ]
)


classification_template = ChatPromptTemplate.from_messages(
    [
        ('system','You are a helpful assistant'),
        ('human','Classify the sentiment of this feedback as positive, negative, neutral, or escalate {feedback}'),
    ]
)

branches = RunnableBranch(
    (   lambda x: 'positive' in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: 'negative' in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: 'neutral' in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
        escalate_feedback_template | model | StrOutputParser()
)

classification_chain = (
    classification_template | model | StrOutputParser()
)

chain = classification_chain | branches


good_review = "The Smartphone XYZ has an incredible battery life and a stunning display, making it perfect for all-day use."
bad_review = "The Smartphone XYZ constantly freezes and the camera quality is disappointingly poor."
neutral_review = "The Smartphone XYZ is decent for its price, with average performance and features."

result = chain.invoke({
    "feedback":bad_review
})

print(result)

