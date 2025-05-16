from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="dolphin-mistral")

template = """
You are an expert in answering questions about movies. Answer the following question:

Here are some relevant reviews: {reviews}

Here are some questions to answer: {questions}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n----------------------------------------------------")
    question = input("Enter a question (or 'q' to quit): ")
    print("\n\n")

    if question == 'q':
        break

    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": "This movie was fantastic! The acting was top-notch and the plot was gripping. I couldn't take my eyes off the screen.",
              "questions": question})

    print(result)
