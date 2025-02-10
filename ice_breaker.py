from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

remedio = "OCYLIN"

if __name__ == "__main__":
    load_dotenv()

    summary_template = """
    me de infomações sobre o remedio {remedio} se não ouver certeza das informações, informar que não sabe:
    1. contra indicações
    """

    summary_promp_template = PromptTemplate(
        input_variables=["remedio"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    # llm = ChatOllama(model="deepseek-r1", temperature=0)
    # llm = ChatOllama(model="llama3.1", temperature=0)

    chain = summary_promp_template | llm | StrOutputParser()
    res = chain.invoke(input={"remedio": remedio})

    print(res)
