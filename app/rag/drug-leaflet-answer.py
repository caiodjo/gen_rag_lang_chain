"""
Module for creating a chain for generating responses based on the provided category,
using Elasticsearch as the knowledge base.
"""

from typing import Dict
from operator import itemgetter

from langchain.output_parsers import OutputFixingParser
from langchain_core.runnables import RunnableParallel, Runnable, RunnablePassthrough
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


from rag.utils.common import format_docs, get_docs_source
from rag.utils.parsers import AnswerOutputParser


class AnalysisFormat(BaseModel):
    analysis: str = Field(description="Texto contendo a análise da resposta")
    answer: str = Field(description="Texto da resposta")


def map_report_retriever(dict_: Dict) -> Dict:
    return {
        "query": f"{dict_['drug']} {dict_['detailed_answers']}",
    }


def get_answer_chain(
    retriever: BaseRetriever,
    model: Runnable,
    prompt: PromptTemplate,
    debug_logger=None,
) -> Runnable:

    if debug_logger is not None:
        model = model | debug_logger

    json_parser = PydanticOutputParser(pydantic_object=AnalysisFormat)
    json_fixing_parser = OutputFixingParser.from_llm(parser=json_parser, llm=model)

    def answer_parser(x):
        answers = x["answers"].split("\n")
        parser = AnswerOutputParser(answers=answers)
        x["llm_out"].answer = parser.invoke(x["llm_out"].answer)
        return x["llm_out"]

    def retrieve_documents_from_queries(dict_: dict):
        documents = []
        for query in dict_["detailed_answers"]:
            documents.extend(
                retriever.get_relevant_documents(
                    {
                        "query": f"{dict_['question']} {query}",
                        "subcategory": dict_["category"],
                    }
                )
            )
        return documents

    _input = RunnableParallel(
        {
            "drug": itemgetter("drug"),
            "answers": itemgetter("answers"),
            "user_input": itemgetter("user_input"),
            "context": retrieve_documents_from_queries,
        }
    )
    _output = (
        RunnablePassthrough.assign(report=(lambda x: format_docs(x["context"])))
        | prompt
        | model
        | json_fixing_parser
    )
    intermediate_chain = _input | RunnableParallel(
        llm_out=_output,
        documents=lambda x: get_docs_source(x["context"]),
        answers=itemgetter("answers"),
    )

    final_chain = intermediate_chain | RunnableParallel(
        {
            "llm_out": answer_parser,
            "documents": itemgetter("documents"),
        }
    )

    return final_chain
