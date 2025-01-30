from langchain_core.documents.base import Document


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_docs_source(docs: list[Document]) -> list[str]:
    try:
        return list(
            set([doc.metadata["_source"]["metadata"]["source"] for doc in docs])
        )
    except KeyError as e:
        print(f"KeyError: Missing key in document metadata - {e}")
        return []
    except Exception as e:
        print(f"Unexpected error while processing documents: {e}")
        return []
