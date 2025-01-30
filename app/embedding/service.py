import os
import time
import itertools
import asyncio
from typing import Any

from langchain_core.documents.base import Document
from langchain_core.documents.base import Blob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from rag.models.models import get_embedding_model
from rag.doc_processing.loader import BlobProcessor, DocumentMetadata
from fastapi import UploadFile, HTTPException


class EmbeddedDocument(BaseModel):
    uuid: str
    success: bool
    contents: list[str]
    embeddings: list[list[float]]
    metadata: list[dict[Any, Any]]


class DocumentMetadata:
    drug: str
    category: str


class DocumentEmbedderService:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,  # TODO: parametrize based on model
        chunk_overlap=0,
        add_start_index=True,
        separators=[".\n", ".", ",", ":"],
    )

    def chunks(self, l, n):
        """Yield n number of striped chunks from l."""
        for i in range(0, n):
            yield l[i::n]

    async def generate_doc_embeddings(self, docs: list[str]) -> list[list[float]]:
        start_time = time.monotonic()
        embedding_model = get_embedding_model("open-ai")
        print("time: ", time.monotonic() - start_time)

        return await embedding_model.aembed_documents(docs)

    async def process_docs(self, docs: list[Document], uuid: str) -> EmbeddedDocument:
        docs_content: list[str] = [doc.page_content for doc in docs]
        docs_metadata = [doc.metadata for doc in docs]

        embedded_docs_content = await asyncio.gather(
            *[
                self.generate_doc_embeddings(chunk)
                for chunk in self.chunks(docs_content, 10)
            ]
        )
        return EmbeddedDocument(
            contents=docs_content,
            embeddings=list(
                itertools.chain(*embedded_docs_content),
            ),
            metadata=docs_metadata,
            uuid=uuid,
            success=True,
        )

    async def embed_document(
        self, upload_file: UploadFile, uuid: str
    ) -> EmbeddedDocument:
        try:
            blob: Blob = Blob.from_data(
                upload_file.file.read(), path=upload_file.filename
            )

            extension: str = ""
            if upload_file.filename is not None:
                extension = os.path.splitext(upload_file.filename)[1]

            metadata = DocumentMetadata()
            metadata.drug = "drug"

            docs: list[Document] = BlobProcessor(
                blob, extension, metadata, self.text_splitter
            ).load()
        except:
            return EmbeddedDocument(
                contents=[], embeddings=[], metadata=[], uuid=uuid, success=False
            )
        return await self.process_docs(docs, uuid)

    async def embed_documents(
        self, upload_files: list[UploadFile], file_uuids: list[str]
    ):
        if len(upload_files) != len(file_uuids):
            raise HTTPException(
                status_code=422, detail="Number of files and uuids don't match"
            )

        processed_documents = await asyncio.gather(
            *[
                self.embed_document(upload_file, file_uuid)
                for upload_file, file_uuid in zip(upload_files, file_uuids)
            ]
        )

        return processed_documents
