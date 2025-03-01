"""
This module contains the code to embed a PDF file into a Qdrant collection.
"""

import glob
from langchain.docstore.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from dotenv import dotenv_values


def embed_text(collection, dataset, markdown_file, overwrite=False):
    """
    Embeds the text from a PDF file into a Qdrant collection.
    Args:
        collection: The name of the Qdrant collection to create.
        markdown_file: The path to the markdown file containing the text to embed.
        overwrite: Whether to overwrite the existing collection with the same name.
    """
    with open(markdown_file, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    if overwrite:
        client = QdrantClient(url="http://localhost:6333")
        client.delete_collection(collection)

    config = dotenv_values(".env")

    embedding_llm = AzureOpenAIEmbeddings(
        azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=config.get("AZURE_OPENAI_Embedding_DEPLOYMENT_NAME"),
        api_key=config.get("AZURE_OPENAI_KEY"),
        openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    )

    doc = Document(
        page_content=markdown_text, metadata={"dataset": dataset, "file": markdown_file}
    )

    markdown_splitter = RecursiveCharacterTextSplitter(
        separators=["#", "##", "###", "\n\n", "\n", " "],
        chunk_size=1000,  # 可根據需求調整 chunk 大小
        chunk_overlap=100,  # 重疊區域，避免語境斷裂
    )
    documents = markdown_splitter.split_documents([doc])

    qdrant = QdrantVectorStore.from_documents(
        documents,
        embedding=embedding_llm,
        url="http://localhost:6333",
        collection_name=collection,
    )

    return qdrant


if __name__ == "__main__":
    files = glob.glob("recipe/*.md")
    for file in files:
        embed_text("bootcamp", "recipe", file)
        print(f"Embedded text from {file}")
