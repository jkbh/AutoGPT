from ..registry import action
from ...memory.chroma_memstore import ChromaMemStore

import os


@action(
    name="store_document",
    description="Embed a document in the vector database for similarity search",
    parameters=[
        {
            "name": "document",
            "description": "Document to store",
            "type": "string",
            "required": True,
        }
    ],
    output_type="None",
)
async def store_document(
    agent,
    task_id: str,
    document: str,
):
    """
    Add a document to the vector databse
    """
    db = ChromaMemStore("./chroma")
    db.add(task_id, document, None)


@action(
    name="query_vectorstore",
    description="Query the vector database with a prompt to find documents similar to the prompt",
    parameters=[
        {
            "name": "document",
            "description": "Document to use for the query",
            "type": "string",
            "required": True,
        }
    ],
    output_type="dict",
)
async def query_vectorstore(
    agent,
    task_id: str,
    document: str,
):
    """
    Query the vector database
    """
    db = ChromaMemStore("./chroma")
    result = db.query(task_id, document)
    agent.workspace.write(
        task_id=task_id, path="resources.txt", data=result["documents"]
    )
    return db.query(task_id, document)
