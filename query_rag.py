import argparse
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms.ollama import Ollama
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
import torch

# Define constants
FAISS_INDEX_PATH = "d:/Rakshit/Local-Rag/faiss_store"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Load FAISS index.
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v1")
    db = FAISS.load_local(FAISS_INDEX_PATH, embedding_function,allow_dangerous_deserialization=True)
    # Search the FAISS index.
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)

    # Format the prompt.
    prompt_template = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

     # Use Ollama for answering
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    # Extract sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


    # Extract sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
