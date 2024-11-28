from RAG1 import rag
def main():
    rag_chat = rag()
    save_path = "faiss_store"
    # File upload
    print("Upload a file (CSV, PDF, or JSON) or enter 'q' to quit:")
    while True:
        file_path = input("Enter file path: ").strip()
        if file_path.lower() == "q":
            break

        try:
            documents = rag_chat.load_documents(file_path)
            print(f"Loaded {len(documents)} documents.")
            
            chunks = rag_chat.split_documents(documents)
            print(f"Split into {len(chunks)} chunks.")

            # Create FAISS index
            rag_chat.create_faiss_vectorstore(chunks)
            print("FAISS vector store created.")

            # Save FAISS index
            rag_chat.save_vector_store(save_path)
            print(f"FAISS vector store saved to {save_path}.")
        except Exception as e:
            print(f"Error: {e}")

    # Querying the FAISS index
    print("\nYou can now ask questions based on the uploaded data:")
    while True:
        query = input("Enter your query (or 'q' to quit): ").strip()
        if query.lower() == "q":
            break

        try:
            rag_chat.load_vector_store(save_path)
            answer = rag_chat.query_rag(query)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")
if __name__ == "__main__":
    main() 