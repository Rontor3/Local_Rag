import streamlit as st
from RAG1 import rag

def main():
    st.title("Interactive RAG Chat Interface")
    st.sidebar.title("Settings")

    # Instantiate RAG object
    rag_chat = rag()
    save_path = "faiss_store"

    # Upload File Section
    st.sidebar.header("Upload Your File")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a file (CSV, PDF, or JSON):", type=["csv", "pdf", "json"]
    )

    if uploaded_file:
        try:
            file_type = uploaded_file.name.split(".")[-1].lower()

            st.sidebar.write(f"File uploaded: {uploaded_file.name} ({file_type.upper()})")
            print(uploaded_file)
            # Process the file using your existing RAG functions
            documents = rag_chat.load_documents(uploaded_file)
            chunks = rag_chat.split_documents(documents)
            rag_chat.create_faiss_vectorstore(chunks)
            rag_chat.save_vector_store(save_path)
            st.success("Data successfully processed and FAISS vector store created.")
        except Exception as e:
            st.error(f"Error processing the file: {e}")

    # Query Section
    st.header("Query Your Data")
    query = st.text_input("Enter your query:")
    if st.button("Submit Query"):
        if uploaded_file:
            try:
                rag_chat.load_vector_store(save_path)
                answer = rag_chat.query_rag(query)
                st.success(f"Answer: {answer}")
            except Exception as e:
                st.error(f"Error querying data: {e}")
        else:
            st.warning("Please upload a file first.")

if __name__ == "__main__":
    main()
