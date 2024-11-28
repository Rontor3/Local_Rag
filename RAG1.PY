from langchain_community.document_loaders import CSVLoader, PyPDFLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import tempfile
import os
class rag:
    def __init__(self, embedding_model="sentence-transformers/distiluse-base-multilingual-cased-v1"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None
    def load_documents(self,uploaded_file):
            # Determine file extension
        file_name = uploaded_file.name
        ext = os.path.splitext(file_name)[1].lower()
        
        try:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            if ext == ".csv":
                loader = CSVLoader(file_path=temp_file_path, encoding="utf-8")
                raw_documents = loader.load()
                combined_docs = []

                for doc in raw_documents:
                    # Extract metadata and data
                    metadata = doc.metadata  # Retains original metadata
                    row_data = doc.page_content.split("\n")  # Ensure this matches your CSV format

                    # Combine all columns into a single string
                    combined_content = " | ".join(row_data)

                    # Create a new Document with combined content
                    combined_docs.append(Document(page_content=combined_content, metadata=metadata))

                return combined_docs

            elif ext == ".pdf":
                loader = PyPDFLoader(temp_file_path)
                return loader.load()

            elif ext == ".json":
                loader = JSONLoader(temp_file_path, jq_schema="*")  # Adjust jq_schema based on your JSON structure
                return loader.load()

            else:
                raise ValueError(f"Unsupported file type: {ext}")

        except Exception as e:
            print(f"Error processing the uploaded file: {e}")
            return []
    def split_documents(self,combined_docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = []
        for doc in combined_docs:
            # Ensure `doc.page_content` is split into strings
            if not isinstance(doc.page_content, str):
                raise ValueError(f"Invalid page_content type: {type(doc.page_content)}. Expected a string.")
            doc_chunks = text_splitter.split_text(doc.page_content)
            for chunk in doc_chunks:
                # Create a new Document for each chunk, preserving metadata
                chunks.append(Document(page_content=chunk, metadata=doc.metadata))
        return chunks
        #return [chunk for doc_chunks in chunks for chunk in doc_chunks]  # Flatten  

    def create_faiss_vectorstore(self,chunks):
        """
        Create a FAISS vector store with LangChain.
        """
        self.vector_store= FAISS.from_documents(chunks,self.embedding_model)
    def save_vector_store(self, path):
        """
        Save the FAISS vector store locally.
        """
        if self.vector_store:
            self.vector_store.save_local('d:/Rakshit/Local_Rag/'+(path))
        else:
            raise ValueError("No vector store to save.")
    def load_vector_store(self,path):
        """
        Load a FAISS vector store from local storage.
        """
        self.vector_store = FAISS.load_local('d:/Rakshit/Local_Rag/'+path, self.embedding_model,allow_dangerous_deserialization=True)
    def query_rag(self,query_text: str):
        # Load FAISS index.
        if not self.vector_store:
            raise ValueError("No vector store loaded. Please create or load a vector store first.")
        # Search the FAISS index.
        results =self.vector_store.similarity_search_with_score(query_text, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        
        PROMPT_TEMPLATE = """
                Answer the question based only on the following context:

                {context}

                ---

                Answer the question based on the above context: {question}
                """

        # Format the prompt.
        prompt_template = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Use Ollama for answering
        model = Ollama(model="mistral")
        response_text = model.invoke(prompt)

        # Extract sources
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        return response_text
    
    
    def inspect_faiss(self):
        """
        Inspect the contents of a FAISS vector store.
        """
        # Get the underlying FAISS index
        faiss_index = self.vector_store.index

        # Number of vectors in the index
        num_vectors = faiss_index.ntotal
        print(f"Number of vectors in the FAISS index: {num_vectors}")

        # Dimensions of the vectors
        dimension = faiss_index.d
        print(f"Dimension of each vector: {dimension}")

        vector_id = 0
        vector = faiss_index.reconstruct(vector_id)
        print("First stored vector:", vector)
        # Optional: Retrieve vectors and inspect metadata
        if num_vectors > 0:
            # Retrieve the first vector and associated metadata
            print("\nSample metadata:")
            for i, doc in enumerate(self.vector_store.docstore._dict.values()):
                print(f"Document {i + 1}:")
                print(f"Content: {doc.page_content}")
                print(f"Metadata: {doc.metadata}")
                if i >= 4:  # Limit output for readability
                    break    
            
        
#if __name__ == "__main__":     
        #embedding_model = SentenceTransformerEmbeddings(model_name="distiluse-base-multilingual-cased-v1")  
        #obj=rag()
        #docs=obj.load_documents()
        #chunks=obj.split_documents(docs)
        
        # Create and save FAISS vector store
        #vector_store = obj.create_faiss_vectorstore(chunks)
        #path='faiss_store'
        
        #obj.inspect_faiss(vector_store)

        # Load FAISS vector store
        #loaded_store = load_vectorstore(path="faiss_store")

        # Query the vector store
        #query = "What is the purpose of ApplyUpdateRequest?"
        #top_results = query_vectorstore(loaded_store, query)