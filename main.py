import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# Gemini API Key
os.environ["GOOGLE_API_KEY"] = "Your GOOGLE API KEY"  # replace with your actual key

# Folder containing your data files
DATA_FOLDER = r"C:\Users\HP\Desktop\Usecase\project_usecase\Data" #Replace your data folder
CHROMA_DB_PATH = r"C:\Users\HP\Desktop\Usecase\project_usecase\ChromaDB"

print(" Checking folder path:", DATA_FOLDER)
print(" Files found in folder:", os.listdir(DATA_FOLDER))


# -------------------- LOAD DOCUMENTS --------------------
def load_documents(folder_path):
    """Load all PDF and Excel files as LangChain Documents."""
    docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.lower().endswith((".xlsx", ".xls")):
                loader = UnstructuredExcelLoader(file_path)
            else:
                print(f"Skipping unsupported file: {filename}")
                continue
            docs.extend(loader.load())
            print(f" Loaded: {filename}")
        except Exception as e:
            print(f" Error loading {filename}: {e}")
    return docs


# -------------------- CREATE OR LOAD CHROMA --------------------
def create_or_load_chroma(documents):
    """Split documents into chunks and store/reuse them in ChromaDB."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Check if Chroma DB already exists
    if os.path.exists(CHROMA_DB_PATH):
        print(" Loading existing Chroma database...")
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    else:
        print(" Creating new Chroma database...")
        vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DB_PATH)
        vectorstore.persist()
        print(" Chroma database saved successfully.")
    return vectorstore


# -------------------- EXECUTIVE SUMMARY --------------------
def get_high_level_summary(llm, retriever):
    """Generate a 2-paragraph executive summary."""
    docs = retriever.vectorstore.similarity_search("overview", k=5)
    context = "\n".join([doc.page_content[:1000] for doc in docs])
    prompt = f"Provide a high-level executive summary of the following insurance data:\n\n{context}\n\nSummarize in 2 paragraphs."
    response = llm.invoke(prompt)
    return response.content


# -------------------- DEEP DIVE --------------------
def deep_dive(llm, retriever, topic):
    """Detailed explanation for a specific topic."""
    docs = retriever.vectorstore.similarity_search(topic, k=5)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Explain in detail about '{topic}' based on these documents:\n\n{context}\n\n"
    response = llm.invoke(prompt)
    return response.content


# -------------------- MAIN --------------------
def main():
    print(" Loading insurance data...")
    documents = load_documents(DATA_FOLDER)
    print(f" Total documents loaded: {len(documents)}")

    print(" Creating or loading Chroma vector store...")
    vectorstore = create_or_load_chroma(documents)
    retriever = vectorstore.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    print("\n Data loaded successfully.")
    ask = input("\nDo you want a high-level Executive Summary? (yes/no): ").strip().lower()

    if ask == "yes":
        summary = get_high_level_summary(llm, retriever)
        print("\n EXECUTIVE SUMMARY:\n")
        print(summary)

    while True:
        deep = input("\nDo you want to deep dive into any specific document or topic? (yes/no): ").strip().lower()
        if deep != "yes":
            print(" Exiting analysis. Thank you!")
            break
        topic = input("Enter topic or document name to analyze: ")
        detail = deep_dive(llm, retriever, topic)
        print(f"\n Deep Dive on '{topic}':\n")
        print(detail)


if __name__ == "__main__":
    main()
