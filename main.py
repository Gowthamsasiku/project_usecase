import os


import numpy as np
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings


os.environ["OPENAI_API_KEY"] = "AIzaSyCifH1ULezGIabrpPk8gcYT1WMoXskvl5k"
DATA_FOLDER = r"C:\Users\HP\Desktop\Usecase\project_usecase\Data"
print("📁 Checking folder path:", DATA_FOLDER)
print("📄 Files found in folder:", os.listdir(DATA_FOLDER))

def load_documents(folder_path):
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
            # load() returns an iterable of Document objects
            docs.extend(loader.load())
            print(f"Loaded: {filename}")
        except Exception as e:
            print(f"⚠️ Error loading {filename}: {e}")
    return docs

# if __name__ == "__main__":
#     documents = load_documents(DATA_FOLDER)
#     print(f"\n📚 Total documents loaded: {len(documents)}")



# vectorDB
def create_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# -------------------- EXECUTIVE SUMMARY --------------------
def get_high_level_summary(llm, retriever):
    context = "\n".join([doc.page_content[:1000] for doc in retriever.vectorstore.similarity_search("overview", k=5)])
    prompt = f"Provide a high-level executive summary of the following insurance data:\n\n{context}\n\nSummarize in 2 paragraphs."
    response = llm.invoke(prompt)
    return response.content

# -------------------- DEEP DIVE --------------------
def deep_dive(llm, retriever, topic):
    docs = retriever.vectorstore.similarity_search(topic, k=5)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Explain in detail about '{topic}' based on these documents:\n\n{context}\n\n"
    response = llm.invoke(prompt)
    return response.content

# -------------------- MAIN --------------------
def main():
    print("Loading insurance data...")
    documents = load_documents(DATA_FOLDER)
    print(f"Loaded {len(documents)} documents")

    print("Creating vector store...")
    vectorstore = create_vectorstore(documents)
    retriever = vectorstore.as_retriever()

    ChatGoogleGenerativeAI(model="gemini-1.5-flash")


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
        print(f"\n🔍 Deep Dive on '{topic}':\n")
        print(detail)

if __name__ == "__main__":
    main()

