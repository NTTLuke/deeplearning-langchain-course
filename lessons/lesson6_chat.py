import os, sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from langchain.vectorstores import Chroma
from llms.llm import azure_openai_embeddings, azure_chat_openai_llm
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter


persist_directory = "data/chroma/"
embedding = azure_openai_embeddings()


def load_vectordb():
    # Load PDF
    loaders = [
        # Duplicate documents on purpose - messy data
        PyPDFLoader("./documents/MachineLearning-Lecture01.pdf"),
    ]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # Split
    token_splitter = TokenTextSplitter(chunk_size=600, chunk_overlap=10)

    splits = token_splitter.split_documents(docs)
    print(splits[0])
    print(len(splits))

    persist_directory = "data/chroma/"

    print("Loading vector database")
    vectordb = Chroma.from_documents(
        documents=splits, embedding=embedding, persist_directory=persist_directory
    )
    print("Done loading vector database")
    # print(f"count : {vectordb._collection.count()}")


def query_vector_using_custom_prompt():
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    llm = azure_chat_openai_llm()

    # print(llm.predict("Hello world!"))

    # Build prompt
    from langchain.prompts import PromptTemplate

    template = """Use the following pieces of context to answer the question at the end.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Use three sentences maximum. Keep the answer as concise as possible.
                Always say "thanks for asking!" at the end of the answer.
    {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # Run chain without memory
    from langchain.chains import RetrievalQA

    question = "Is probability a class topic?"
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    result = qa_chain({"query": question})
    result["result"]


def query_vector_using_default_prompt_with_memory():
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    llm = azure_chat_openai_llm()

    # Run chain with Memory
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain

    retriever = vectordb.as_retriever()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory
    )

    question = "Is probability a class topic?"
    result = qa({"question": question})
    print(result["answer"])

    question = "why are those prerequesites needed?"
    result = qa({"question": question})
    print(result["answer"])


load_vectordb()
