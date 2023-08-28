import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, user_template, bot_template

def get_pdf_text(docs):
    text = ""
    for doc in docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(large_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(large_text)
    return chunks

def get_vector_store(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name= 'hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    converstational_chain  = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return converstational_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question':user_question})
    #st.write(response)
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if (i%2==0):
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    # st.write(user_template.replace("{{MSG}}", "Hello DocuChat!"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "Hello Earther!"), unsafe_allow_html=True)

def main():
    load_dotenv()   # import the values from .env

    # to reinintialize the converstation varibale ONLY IF the app reruns
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.set_page_config(page_title="DocuChat", page_icon=":books:")
    st.header("Chat with your documents here! :books:")
    user_question = st.text_input("Ask your question here.")
    if user_question:
        handle_user_input(user_question)

    st.write(css, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        raw_docs = st.file_uploader("Upload your documents here (in PDF format)", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get the pdf's text
                pdf_text = get_pdf_text(raw_docs)
                # with open("extracted.txt", 'w', encoding="utf-8") as file:
                #     file.write(pdf_text)

                # chunk the text
                #text_chunks = get_text_chunks(pdf_text)
                # with open("chunks.txt", 'w', encoding="utf-8") as file:
                #     for chunk in text_chunks:
                #         file.write(chunk)
                        #file.write("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

                # create a vector store
                #vector_store = get_vector_store(text_chunks)

                # create conversation chain
                #st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == "__main__":
    main()