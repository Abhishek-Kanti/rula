import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import faiss
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_Template, user_Template

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = faiss.FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    Myllm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=Myllm, 
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversational_chain

def handle_userInput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_Template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
           st.write(bot_Template.replace("{{MSG}}", message.content), unsafe_allow_html=True) 

def main():
    load_dotenv()
    st.set_page_config(page_title="Rula")

    st.write(css, unsafe_allow_html=True)

    # Adding image in circle on the right side of "Rula"
    st.markdown('<div class="circle-image"><img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" alt="Profile Image"> </div>', unsafe_allow_html=True)

    st.markdown("# Hi I am Parker!")
    st.caption("### Park with Parker!")

    st.markdown(
        """
        <div class="beginning-background">
            <div class="circle-image img"><img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" alt="Logo" class="logo"></div>
            <h2>How can I help you today?</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    user_question = st.chat_input("Ask your question here") #user question
    if user_question:
        handle_userInput(user_question)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on ***Process***.", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing.."):
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                # Adding memory for context
                st.session_state.conversation = get_conversation_chain(vectorstore)
                # tokens_used = conversation["usage"]["total_tokens"]
                # st.write(f"Number of tokens used: {tokens_used}")

if __name__== '__main__' :
    main()
