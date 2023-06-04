from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from PIL import Image
import os
load_dotenv()
def makeknowledge():
    directory="pdf_files"
    text=""

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                
                pdf_reader = PdfReader(directory+'\\'+file)
                for page in pdf_reader.pages:
                    text +=page.extract_text()
                    


    
    # Split
    text_splitter= CharacterTextSplitter(separator="\n", chunk_size=500,chunk_overlap=100, length_function=len)
    chunks=text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

knowledge_base = makeknowledge()



def main():
    #load_dotenv()
    #os.environ["OPENAI_API_KEY"]=st.secrets["OPENAI_API_KEY"]

    st.set_page_config(page_title="IMD Weatherman")
    st.header("Interactive weather Chat")
    logo_image = Image.open("logo.jpg")

    # Display the logo in the top left-hand side
    st.image(logo_image, use_column_width=False, width=100)

    
 
      
      # show user input
    user_question = st.text_input("Ask me a question about what I learned:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)


if __name__ == '__main__':
    main()
