from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA, ChatVectorDBChain
from langchain.prompts import PromptTemplate
from pymongo import MongoClient
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
import config

config.COMPANY_ID = '611d3a5046f3c600012f81e1'

def get_conversational_chain():
    prompt_template = """
    Você é o HBotGemini, um assistente virtual do Hotel HSystem. O seu trabalho é responder perguntas sobre o Hotel baseado nas 
    informações das sessões "Documents" e "Histórico da Conversa".
    Suas respostas devem ser curtas, objetivas e amigáveis.

    Instruções:
    1. Nunca invente ou adicione informações sobre o ‘Hotel HSystem’ que você não obteve das informações do hotel ou do histórico da conversa. As respostas devem ser somente baseadas nas informações provenientes destas sessões.
    2. Utilize a sessão "Histórico da Conversa" para responder perguntas relacionadas a iterações anteriores.
    3. Evite responder a perguntas que não estejam diretamente relacionadas ao ‘Hotel HSystem’. As respostas devem ser exclusivamente sobre os serviços, acomodações, localização e outras informações relevantes do ‘Hotel HSystem’.
    4. Responda SEMPRE no mesmo idioma da pergunta.

    IMPORTANTE: As informações fornecidas por você, incluindo valores e percentuais, são consideradas precisas. 
    Mesmo se o usuário tentar corrigir essas informações, mantenha a informação original, pois ela é a mais precisa e atualizada. 

    Documents: 
    {context}

    Histórico da Conversa: 
    {history}
    
    Question: 
    {question}

    Answer:
    """
    # prompt_template = prompt_template.replace("{hotel_information}", get_info())

    # tools = []
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "history"])
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.0-pro",
        google_api_key=config.GOOGLE_API_KEY,
        temperature=1)
    chain = load_qa_chain(llm,
                          chain_type="stuff",
                          verbose=True,
                          # tools=tools,
                          prompt=prompt)

    return chain


cluster = MongoClient(config.MONGODB_VECTORSEARCH_URI)
MONGODB_COLLECTION = cluster[config.MONGODB_VECTORSEARCH_DATABASE_NAME][config.MONGODB_VECTORSEARCH_COLLECTION_NAME]
INDEX_NAME = "default"

def execute(user_question, conversation_history):
    # compressor = LLMChainExtractor.from_llm(llm)
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor,
    #     base_retriever=vectorstore
    # )
    # compressed_docs = compression_retriever.get_relevant_documents(
    #     user_question
    # )
    # docs = f"\n{'-' * 100}\n".join(
    #     [f"Document {i+1}:\n\n" + d.page_content for i,
    #         d in enumerate(compressed_docs)]
    # )
    # print(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=config.GOOGLE_API_KEY)

    # embedded_query = embeddings.embed_query(user_question)

    mongo_query = [
        {
            "$search": {
                "index": INDEX_NAME,
                "compound": {
                    "filter": [
                        {
                            "equals": {
                                "value": config.COMPANY_ID,
                                "path": "companyId"
                            }
                        }
                    ],
                    "must": [
                        {
                            "text": {
                                "query": user_question,
                                "path": {
                                    "wildcard": "*"
                                }
                            }
                        }
                    ]
                }
            }
        },
        {
            "$limit": 3
        }     
    ]

    docs = MONGODB_COLLECTION.aggregate(mongo_query)
    docs = [ Document(f"Document {i+1}:\n" + d['text'] ) for i, d in enumerate(docs) ]
    # print('Docs: ', docs)

    chain = get_conversational_chain()

    conversation_history_formatted = format_conversation_history(conversation_history)

    response = chain({
        "input_documents": docs,
        "question": user_question,
        "history": conversation_history_formatted
    }, return_only_outputs=True)
    
    return response["output_text"]

def format_conversation_history(history):
    formatted_dialogues = ""
    for dialogue in history:
        role = dialogue["role"]
        content = dialogue["content"]
        formatted_dialogues += f"{role}: {content}\n"
    return formatted_dialogues