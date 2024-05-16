import google.generativeai as genai
from groq import Groq
from langchain.chains import ConversationChain
from langchain_groq import ChatGroq


groq_api_key = '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'

genai.configure(api_key='AIzaSyAm5HDnero63r30sO3kLyyaYLrGzUDOA20')

API_KEY = 'AIzaSyAm5HDnero63r30sO3kLyyaYLrGzUDOA20'


def summarizer(text, model):
    model = model.lower()
    prompt = """Welcome, Text Summarizer! Your task is to distill the essence of a given text document into a concise summary. Your summary should capture the key points and essential information, presented in bullet points, within a 250-word limit. Let's dive into the provided transcript and extract the vital details for our audience. Content is: """
    
    try:
        if model == 'gemini-pro':
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt + "".join(text))
            return response.text

        groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)
        conversation = ConversationChain(llm=groq_chat)
        response = conversation(prompt + " ".join(text))
        return response['response']
    except Exception as e:
        print(f"Error encountered: {e}. Switching to Gemini model.")
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + "".join(text))
        return response.text