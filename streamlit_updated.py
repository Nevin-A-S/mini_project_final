import streamlit as st
from scraper import scraper
from summarizer import summarizer
from chatbot_aio import chatbot as GBot
from chatbot_aio import chatbotLLama as LBot
from chatbot_aio import chatbotMix as MBot

def main():
    st.set_page_config(page_title="Web Summarizer Chatbot")
    st.markdown("<h1 style='text-align: center;'>Web Summarizer Chatbot</h1>", unsafe_allow_html=True)
    
    model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768', 'llama3-70b-8192' , 'gemini-pro']
    )

    url = st.text_input("Enter URL:")
    if st.button("Get Summary"):
        st.session_state.text = scraper(url)
        st.session_state.summary = summarizer(st.session_state.text,model)
        st.success(st.session_state.summary)
    
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if model =='gemini-pro':

        if prompt := st.chat_input("Enter your question (gemini-pro): "):

            st.session_state.messages.append({"role": "user", "content": prompt})


            with st.chat_message("user"):
                st.markdown(prompt)


            with st.chat_message("assistant"):
                text = scraper(url)
                gbot = GBot(text)
                response = gbot.qa_chain({"query": prompt})
            
        
                st.write(response['result'])
                
            st.session_state.messages.append({"role": "assistant", "content": response['result']})
            
    elif model=='llama3-70b-8192':
        try:
            if prompt := st.chat_input("Enter your question (LLama3): "):

                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    text = scraper(url)
                    lbot = LBot(text)
                    response = lbot.final_result(prompt)
                
                    st.write(response['result'])

                st.session_state.messages.append({"role": "assistant", "content": response['result']})
        except Exception as e:
            print(e)
            with st.chat_message("assistant"):
                text = scraper(url)
                gbot = GBot(text)
                response = gbot.qa_chain({"query": prompt})
        
    
                st.write(response['result'])
            
            st.session_state.messages.append({"role": "assistant", "content": response['result']})
            
    
    else:
        try:
            if prompt := st.chat_input("Enter your question (mixtral-8x7b-32768): "):

                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    text = scraper(url)
                    mbot = MBot(text)
                    response = mbot.qa_chain({"query": prompt})
                
                    st.write(response['result'])

                st.session_state.messages.append({"role": "assistant", "content": response['result']})
        except Exception as e:
            print(e)
            with st.chat_message("assistant"):
                text = scraper(url)
                gbot = GBot(text)
                response = gbot.qa_chain({"query": prompt})
        
    
                st.write(response['result'])
            
            st.session_state.messages.append({"role": "assistant", "content": response['result']})

if __name__ == '__main__':
    main()