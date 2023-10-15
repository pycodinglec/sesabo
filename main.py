"""
세사보: 세화고등학교 사무 보조
세화고등학교의 각종 규정 및 문서를 기반으로 응답을 제공하는 프로그램
"""

import openai
import streamlit as st
import os
import dotenv
dotenv.load_dotenv()


def initialize_conversation():
    system_message = 'You are a helpful assistant'
    hello_message = '👋안녕하세요 선생님, 무엇을 도와드릴까요?'
    msgs = list()
    system = {'role':'system', 'content': system_message}
    hello = {'role':'assistant', 'content': hello_message}
    msgs += [system, hello]
    return msgs

def main():
    with st.sidebar:
        if st.button('대화 초기화'):
            st.session_state['msgs'] = initialize_conversation()
        st.caption('현재 검색 가능한 문서: a, b, c, ...')
    st.title('세사보: 세화고등학교 사무 보조')    
    #openai.api_key = st.text_input('OpenAI API KEY:', type = 'password')
    openai.api_key = os.getenv('OPENAI_API_KEY') # for debug
    st.caption('세사보는 선생님의 사무를 돕기 위하여 학교의 문서를 검색합니다.')
    
    if 'openai_model' not in st.session_state:
        st.session_state['openai_model'] = 'gpt-3.5-turbo'
    if 'msgs' not in st.session_state:
        st.session_state['msgs'] = initialize_conversation()
    

    for msg in st.session_state['msgs'][1:]:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    if prompt:= st.chat_input("이곳에 요청을 입력"):
        st.session_state['msgs'].append({'role':'user', 'content': prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state['msgs']
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state['msgs'].append({"role": "assistant", "content": full_response})

if __name__=='__main__':
    main()

    # 향후 st.multiselect를 이용해 검색 대상 문서를 사용자가 선택할 수 있도록 하자
    # 향후 st.warning 등을 이용해 사용자에게 경고를 보내자(너무 많은 이용 시 등)
    # 향후 st.rerun을 이용해서 페이지 새로고침 없이도 새 대화를 시작하게 하자.
    # 향후 @sehwa.hs.kr 테넌트 내의 사람들만 이용할 수 있도록 할... 수 있을까?
