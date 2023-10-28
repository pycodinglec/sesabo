"""
세사보: 세화고등학교 사무 보조
세화고등학교의 각종 규정 및 문서를 기반으로 응답을 제공하는 프로그램
"""

# 향후개발목표

# 개선된 청킹 및 검색 적용하기
# Request 전 토큰 수 제한 검토 단계를 추가하자. 
    # 토큰 수 검토 후, 초과된다면 가장 오래된 요청부터 날리자
    # msgs[1::2]
# 향후 @sehwa.hs.kr 테넌트 내의 사람들만 이용할 수 있도록 하자.
# 향후 들어오는 요청을 두 가지 종류로 나누고, 종류에 따라 응답하자.
    # 첫 번째 요청 타입: 문서의 참고(검색)이 필요한 경우 함수 호출
    # 두 번째 요청 타입: 문서의 참고(검색)이 불필요한 경우 함수 비호출
# 현재 15개(아무튼 상수개)의 응답을 검토하도록 Fix해놨는데, 개선가능
    # 'text A가 query B에 대한 응답을 생성하기에 충분한 답변을 포함하고 있어?'
    # 이것을 n번 반복하다가, 최대 개수에 다다르면 '죄송합니다 문서 내용에 없습니다'


# 향후 st.multiselect를 이용해 검색 대상 문서를 사용자가 선택할 수 있도록 하자
# 향후 st.warning 등을 이용해 사용자에게 경고를 보내자(너무 많은 이용 시 등)


import openai
import streamlit as st
import os
import pickle
#import dotenv
import pandas as pd
import numpy as np
from scipy import spatial

#dotenv.load_dotenv()
GPT_MODEL = 'gpt-4'
EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_FOLDER = 'documents_embed_150_30'

def initialize_conversation():
    with open("system_message.txt", "r", encoding="utf-8") as f:
        system_message = f.read()
    hello_message = '👋안녕하세요 선생님, 무엇을 도와드릴까요?'
    system = {'role':'system', 'content': system_message}
    hello = {'role':'assistant', 'content': hello_message}
    msgs = [system, hello]
    return msgs

def initialize_documents_embedding():
    all_data = []
    embedding_files = [file for file in os.listdir(EMBEDDING_FOLDER) if file.endswith('pkl')]
    for embedding_file in embedding_files:
        with open(f'{EMBEDDING_FOLDER}/{embedding_file}', 'rb') as f:
            df2 = pickle.load(f)
            all_data.append(df2)
    
    df = pd.concat(all_data, ignore_index=True)
    return df

def get_modified_prompt(original_prompt):
    if 'df' not in st.session_state:
        st.session_state['df'] = initialize_documents_embedding()
    df = st.session_state['df']
    response = openai.Embedding.create(
                model = EMBEDDING_MODEL,
                input = original_prompt,
    )
    prompt_embedding = response["data"][0]["embedding"]
    df['similarity'] = df['embedding'].apply(lambda x: np.nan_to_num(1 - spatial.distance.cosine(prompt_embedding, x), nan=0))
    df = df.sort_values(by='similarity', ascending=False)
    modified_prompt = f'''근거 자료를 줄 테니까 질문에 대답해. 만약 질문에 관련된 내용을 근거 자료에서 찾지 못하겠다면, '관련 내용을 찾을 수 없다'고 답하면 돼.
    질문: {original_prompt}
    근거 자료: {df.text[:15]}'''
    
    return [{'role': 'user', 'content': modified_prompt}]

def main():
    with st.sidebar:
        if st.button('대화 초기화'):
            st.session_state['msgs'] = initialize_conversation()
        st.write('아직 프로토타입이라 정확도가 낮지만, 개선될 예정입니다.')
        st.write('현재 검색 가능한 문서:')
        st.caption('교육계획서, 학교운영위원회 규정, 학업성적관리규정 등')
    st.title('세사보: 세화고등학교 사무 보조')
    # openai.api_key = st.text_input('OpenAI API KEY:', type = 'password')
    # openai.api_key = os.getenv('OPENAI_API_KEY') # for debug
    st.caption('세사보는 선생님의 사무를 돕기 위하여 학교의 문서를 검색합니다.')
     
    if 'msgs' not in st.session_state:
        st.session_state['msgs'] = initialize_conversation()
    if 'df' not in st.session_state:
        st.session_state['df'] = initialize_documents_embedding()

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
            modified_msgs = st.session_state['msgs'][:-1] + get_modified_prompt(st.session_state['msgs'][-1]['content'])
            responses = openai.ChatCompletion.create(
                model = GPT_MODEL,
                messages = modified_msgs,
                stream = True,
            )
            for response in responses:
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state['msgs'].append({"role": "assistant", "content": full_response})

if __name__=='__main__':
    main()
