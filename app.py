import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

load_dotenv()

def create_agent_chain():
    chat = ChatOpenAI(
        model_name=os.environ["OPENAI_API_MODEL"],
        temperature=os.environ["OPENAI_API_TEMPERATURE"],
        streaming=True
    )

    # OpenAI Functions Agent のプロンプトに Memory の会話履歴を追加するための設定
    agent_kwards = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")]
    }
    # OpenAI Functions Agent が使える設定で Memory を初期化
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
    
    tools = load_tools(["ddg-search", "wikipedia"])
    return initialize_agent(
        tools,
        chat,
        agent=AgentType.OPENAI_FUNCTIONS,
        agent_kwargs=agent_kwards,
        memory=memory
    )

if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()

st.title("langchain-streamlit-app")

if "messages" not in st.session_state:   # セッションステートに messages がない場合 
    st.session_state.messages = []       # messages を空のステートで初期化

for message in st.session_state.messages:   # セッションステートの messages でループ
    with st.chat_message(message["role"]):  # ロール毎に
        st.markdown(message["content"])     # 保存されているテキストを表示

prompt = st.chat_input("What is up")

if prompt: # 入力された文字列がある (None でもから文字列でもない) 場合
    # ユーザーの入力内容を messages に追加
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
        })

    with st.chat_message("user"):   # ユーザーのアイコンで
        st.markdown(prompt)         # prompt をマークダウンとして整形して表示

    with st.chat_message("assistant"): # AI のアイコンで
        callback = StreamlitCallbackHandler(st.container())
        response = st.session_state.agent_chain.run(prompt, callbacks=[callback])
        st.markdown(response) # 応答をマークダウンとして整形して表示
    
    # 応答を messages に追加
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

