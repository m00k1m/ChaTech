import os
import csv
import time
import torch
import argparse
import chromadb
import datetime
import gradio as gr

from groq import Groq
from pathlib import Path
from prompt_db import *
from chromadb.utils import embedding_functions


def get_chroma_collection(db_path: str, collection_name: str, *, embedf_name: str = "") -> chromadb.Collection | None:
    """
    ChromaDB 클라이언트 및 컬렉션 로드
    input
        dp_path         : chromadb colletion이 존재하는 절대 경로
        collection_name : chromadb colletion의 이름
    output
        collectoin      : chromadb collection 객체
    """
    if not os.path.exists(db_path):
        print(f"collection {collection_name} 을(를) 찾을 수 없습니다. 경로를 다시 확인해주세요.")
        return None

    chro_client = chromadb.PersistentClient(path=db_path)

    if embedf_name:
        embed_fun = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name = embedf_name,
            device = "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"임베딩 함수로 {embedf_name} 를 사용합니다. ")
    else:
        embed_fun = embedding_functions.DefaultEmbeddingFunction()
        print("임베딩 함수로 기본 임베딩 함수를 사용합니다. ")

    # 기존 collection 불러오기
    try:
        collection = chro_client.get_collection(
            name = collection_name, 
            embedding_function = embed_fun)
        print(f"Collection '{collection_name}' 을(를) 성공적으로 불러왔습니다. ")
        return collection
    
    except Exception as e:
        print(f"Collection '{collection_name}' 을(를) 불러오지 못했습니다 : {e}")
        return None


def query_db(collection: chromadb.Collection, 
             query_text: str, 
             n_results: int) -> str:
    """
    사용자 질문과 관련된 문서를 DB(collection)에서 검색하여 반환
    input
        collection : 
        query_text : 
        n_results  : 
    output
        data       : 사용자의 질문과 관련된 문서
    """
    if collection is None:
        print("데이터베이스가 연결되지 않았습니다.")
        return ""
    
    try:
        results = collection.query(
            query_texts = [query_text],
            n_results = n_results
        )

        # 검색된 문서가 없는 경우
        if not results["documents"] or not results["documents"][0]:
            print("관련된 문서를 찾을 수 없습니다.")
            return ""
        
        # 검색된 문서들을 하나의 문자열로 결합
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        
        context_parts = []
        for i, doc in enumerate(documents):
            source = metadatas[i].get("title", "제목 없음")
            date = metadatas[i].get("date", "날짜 없음")
            context_parts.append(f"문서{i+1} [제목: {source}, 날짜: {date}]\n내용 : {doc}")
        
        data = "\n\n".join(context_parts)
        return data
        
    except Exception as e:
        print(f"검색 중 오류 발생: {e}")
        return ""


def save_log(base_dir, log_dir, request, user_message, assistant_message):
    """
    대화 로그 저장 함수
    """
    log_path = os.path.join(base_dir, log_dir)

    if not os.path.exists(log_path):
        os.mkdir(log_path)
        print(f"{log_dir} 폴더가 생성되었습니다 : {log_path}")

    # 현재 경로 내에 있는 {log_dir} 폴더 내에 대화 로그 파일이 없는 경우 -> csv파일 생성
    # 각 csv 파일은 날짜별로 구분
    today = datetime.datetime.now().strftime("%y%m%d")
    file_name = f"chat_log_{today}.csv"
    dest_file_path = os.path.join(log_path, file_name)

    if not os.path.exists(dest_file_path):
        with open(dest_file_path, mode = "w", newline = "", encoding = "utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["user_ip", "time_stamp", "user_message", "assistant_message"])
    
    # 챗봇과의 대화 로그를 기록
    user_ip = request.client.host if request else "Unknown_IP"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_conv_log = [user_ip, timestamp, user_message, assistant_message]
    
    try:
        with open(dest_file_path, mode = "a", newline = "", encoding = "utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(user_conv_log)
    except Exception as e:
        print(f"대화 로그 저장 실패 : {e}")

def get_response(user_message: str, 
                 system_prompt: str, 
                 collection: chromadb.Collection, 
                 history: list[dict | list], 
                 request: gr.Request, 
                 client: Groq, 
                 base_dir: str,
                 log_dir: str, 
                 model_name: str, 
                 n_results: int,
                 temperature: float):
    
    if user_message.strip() == "끝끝":
        end_message = "대화를 종료합니다. 새 대화를 시작하려면 오른쪽 상단의 Clear 버튼(휴지통 아이콘)을 클릭해주세요."
        yield end_message
        return
    
    # RAG: 사용자 질문과 관련된 Context 검색
    context = query_db(collection = collection, 
                       query_text = user_message, 
                       n_results= n_results)

    # System Prompt에 Context 주입
    formatted_system_prompt = system_prompt.format(context=context)

    # 메시지 구성
    messages = [{"role": "system", "content": formatted_system_prompt}]

    for chat in history:
        if isinstance(chat, dict):
            messages.append({"role": chat["role"], "content": chat["content"]})
        # 구버전 gradio 위함
        elif isinstance(chat, list) and len(chat) == 2:
            messages.append({"role": "user", "content": chat[0]})
            messages.append({"role": "assistant", "content": chat[1]})
    
    messages.append({"role": "user", "content": user_message})

    # LLM에게 답변 생성 요청    
    try:
        response = client.chat.completions.create(
            model = model_name, 
            messages = messages, 
            temperature = temperature, 
            stream = True 
        )

        # 사용자에게 챗봇의 답변이 실시간으로 입력되는 것처럼 보여줌
        assistant_message = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                assistant_message += delta
                yield assistant_message

    except Exception as e:
        error_message = f"답변 생성 중 오류가 발생했습니다. : {str(e)}"
        yield error_message
        assistant_message = error_message
    
    save_log(base_dir, log_dir, request, user_message, assistant_message)


def chat_with_rag(api_key: str, 
                  collection: chromadb.Collection, 
                  system_prompt: str, 
                  args: argparse.ArgumentParser) -> None:
    """
    RAG 챗봇 실행
    input
        dd
    output
        -
    """
    try:
        groq_client = Groq(api_key = api_key)
    except Exception as e:
        print(f"Groq client를 불러오지 못했습니다. API Key를 확인해주세요 : {e}")

    def predict(user_message, history, request: gr.Request):
        yield from get_response(
                user_message = user_message,
                system_prompt = system_prompt,
                collection = collection, 
                history = history, 
                request = request, 
                client = groq_client, 
                base_dir = args.base_dir,
                log_dir = args.log_dir, 
                model_name = args.model_name, 
                n_results = args.n_results,
                temperature = args.temperature
        )
    
    title = "ChaTech"
    description = """
    서울과학기술대학교 공지사항 기반 질의응답 챗봇입니다.
    데이터베이스에 저장된 공지사항 내용을 바탕으로 답변합니다.
    대화 종료를 원하실 경우 채팅창에 \'끝끝\'을 입력해주세요. 
    """

    demo = gr.ChatInterface(
        fn = predict, 
        title = title, 
        description = description
    ).queue()

    demo.launch(debug = True, share = True)



def get_system_prompt(prompt_type: str) -> str:
    """
    prompt_db.py로부터 시스템 프롬프트를 불러와서 반환
    input
        prompt_type  : 사용할 시스템 프롬프트 종류
            v    : vanilla prompt
            adv1 : advanced prompt ver.1 (미구현) 
    output
        system_prompt : 시스템 프롬프트 전문
    """

    if prompt_type == "v":
        vanilla = Vanilla()
        system_prompt = vanilla.get_prompt()
        return system_prompt
    
    # 개선된 프롬프트 버전, 아직 미구현
    elif prompt_type == "adv1":
        system_prompt = ""
        return system_prompt
    else:
        print("유효하지 않은 프롬프트 타입입니다. 기본값(Vanilla)을 사용합니다. ")
        system_prompt = vanilla.get_prompt()
        return system_prompt


def main(args):
    # chromadb collection 경로 설정
    abs_db_path = os.path.join(args.base_dir, args.db_dir)

    # collection 객체 불러오기
    collection = get_chroma_collection(abs_db_path, args.collection_name)
    # embedding function로 다른 모델을 사용할 경우
    # collection = get_chroma_collection(abs_db_path, args.collection_name, embedf_name = args.embedf_name)
    
    if collection is None:
        print("Chromadb Collection을 불러오지 못했습니다. 프로그램을 종료합니다. ")
        return

    # 시스템 프롬프트 불러오기
    system_prompt = get_system_prompt(args.prompt_type)

    # 챗봇 실행
    chat_with_rag(api_key = args.api_key, 
                  collection = collection, 
                  system_prompt = system_prompt,
                  args = args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type = str, default = "")
    parser.add_argument("--base_dir", type = str, default = str(Path(__file__).resolve().parent))
    parser.add_argument("--db_dir", type = str, default = "seoultech_data_db")
    parser.add_argument("--log_dir", type = str, default = "chat_log")
    parser.add_argument("--model_name", type = str, default = "llama-3.3-70b-versatile")   # llama-3.1-8b-instant  llama-3.3-70b-versatile openai/gpt-oss-120b
    parser.add_argument("--temperature", type = float, default = 0.5)
    parser.add_argument("--n_results", type = int, default = 3)
    parser.add_argument("--collection_name", type = str, default = "seoultech_notices")
    parser.add_argument("--embedf_name", type = str, default = "BAAI/bge-m3")
    parser.add_argument("--prompt_type", type = str, default = "v")
    
    args = parser.parse_args()
    main(args)



