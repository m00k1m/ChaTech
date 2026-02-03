import os
import argparse
from pathlib import Path
from crawl import *
from chatbot import *


def main(args):
    # 허깅페이스 secret key로부터 api key 읽어오기
    api_key = args.api_key
    if not api_key:
        api_key = str(os.environ.get("GROQ_API_KEY", ""))
        print(f"Groq API Key를 성공적으로 불러왔습니다. 뒷자리 4글자 : ...{api_key[-4:]}")

    if not api_key:
        print("API Key가 설정되지 않았습니다. Hugging Face Secrets에서 'GROQ_API_KEY'를 설정하거나 --api_key 인자를 입력해주세요.")


    # 크롤러 파트
    abs_download_path = os.path.join(args.base_dir, args.download_dir)
    abs_db_path = os.path.join(args.base_dir, args.db_dir)

    collection = make_db(abs_download_path, abs_db_path, args.collection_name)
    # 기본 임베딩 함수 외의 함수를 이용할 경우
    #collection = make_db(abs_download_path, abs_db_path, args.collection_name, embedf_name = args.embedf_name)

    crawl_seoultech_notice(abs_download_path, args.base_url, args.num_page, collection)

    # 챗봇 파트
    collection = get_chroma_collection(abs_db_path, args.collection_name)
    # embedding function로 다른 모델을 사용할 경우
    # collection = get_chroma_collection(abs_db_path, args.collection_name, embedf_name = args.embedf_name)
    
    if collection is None:
        print("Chromadb Collection을 불러오지 못했습니다. 프로그램을 종료합니다. ")
        return

    # 시스템 프롬프트 불러오기
    system_prompt = get_system_prompt(args.prompt_type)

    # 챗봇 실행
    chat_with_rag(api_key = api_key, 
                  collection = collection, 
                  system_prompt = system_prompt,
                  args = args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 공통 인자
    parser.add_argument("--base_dir", type = str, default = str(Path(__file__).resolve().parent))     # 현재 이 파일이 있는 디렉토리
    parser.add_argument("--db_dir", type = str, default = "seoultech_data_db")
    parser.add_argument("--collection_name", type = str, default = "seoultech_notices")
    parser.add_argument("--embedf_name", type = str, default = "BAAI/bge-m3")
    
    # 크롤러
    parser.add_argument("--base_url", type = str, default = "https://www.seoultech.ac.kr/service/info/notice")
    parser.add_argument("--download_dir", type = str, default = "seoultech_data_download")
    parser.add_argument("--header", type = dict, default = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
    parser.add_argument("--num_page", type = int, default = 1)

    # 챗봇
    parser.add_argument("--api_key", type = str, default = "")
    parser.add_argument("--log_dir", type = str, default = "chat_log")
    parser.add_argument("--model_name", type = str, default = "llama-3.3-70b-versatile") # llama-3.1-8b-instant  llama-3.3-70b-versatile openai/gpt-oss-120b
    parser.add_argument("--temperature", type = float, default = 0.5)
    parser.add_argument("--n_results", type = int, default = 3)
    parser.add_argument("--prompt_type", type = str, default = "v")

    args = parser.parse_args()
    main(args)