import os
import time
import torch
import requests
import fitz
import olefile
import zlib
import re
import traceback
import chromadb
import argparse
import zipfile
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from docx import Document
from urllib.parse import urljoin
from pathlib import Path
from chromadb.utils import embedding_functions


def make_db(download_dir: str, db_dir: str, collection_name: str, *, embedf_name: str = "") -> chromadb.Collection:
    """
    chromadb의 collection 객체를 만들어서 반환하는 함수

    input
        download_dir : 웹페이지에서 각 게시글의 첨부파일을 다운로드해서 저장할 폴더 (절대 경로)
        db_dir       : chroma db를 저장할 로컬 폴더 (절대 경로)
        embedf_name(optional) : 텍스트를 임베딩할 함수 이름
    output
        collection 객체

    ***
    DefaultEmbeddingFunction 말고 다른걸 사용하려면

    pip install -q sentence-transformers 설치하고

    """

    print("chromadb collection 객체를 로컬에 생성합니다. ")

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"게시글의 첨부파일을 다운로드할 폴더가 생성되었습니다 : {download_dir}")
    else:
        print(f"게시글의 첨부파일을 다운로드할 폴더가 이미 존재합니다. :{download_dir}")
    
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"chromadb collection을 저장할 폴더가 생성되었습니다 : {db_dir}")
    else:
        print(f"chromadb collection을 저장할 폴더가 이미 존재합니다. : {db_dir}")

    # chroma db 클라이언트 객체 생성, 데이터는 로컬에 저장
    chroma_client = chromadb.PersistentClient(path = db_dir)
    print("chromadb client가 생성되었습니다.")
    
    # 데이터를 embedding할 때 사용할 함수 지정, 사용가능한 함수 목록은 아래 웹페이지 참고
    # https://docs.trychroma.com/docs/embeddings/embedding-functions
    if embedf_name:
        embed_fun = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name = embedf_name,
            device = "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"임베딩 함수로 {embedf_name} 를 사용합니다. ")
    else:
        embed_fun = embedding_functions.DefaultEmbeddingFunction()
        print("임베딩 함수로 기본 임베딩 함수를 사용합니다. ")

    # Collection 존재 여부 확인 및 생성 혹은 불러오기
    existing_collections = [c.name for c in chroma_client.list_collections()]

    # Collection이 이미 존재하는 경우 -> 불러오기
    if collection_name in existing_collections:
        print(f"기존 데이터베이스를 발견했습니다. \'{collection_name}\' Collection을 불러옵니다.")
        collection = chroma_client.get_collection(
            name = collection_name,
            embedding_function = embed_fun
        )
        print(f"Collection 객체를 성공적으로 불러왔습니다 : {db_dir}")
    
    # Collection이 존재하지 않는 경우 -> 새로 생성
    else:
        print(f"기존 데이터베이스가 존재하지 않습니다. \'{collection_name}\' Collection을 새로 생성합니다.")
        collection = chroma_client.create_collection(
            name = collection_name,
            embedding_function = embed_fun
        )
        print(f"Collection 객체가 성공적으로 생성되었습니다 : {db_dir}")

    return collection


def get_post_urls(current_url: str) -> list[str]:
    """
    게시판 페이지에서 개별 게시글의 url을 추출해서 리스트로 모아주는 함수

    input
        current_url : 게시판의 현재 페이지 링크, 기본적으로 {args.base_url}?page=1 (서울과기대 공지사항 url)
    output
        post_urls : 현재 페이지 내의 각 게시글 링크들의 리스트
    """

    print(f"{current_url}의 게시글을 수집합니다. ")
    post_urls = []

    try:
        res = requests.get(current_url)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        links = soup.select("td.tit a")

        if not links:
            links = soup.select(".board_list a")

        for link in links:
            href = link.get("href")
            if href and "javascript" not in href:
                full_url = urljoin(current_url, href)
                post_urls.append(full_url)
        
        output = list(set(post_urls))
        print(f"{len(output)}개의 게시글 링크를 수집 완료했습니다. ")
        return output
    
    except Exception as e:
        print(f"게시글 링크 수집 중 에러가 발생했습니다 : {e}")
        return []


def download_file(url: str, filename :str, download_dir: str) -> None:
    """
    주어진 다운로드 링크로부터 파일을 로컬에 다운로드하는 함수

    input
        url          : 다운로드 링크
        filename     : 저장할 파일 이름
        download_dir : 파일을 저장할 폴더 (절대 경로)
    output
        -
    """
    print(f"{url} 로부터 \'{filename}\' 을(를) {download_dir} 에 다운로드합니다.")

    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        
        # 파일명에 윈도우에서 사용할 수 없는 특수문자가 있다면 제거
        valid_name = "".join(c for c in filename if c not in '<>:"/\\|?*')
        save_path = os.path.join(download_dir, valid_name)
        
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"다운로드 성공 : {filename}")

    except Exception as e:
        print(f"다운로드 실패 ({filename}): {e}")


class TextParser:
    """
    각 파일의 확장자에 맞게 파일로부터 텍스트를 추출하는 역할

    @staticmethod 의미
     : 해당 메소드가 클래스 안에 있긴 하지만, 클래스 내부 정보는 필요 없다
       = self.xxx 이런식으로 클래스 내부 변수 등 사용 X
       그냥 소속만 이 클래스고 클래스랑은 독립적으로 사용하겠다. 요런 의민가봄

       사용하는 이유 : 여러 용도가 비슷한 함수들을 하나의 클래스 안에 모아놓는 용도
    """
    @staticmethod
    def parse_pdf(file_path: str) -> list[str]:
        """
        PDF 파일에서 텍스트를 추출하는 함수

        input
            file_path : (로컬에) 저장된 .pdf 형식 파일의 경로
        output
            texts : pdf 파일에서 추출한 텍스트 리스트
        """

        texts = []

        try:
            doc = fitz.open(file_path)
            for page in doc:
                text = page.get_text().strip()
                if text:
                    texts.append(text)
            doc.close()
            print(f"{os.path.basename(file_path)}로부터 텍스트를 성공적으로 파싱하였습니다. ")
            return texts
        
        except Exception as e:
            err_msg = f"{os.path.basename(file_path)}를 파싱하는 과정에서 에러가 발생했습니다 : {str(e)}]"
            print(err_msg)
            return [err_msg]
    # pdf 파싱은 문제 없음

    @staticmethod
    def parse_docx(file_path: str) -> list[str]:
        """
        워드 파일에서 텍스트를 추출하는 함수

        input
            - file_path : (로컬에) 저장된 .docx 형식 파일의 경로
        output
            - texts : docx 파일에서 추출한 텍스트 리스트
        """
        texts = []

        try:
            doc = Document(file_path)
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    texts.append(text)
            print(f"{os.path.basename(file_path)}로부터 텍스트를 성공적으로 파싱하였습니다. ")
            return texts
        
        except Exception as e:
            err_msg = f"{os.path.basename(file_path)}를 파싱하는 과정에서 에러가 발생했습니다 : {str(e)}]"
            print(err_msg)
            return [err_msg]


    @staticmethod
    def parse_hwp(file_path: str) -> list[str]:
        """
        한글 파일에서 텍스트를 추출하는 함수  ### hwp5-txt 등의 도구 이용?
        현재는 젬미니가 짜준 코드 그대로 가져옴 -> 최적화 필요

        input
            file_path : (로컬에) 저장된 .hwp 형식 파일의 경로
        output
            texts : hwp 파일에서 추출한 텍스트 리스트
        """
        texts = []

        try:
            # .hwp 파일인지 확인
            if not olefile.isOleFile(file_path):
                err_msg = f"{os.path.basename}은(는) 유효한 OLE 파일이 아닙니다"
                print(err_msg)
                return [err_msg]
            
            f = olefile.OleFileIO(file_path)

            # 2. 본문(BodyText) 섹션 목록 가져오기
            # HWP 파일 내부는 'BodyText/Section0', 'BodyText/Section1'... 형태로 저장됨
            dirs = f.listdir()

            # 'BodyText' 디렉토리 하위의 'Section'으로 시작하는 스트림만 필터링하고 번호순 정렬
            body_sections = [d for d in dirs if d[0] == "BodyText" and d[1].startswith("Section")]
            # Section 뒤의 번호를 기준으로 정렬 (Section0, Section1, ...)
            body_sections.sort(key=lambda x: int(x[1].replace("Section", "")))

            # 3. 각 섹션의 텍스트 추출
            for section in body_sections:
                # 스트림 읽기
                body_stream = f.openstream(section)
                data = body_stream.read()
                
                # HWP 5.0부터는 본문이 zlib으로 압축되어 있음 (-15는 헤더 없는 raw stream을 의미)
                try:
                    unpacked_data = zlib.decompress(data, -15)
                except Exception:
                    # 압축되지 않은 경우(매우 드뭄)를 대비
                    unpacked_data = data

                # UTF-16LE로 디코딩
                extracted_text = unpacked_data.decode('utf-16le', errors='ignore')

                # 4. 텍스트 정제
                # HWP 바이너리에는 텍스트 외에 제어 문자나 태그 정보가 섞여 있어 정제가 필요함
                # 한글, 영문, 숫자, 기본 특수문자 및 개행 문자만 남기고 필터링하는 방식 사용
                cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9\s.,?!()~%+-]', '', extracted_text)
                
                # 불필요한 공백 제거
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

                if cleaned_text:
                    texts.append(cleaned_text)

            f.close()
            print(f"{os.path.basename(file_path)}로부터 텍스트를 성공적으로 파싱하였습니다.")
            return texts
        
        except Exception as e:
            err_msg2 = f"{os.path.basename(file_path)}를 파싱하는 과정에서 에러가 발생했습니다 : {str(e)}]"
            print(err_msg2)
            return [err_msg2]
        
    
    @staticmethod
    def parse_hwpx(file_path: str) -> list[str]:
        """
        hwpx 파일에서 텍스트를 추출하는 함수
        hwpx는 zip 파일 형식이므로 zipfile을 이용해 xml을 파싱함
        input
            file_path : (로컬에) 저장된 .hwpx 형식 파일의 경로
        output
            texts : hwpx 파일에서 추출한 텍스트 리스트
        """
        texts = []

        try:
            # zip 파일로 열기
            with zipfile.ZipFile(file_path, 'r') as zf:
                # zip 파일 내의 파일 목록 가져오기
                file_list = zf.namelist()

                # 본문 텍스트가 들어있는 xml 파일 찾기 (보통 Contents/section0.xml 형태)
                # 순서를 보장하기 위해 정렬
                section_files = sorted([f for f in file_list if f.startswith('Contents/section') and f.endswith('.xml')])

                if not section_files:
                    return ["본문 섹션을 찾을 수 없습니다."]

                for section in section_files:
                    # xml 데이터 읽기
                    xml_data = zf.read(section)

                    # XML 파싱
                    root = ET.fromstring(xml_data)

                    # 네임스페이스 처리 (hwpx 내부 xml은 네임스페이스를 사용함)
                    # 보통 <hp:t> 태그 안에 텍스트가 있음. 네임스페이스 무시하고 태그 이름만으로 찾거나 namespace map을 써야 함.
                    # 여기서는 간단히 모든 텍스트 노드를 순회하며 추출

                    # hp:t (text) 태그를 찾는 것이 가장 정확함.
                    # 편의상 namespace {http://www.hancom.co.kr/hwpml/2011/paragraph} 를 고려해야 함
                    ns = {'hp': 'http://www.hancom.co.kr/hwpml/2011/paragraph'}

                    # 해당 섹션 내의 모든 문단(<hp:p>)을 찾음
                    for para in root.findall('.//hp:p', ns):
                        para_text = ""
                        # 문단 내의 런(<hp:run>) -> 텍스트(<hp:t>) 추출
                        for text_node in para.findall('.//hp:t', ns):
                            if text_node.text:
                                para_text += text_node.text

                        # 텍스트 정제 (기존 hwp 코드와 동일한 로직 적용)
                        cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9\s.,?!()~%+-]', '', para_text)
                        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

                        if cleaned_text:
                            texts.append(cleaned_text)

            print(f"{os.path.basename(file_path)}로부터 텍스트를 성공적으로 파싱하였습니다.")
            return texts

        except Exception as e:
            err_msg = f"{os.path.basename(file_path)} 파싱 중 에러 발생: {str(e)}"
            print(err_msg)
            return [err_msg]


    @staticmethod
    def get_text(file_path: str) -> list[str]:
        """
        확장자에 따라 적절한 파서 호출
        input:
            file_path : 로컬에 저장된 게시글의 첨부파일의 경로
        output:
            list[str] : 게시글에서 추출된 텍스트
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return TextParser.parse_pdf(file_path)
        elif ext == ".docx":
            return TextParser.parse_docx(file_path)
        elif ext == ".hwp":
            return TextParser.parse_hwp(file_path)
        elif ext == ".hwpx":
            return TextParser.parse_hwpx(file_path)
        else:
            print(f"지원하지 않는 파일 형식입니다. : {ext}")
            return [""]


def parse_attachment(filename: str, download_dir: str) -> list[str]:
    """
    로컬에 저장된 첨부파일을 확장자에 맞게 파싱하여 텍스트를 반환하는 함수

    input 
        filename     : 다운로드 된 각 첨부파일의 이름
        download_dir : 다운로드된 첨부파일의 절대 경로
    output
        texts : 파일로부터 추출된 텍스트 데이터
    """

    try:
        file_path = os.path.join(download_dir, filename)

        texts = TextParser.get_text(file_path)

        return texts
    
    except Exception as e:
        err_msg = f"{filename}이(가) 다운로드되지 않았습니다 : {e}"
        print(err_msg)
        return [err_msg]


def parse_post_content(post_url: str, download_dir: str) -> dict[str, str | list[str]] | None:
    """
    각 게시글 상세페이지에서 본문 텍스트와 첨부파일을 크롤링하는 함수

    input
        post_url     : 각 게시글의 url
        download_dir : 게시글의 첨부파일을 저장할 절대 경로
    output
        output : 게시글로부터 추출된 텍스트 및 메타데이터를 담고 있는 딕셔너리 
        output = {
                "source_url": post_url,             # 게시글의 url
                "title": title,                     # 게시글 제목
                "date": date,                       # 게시글 작성 날짜
                "category": sub_title,              # 게시글이 담긴 카테고리
                "text": full_text,                  # 게시글 제목 + 본문
                "attachments": attachments_str,     # 게시글의 첨부파일 리스트(파일명)
                "extracted": extracted_str,         # 첨부파일로부터 추출한 텍스트 
                "crawl_date": now_time              # 크롤링된 날짜와 시간
                }
    """

    try:
        res = requests.get(post_url)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")


        # 게시글 제목 얻기
        title_element = soup.find("th", string = "제목")
        if title_element:
            title = title_element.find_next_sibling("td").get_text(strip=True)
        else:
            title = "제목없음"
        

        # 게시글 작성 날짜 얻기
        date_element = soup.find("th", string = "날짜")
        if date_element:
            date = date_element.find_next_sibling("td").get_text(strip=True)
        else:
            date = "날짜없음"


        # 게시글 내용 얻기
        content_element = soup.select_one(".cont")
        
        if content_element:
            # 불필요한 스크립트, 스타일, 주석 제거
            for script in content_element(["script", "style", "iframe"]):
                script.decompose()
            body_text = content_element.get_text(separator='\n', strip=True)
        else:
            body_text = "본문없음"

        full_text = f"제목: {title}\n본문: {body_text}"
        

        # 게시글 카테고리 얻기
        # 카테고리 : 대학공지사항, 학사공지, 장학공지, 대학원공지, 등등..
        sub_title_element = soup.select_one(".sub_title")
        if sub_title_element:
            sub_title = sub_title_element.get_text(strip=True)
        else:
            sub_title = "카테고리없음"


        # 첨부파일 얻기
        attachments = []
        extracted_text = []

        file_links = soup.select(".list_attach a")

        if file_links:
            for f_link in file_links:
                href = f_link.get("href")

                if not href:
                    continue
                
                if "javascript:downloadfile" in href:
                    try:
                        match = re.search(r"downloadfile\(\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*'([^']*)'\s*\)", href)

                        if match:
                            path = match.group(1)
                            server_fname = match.group(2)    # 이건 서버에 저장된 파일 이름. e.g. 55E0D06B18CD4638B3515EDCC4D43130_.hwp
                            origin_fname = match.group(3)    # 이게 실제 파일 이름. e.g. 연구활동종사자 안전교육 신청 및 이수 방법(모바일, 온라인).hwp

                            base_url = "https://www.seoultech.ac.kr"
                            download_url = f"{base_url}{path}/{server_fname}"

                            # attachments는 메타데이터용 텍스트
                            attachments.append(f"{origin_fname} ({download_url})")

                            # 첨부파일을 실제로 로컬에 다운로드
                            download_file(download_url, origin_fname, download_dir)

                            # 다운로드된 첨부파일로부터 텍스트 추출
                            details = parse_attachment(origin_fname, download_dir)
                            
                            # 추출된 텍스트를 extracted_text에 추가
                            for s in details:
                                extracted_text.append(s)

                    except Exception as e:
                        print(f"Javascript 파싱 에러 : {e}")

                else:
                    full_url = urljoin(post_url, href)
                    f_name = f_link.get_text(strip=True)
                    attachments.append(f"{f_name} ({full_url})")

        
        attachments_str = ", ".join(attachments) if attachments else "첨부파일 없음"
        extracted_str = ", ".join(extracted_text) if extracted_text else "첨부파일 없음"


        # DB에 저장할 최종 데이터 형식은 다음과 같음 
        now_time = time.strftime("%Y-%m-%d %H:%M:%S")

        output = {
            "source_url": post_url,             # 게시글의 url
            "title": title,                     # 게시글 제목
            "date": date,                       # 게시글 작성 날짜
            "category": sub_title,              # 게시글이 담긴 카테고리
            "text": full_text,                  # 게시글 제목 + 본문
            "attachments": attachments_str,     # 게시글의 첨부파일 리스트(파일명)
            "extracted": extracted_str,         # 첨부파일로부터 추출한 텍스트 
            "crawl_date": now_time              # 크롤링된 날짜와 시간
            }
        return output


    except Exception as e:
        err_msg = f"{post_url}을(를) 파싱하는 과정에서 에러가 발생했습니다 : {e}"
        print(err_msg)
        traceback.print_exc()
        return None


def save_to_db(data: dict[str, str | list[str]], collection: chromadb.Collection) -> None:
    """
    각 게시글마다 parse_post_content를 통해 얻어낸 결과물을 ChromaDB에 저장하는 함수
    input 
        data       : parse_post_content의 output
        collection : chromadb 클라이언트 객체
    output
        - 
    기능 : collection에 각 data를 저장
            id는 각 게시글의 url(각 데이터의 고유한 주소이므로)
    
    참고) chromadb에 데이터를 저장하려면 
        documents : 벡터화되어 검색의 대상이 되는 텍스트 -> 본문 + 첨부파일
        metadatas : 검색 결과와 함께 반환되거나, 필터링에 사용할 정보 -> 제목, 날짜 등
        ids       : 각 데이터의 고유한 id 
    """

    # 임베딩할 텍스트 : 게시글 본문
    # 첨부파일이 있을 경우 해당 정보도 함께 임베딩
    content_to_embed = data["text"]
    if data.get("extracted"):
        content_to_embed += f"\n\n[첨부파일 내용]\n{data["extracted"]}"

    metadata_dict = {
        "source_url": data["source_url"],
        "title": data["title"],
        "date": data["date"],
        "category": data["category"],
        "attachments": data["attachments"] if data["attachments"] else "",
        "crawl_date": data["crawl_date"]
    }

    # upsert는 기존에 동일한 id가 있으면 업데이트하고, 새로운 id면 db에 추가함. 
    collection.upsert(
        ids = [data["source_url"]],
        documents = [content_to_embed],
        metadatas = [metadata_dict]
    )

    print(f"다음 게시글이 저장되었습니다. : {data["title"]}")


def crawl_seoultech_notice(download_dir: str, base_url: str, num_pages: str, collection: chromadb.Collection) -> None:
    """
    crawl_seoultech_notice의 Docstring
    
    input
        download_dir : 게시글의 첨부파일을 다운로드할 폴더 (절대 경로)
        base_url     : 기본 공지사항 페이지 url
        num_pages    : 크롤링할 페이지 수
        collection   : Chromadb Collection 객체 - 데이터를 저장할 저장소
    output
        -
    """
    t0 = time.time()
    print(f"{base_url} 로부터 데이터 크롤링을 시작합니다. ")
    print(f"예상 소요 시간 : 약 {num_pages * 20} ~ {num_pages * 30}초")

    for i in range(num_pages):
        current_url = f"{base_url}?page={i+1}"
        print(f"=== {i+1} 페이지 크롤링 시작 : {current_url} ===")

        post_urls = get_post_urls(current_url)

        for post in post_urls:
            post_data = parse_post_content(post, download_dir)

            if post_data:
                try:
                    save_to_db(post_data, collection)
                
                except Exception as e:
                    err_msg = f"게시물의 데이터를 DB에 저장하는 과정에서 에러가 발생하였습니다 : {e}"
                    print(err_msg)

            else:
                print(f"파싱 실패로 데이터 저장을 건너뜁니다 : {post}")
    
    t1 = time.time()
    print(f"=== {i+1} 페이지 수집 완료 (소요 시간 : {t1 - t0:.2f}초)===")


def main(args):

    abs_download_path = os.path.join(args.base_dir, args.download_dir)
    abs_db_path = os.path.join(args.base_dir, args.db_dir)

    collection = make_db(abs_download_path, abs_db_path, args.collection_name)
    # 기본 임베딩 함수 외의 함수를 이용할 경우
    #collection = make_db(abs_download_path, abs_db_path, args.collection_name, args.embedf_name)


    crawl_seoultech_notice(abs_download_path, args.base_url, args.num_page, collection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type = str, default = "https://www.seoultech.ac.kr/service/info/notice")
    parser.add_argument("--base_dir", type = str, default = str(Path(__file__).resolve().parent))     # 현재 이 파일이 있는 디렉토리
    parser.add_argument("--download_dir", type = str, default = "seoultech_data_download")
    parser.add_argument("--db_dir", type = str, default = "seoultech_data_db")
    parser.add_argument("--header", type = dict, default = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
    parser.add_argument("--num_page", type = int, default = 1)
    parser.add_argument("--collection_name", type = str, default = "seoultech_notices")
    parser.add_argument("--embedf_name", type = str, default = "BAAI/bge-m3")

    args = parser.parse_args()
    main(args)