import os
import json
import torch
from flask import Flask, request, jsonify, Response
from transformers import AutoTokenizer, AutoModel
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings.base import Embeddings
from sklearn.preprocessing import normalize

# Flask 앱 생성
app = Flask(__name__)

# 전역 설정
FOLDER_PATH = "db"  # 벡터 저장소 디렉토리
PDF_FOLDER = "pdf"  # PDF 파일 업로드 디렉토리
LLM_MODEL = "llama3"  # 사용할 LLM 모델

# KoE5 임베딩 클래스
class KoE5Embedding(Embeddings):
    def __init__(self):
        # KoE5 모델과 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained("nlpai-lab/KoE5")
        self.model = AutoModel.from_pretrained("nlpai-lab/KoE5")

    def embed_documents(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # CLS 토큰 사용
        normalized_embeddings = normalize(embeddings, norm="l2")  # L2 정규화
        return normalized_embeddings.tolist()

    def embed_query(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # CLS 토큰 사용
        normalized_embeddings = normalize(embeddings, norm="l2")  # L2 정규화
        return normalized_embeddings.squeeze().tolist()

# 전역 객체 초기화
embedding_function = KoE5Embedding()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
cached_llm = Ollama(model=LLM_MODEL)
raw_prompt = PromptTemplate.from_template(
    """
    <s>[INST]너는 문서 검색을 매우 잘하는 비서야. 주어진 문서를 보고 질문에 대해 알맞은 답변을 도출해줘. 답변은 무조건 한국어로만 해줘. [/INST]</s>
    [INST] {input}
           Context: {context}
           Answer: 
    [/INST]
    """
)

# pdf 업로드 및 벡터DB에 저장
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        file_name = file.filename
        save_path = os.path.join(PDF_FOLDER, file_name)
        os.makedirs(PDF_FOLDER, exist_ok=True)
        file.save(save_path)

        # PDF 파일 로드 및 텍스트 분할
        loader = PDFPlumberLoader(save_path)
        docs = loader.load_and_split()

        # 텍스트 분할 확인
        if not docs or not isinstance(docs, list):
            return jsonify({"error": "No valid documents found in the PDF"}), 400

        chunks = text_splitter.split_documents(docs)

        # chunks가 올바른 리스트인지 확인
        if not chunks:
            return jsonify({"error": "No valid text chunks generated"}), 400
        
        # 벡터 저장소 생성 및 저장
        vector_store = Chroma.from_documents(chunks, embedding_function, persist_directory=FOLDER_PATH)
        vector_store.persist()

        response = {
            "status": "success",
            "filename": file_name,
            "doc_len": len(docs),
            "chunk_len": len(chunks)
        }
         
        return Response(
            json.dumps(response, ensure_ascii=False),
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# PDF 기반 답변 생성
@app.route("/ask_pdf", methods=["POST"])
def ask_pdf():
    try:
        json_content = request.json
        query = json_content.get("query")
        if not query:
            return jsonify({"error": "No query provided"}), 400

        # 벡터 저장소 로드 (저장된 벡터 저장소 사용)
        vector_store = Chroma(persist_directory=FOLDER_PATH, embedding_function=embedding_function)

        # 검색기 및 체인 생성
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"k": 10, "score_threshold": 0.3}  # score_threshold을 높이고 k를 적당히 설정
        )
        document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
        chain = create_retrieval_chain(retriever, document_chain)

        # 문서 검색
        retrieved_docs = retriever.get_relevant_documents(query)

        # 디버깅: 검색된 문서 출력
        if not retrieved_docs:
            return jsonify({
                "answer": "No relevant documents were retrieved for the query.",
                "context": []
            })

        # 체인 실행
        result = chain.invoke({"input": query, "context": retrieved_docs})
        response_answer = {
            "answer": result.get("answer", "No answer generated."),
            "context": [doc.page_content[:200] for doc in retrieved_docs]  # 문서 일부 반환
        }

        return Response(
        json.dumps(response_answer, ensure_ascii=False),
        mimetype='application/json'
    )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# AI 호출 (단순 LLM 호출)
@app.route("/ai", methods=["POST"])
def ai_post():
    try:
        json_content = request.json
        query = json_content.get("query")
        if not query:
            return jsonify({"error": "No query provided"}), 400

        response = cached_llm.invoke(query)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask 앱 실행
def start_app():
    os.makedirs(FOLDER_PATH, exist_ok=True)
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()
