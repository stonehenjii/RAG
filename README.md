# RAG
RAG구축 프로젝트_한국어

기업에서는 보안 문제로 인해 클라우드 사용 등 외부망으로 이어지는 LLM 및 DB사용에 어려움을 겪고 있다. 

이에, 본 프로젝트는 온디바이스 RAG 체제를 구축하여 LLM을 사용할 수 있게 함에 목적을 둔다. 



nlpai-lab-KoE5 한국어 파인튜닝 모델인 E5모델을 활용해 RAG를 구축한다. 
https://huggingface.co/nlpai-lab/KoE5

PDF 파일을 Chroma DB에 벡터화해 적재하고, 문서에 대한 QA시스템을 지원한다.

Ollama(llama3)모델을 llm으로 사용한다. 


cli/srv 형태로, 간단히 두 개의 터미널만 띄우고 server와 client가 QA를 이어갈 수 있게끔 한다. 

1. 사용자가 검색에 필요한 문서 적재
   
   ![image](https://github.com/user-attachments/assets/b0b78d97-066f-4ba9-8bd5-51615bb56e04)

2. 문서에 대한 쿼리를 날림
   
![image](https://github.com/user-attachments/assets/d370c545-9dd5-4d2d-9ecc-93052c381ebd)

3. 서버에 위치한 Vector DB에서 유사도 검색 후 context를 llm에 전달
   
![image](https://github.com/user-attachments/assets/28e34e5e-426e-4de2-b7c0-4f0bc476ceb3)

4. context를 바탕으로 llm이 답변 출력
   
![image](https://github.com/user-attachments/assets/4440960d-925e-49a7-8426-793290ab8428)

