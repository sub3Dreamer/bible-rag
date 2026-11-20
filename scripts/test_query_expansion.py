import json

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from app.rag.hybrid_retriever import HybridRetriever


def main():
    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
        base_url="http://127.0.0.1:11434",
        format="json",
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
당신은 성경 검색을 위한 질의 확장 도우미입니다.

사용자의 질문을 성경 본문 검색에 적합한 문장으로 바꾸세요.

규칙:
1. 원래 질문의 의도를 유지하세요.
2. 사건, 인물, 행동 등 구체적인 특징을 활용하세요.
3. 확실하지 않은 사실은 만들어내지 마세요.
4. 성경 책 이름이나 장절을 추측하여 추가하지 마세요.
5. 답변을 생성하지 말고 검색 질의만 생성하세요.
6. 한국어로 작성하세요.

JSON 형식:
{{
    "expanded_query": "확장된 검색 질의"
}}
"""
        ),
        (
            "human",
            "{question}"
        ),
    ])

    chain = prompt | llm
    retriever = HybridRetriever()

    question = "탕자의 비유는 어디에 나오는가?"

    response = chain.invoke({"question": question})
    data = json.loads(response.content)

    expanded_query = data["expanded_query"]

    print("Original Query:", question)
    print("Expanded Query:", expanded_query)

    results = retriever.search(
        question=expanded_query,
        translation_id="krv",
        k=5,
    )

    print("\nSearch Results:")

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['chunk_id']}")


if __name__ == "__main__":
    main()