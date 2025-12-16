from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# 1. 모델 정의 (로컬 Llama.cpp 서버 연결)
local_llm = ChatOpenAI(
    base_url="http://localhost:8080/v1", # Llama.cpp 서버 주소
    api_key="lm-studio", # 아무 값이나 입력 (로컬이라 검증 안 함)
    model="llama-3-8b-instruct", # 서버 실행 시 로드한 모델과 논리적 매칭
    temperature=0
)

# 2. 상태(State) 정의
class State(TypedDict):
    messages: List[str]

# 3. 노드(Node) 정의
def chatbot(state: State):
    response = local_llm.invoke(state["messages"])
    return {"messages": [response.content]}

# 4. 그래프(Graph) 구성
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.add_edge("chatbot", END)

app = graph_builder.compile()

# 5. 실행 테스트
result = app.invoke({"messages": ["안녕하세요! AI 교육 도우미입니다."]})
print(result["messages"][-1])