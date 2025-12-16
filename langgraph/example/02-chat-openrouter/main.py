import os
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
# 1. 환경 설정 및 모델 초기화
# ==========================================
# 실제 서비스 배포 시 API Key는 환경변수로 관리하는 것이 필수입니다.
os.environ["OPENROUTER_API_KEY"] = "여기에_OPENROUTER_API_KEY_입력"

# OpenRouter를 사용하기 위해 ChatOpenAI 클라이언트를 설정합니다.
llm = ChatOpenAI(
    # [옵션] 사용할 모델명: OpenRouter에서 지원하는 모델 ID (예: meta-llama/llama-3.1-8b-instruct)
    model="meta-llama/llama-3.1-8b-instruct",
    
    # [중요] OpenRouter의 API 엔드포인트 설정
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    
    # [옵션] temperature: 0.0 ~ 1.0 (0에 가까울수록 사실적/정해진 답, 1에 가까울수록 창의적)
    # 교육 에이전트는 정확도가 중요하므로 0.3~0.5 정도를 추천합니다.
    temperature=0.3,
    
    # [옵션] max_tokens: 생성할 최대 토큰 수 (비용 관리 및 답변 길이 제한)
    max_tokens=512,
)

# ==========================================
# 2. 상태(State) 정의
# ==========================================
# LangGraph의 핵심인 '상태'입니다. 대화가 진행되면서 이 상태 객체가 계속 업데이트됩니다.
# messages: 대화 내역을 저장하는 리스트. add_messages는 새 메시지를 기존 리스트에 '추가(append)'하는 리듀서 함수입니다.
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ==========================================
# 3. 노드(Node) 함수 정의
# ==========================================
# 챗봇의 두뇌 역할을 하는 함수입니다. 현재 상태를 받아 LLM의 답변을 반환합니다.
def chatbot_node(state: State):
    # 현재까지의 대화 기록(state["messages"])을 LLM에 전달하여 답변을 생성합니다.
    return {"messages": [llm.invoke(state["messages"])]}

# ==========================================
# 4. 그래프(Graph) 구성
# ==========================================
# 워크플로우 그래프를 생성합니다.
graph_builder = StateGraph(State)

# 노드 추가: 'chatbot'이라는 이름의 노드를 등록하고 chatbot_node 함수를 연결합니다.
graph_builder.add_node("chatbot", chatbot_node)

# 엣지(Edge) 연결: 시작(START)하자마자 'chatbot' 노드로 이동합니다.
graph_builder.add_edge(START, "chatbot")

# 엣지 연결: 'chatbot' 노드가 끝나면 종료(END)합니다. (이번 턴 종료)
graph_builder.add_edge("chatbot", END)

# [기능] 메모리(Checkpointer) 설정
# 대화가 이어지려면 이전 내용을 기억해야 합니다. 
# MemorySaver는 메모리(RAM)에 대화 상태를 저장합니다. (앱 재시작 시 초기화됨)
# 실제 서비스에서는 SqliteSaver나 PostgresSaver를 사용하여 DB에 저장합니다.
memory = MemorySaver()

# 그래프 컴파일: 이제 실행 가능한 어플리케이션이 됩니다.
app = graph_builder.compile(checkpointer=memory)

# ==========================================
# 5. 실행 및 테스트 (채팅 루프)
# ==========================================
def main():
    print("🤖 AI 교육 에이전트 프로토타입 (종료하려면 'q' 입력)")
    
    # [기능] 시스템 프롬프트 설정 (페르소나 부여)
    # 챗봇에게 역할을 부여합니다. 교육 에이전트로서의 톤앤매너를 설정합니다.
    system_prompt = SystemMessage(content="너는 친절하고 인내심 강한 컴퓨터 공학 튜터야. 학생의 질문에 이해하기 쉽게 설명해줘.")
    
    # [기능] Thread ID 설정
    # 사용자별 혹은 대화방별로 기억을 구분하기 위한 ID입니다.
    # 이 ID가 같으면 이전 대화를 기억합니다.
    config = {"configurable": {"thread_id": "student_1"}}
    
    # 초기 시스템 메시지 주입 (사용자에게 보이지 않지만 문맥에 포함됨)
    # 사용자가 처음 실행할 때만 주입하는 로직이 필요하지만, 여기선 테스트를 위해 매번 실행 전 확인
    # (실제론 checkpointer 확인 후 주입)
    
    # 대화 루프
    while True:
        user_input = input("\n나: ")
        if user_input.lower() in ["q", "quit", "exit"]:
            print("대화를 종료합니다.")
            break
            
        # 사용자의 입력을 메시지 목록에 추가
        # 첫 턴에는 시스템 프롬프트도 함께 전송하여 페르소나를 잡게 할 수 있습니다.
        input_messages = [HumanMessage(content=user_input)]
        
        # 만약 대화의 시작(메시지가 없는 경우)이라면 시스템 메시지를 앞에 추가하는 로직을 짤 수도 있습니다.
        # 여기서는 간단하게 invoke 시 system_prompt를 매번 포함하거나, 
        # state 관리에 system_message를 영구적으로 넣는 방법 등이 있습니다.
        # 가장 쉬운 방법: 그래프 실행 시 input에 같이 전달 (이전 기록은 memory가 처리)
        
        events = app.stream(
            {"messages": [system_prompt, HumanMessage(content=user_input)]}, 
            config=config
        )

        print("튜터: ", end="", flush=True)
        # 스트리밍 출력: 답변이 생성되는 대로 한 글자씩 출력
        for event in events:
            if "chatbot" in event:
                response_msg = event["chatbot"]["messages"][-1]
                print(response_msg.content, end="", flush=True)
        print() # 줄바꿈

if __name__ == "__main__":
    main()