# 공통적으로 사용되는 함수를 정의한 파일
# 예를 들면 .env 파일 읽어오고 필수 설정값 있는지 검사한다거나

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

class APISetting:
    """Holds LLM connection settings and helpers.

    Features:
    - Construct from explicit parts or from environment variables.
    - Validate required settings.
    - Create a configured `ChatOpenAI` instance via `get_llm()`.
    """
    def __init__(self, api_url: str, port: str | None = None, model: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0):
        if port:
            self.base_url = api_url.rstrip('/') + ':' + str(port)
        else:
            self.base_url = api_url
        self.model = model
        self.api_key = api_key
        self.temperature = float(temperature)

    @classmethod
    def from_env(cls, *, url_var='SERVER_URL', port_var='SERVER_PORT',
                 model_var='MODEL', api_key_var='API_KEY', temp_var='TEMPERATURE'):
        """Create APISetting from environment variables (loads .env).

        Raises AttributeError if required values are missing.
        """
        load_dotenv()
        api_url = os.getenv(url_var)
        port = os.getenv(port_var)
        model = os.getenv(model_var)
        api_key = os.getenv(api_key_var)
        temp = os.getenv(temp_var, '')

        missing = [name for name, val in ((url_var, api_url), (port_var, port), (model_var, model)) if not val]
        if missing:
            raise AttributeError(f'Missing environment variables: {missing}')

        temperature = float(temp) if temp != '' else 0.0
        return cls(api_url=api_url, port=port, model=model, api_key=api_key, temperature=temperature)

    def validate(self):
        if not self.base_url:
            raise ValueError('base_url must be set')
        if not self.model:
            raise ValueError('model must be set')

    def get_llm(self):
        """Return a configured `ChatOpenAI` instance.

        This method validates settings and passes sensible defaults.
        """
        self.validate()
        llm = ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key or '',
            model=self.model,
            temperature=self.temperature,
        )
        return llm

    def to_dict(self):
        return {
            'base_url': self.base_url,
            'model': self.model,
            'temperature': self.temperature,
            'api_key_set': bool(self.api_key),
        }


def get_env(*args):
    # .env 파일 로드하고 환경변수 체크
    # 필수 인자 args에 전달

    essential_vals = ['MODE', 'SERVER_URL', 'SERVER_PORT']
    load_dotenv()

    for val in [_ for _ in args] + essential_vals:
        if os.getenv(val) == None:
            raise AttributeError(f'환경변수에 필수 인자 "{val}" 이 없습니다. ')

def get_llm():
    
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:8888", # Llama.cpp 서버 주소
        api_key="lm-studio", # 아무 값이나 입력 (로컬이라 검증 안 함)
        model="Qwen3-4B-Instruct-2507-IQ4_XS-4.54bpw", # 서버 실행 시 로드한 모델과 논리적 매칭
        temperature=0
    )
    return llm

def run_local_llm():
    #llama-server -m ..\models\Qwen3-4B-Instruct-2507-IQ4_XS-4.54bpw.gguf --host 127.0.0.1 --port 8888
    pass

