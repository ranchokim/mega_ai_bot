# Multi Local AI Telegram Assistant (Ollama + Open Interpreter)

5800X + RAM 32GB + RTX 3080 10GB 환경에서 여러 로컬 모델을 라우팅하고, 필요 시 Open Interpreter까지 텔레그램으로 호출하는 비서입니다.

## 핵심 기능
- 텔레그램으로 명령/질문 입력
- 진행 상태를 텔레그램 메시지로 즉시 공유
- 작업 종류에 따라 모델 자동 선택 (Ollama)
  - 기본: `qwen3:latest`
  - 코드: `qwen3-coder:30b`
  - 고난도 추론: `deepseek-r1:32b`
  - 실패 시 fallback: `phi3:latest`
- `/oi` 명령으로 Open Interpreter CLI 작업 실행

## 왜 이 구성이 실용적인가
- VRAM 10GB에서는 30B/32B를 기본 모델로 두면 지연이 큽니다.
- 따라서 기본 질의는 5~8B로 처리하고, 고난도 작업만 대형 모델 또는 Open Interpreter로 분기하는 하이브리드가 체감 성능이 좋습니다.

## 설치
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open Interpreter를 쓰려면(선택):
```bash
pip install open-interpreter
```

`.env` 예시:
```env
TELEGRAM_BOT_TOKEN=여기에_토큰
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_TIMEOUT_SECONDS=1800
OLLAMA_NUM_CTX=4096

ENABLE_OPEN_INTERPRETER=true
OI_COMMAND_TEMPLATE=interpreter --yes --offline --message {prompt}
OI_TIMEOUT_SECONDS=1800
OI_OUTPUT_LIMIT=3500
```

> 보안 권장: 토큰이 이미 외부에 노출됐다면 BotFather에서 즉시 재발급(rotate) 하세요.

## 실행
```bash
python3 local_multi_ai_assistant.py
```

## 텔레그램 명령
- `/start` : 도움말
- `/models` : 라우팅 모델 목록
- `/status` : 상태 확인
- `/fast 질문` : 빠른 일반 응답
- `/general 질문` : 일반 모델 지정
- `/code 질문` : 코드 전용 모델
- `/reason 질문` : 추론 전용 모델
- `/oi 작업지시` : Open Interpreter CLI 작업 실행

명령어 없이 일반 텍스트를 보내면 자동으로 fast 라우팅됩니다.

## Open Interpreter 관련 주의
- `/oi`는 실제 로컬 명령을 실행할 수 있으므로 운영 환경에서 접근 가능한 텔레그램 사용자/채팅을 제한하세요.
- 필요 시 `ENABLE_OPEN_INTERPRETER=false`로 즉시 비활성화할 수 있습니다.
