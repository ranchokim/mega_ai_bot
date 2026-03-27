# Multi Local AI Telegram Assistant (Ollama + Open Interpreter)

5800X + RAM 32GB + RTX 3080 10GB 환경에서 여러 로컬 모델을 라우팅하고, 필요 시 Open Interpreter까지 텔레그램으로 호출하는 비서입니다.

## 핵심 기능
- 텔레그램으로 명령/질문 입력
- 진행 상태를 텔레그램 메시지로 즉시 공유
- 모델 역할 분담 + 순차 실행 기반 협업 응답 (Ollama)
  - 1단계 계획: `qwen3:latest`
  - 2단계 전문가: 작업 유형별 모델(`qwen3-coder:30b`, `deepseek-r1:32b`, `llama3.1:8b` 등)
  - 3단계 검토: `llama3.1:8b` (Verifier)
  - 4단계 합성: `llama3.1:8b`
  - 실패 시 fallback 단일 응답
- `/oi` 명령으로 Open Interpreter CLI 작업 실행
- Planner가 필요 시 Open Interpreter를 자동 선택(Agentic tool calling 유사 흐름)

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
OI_MODEL=qwen3:latest
OI_COMMAND_TEMPLATE=interpreter --yes --offline --message {prompt}
OI_FALLBACK_COMMAND_TEMPLATE=python3 -m interpreter --yes --offline --message {prompt}
OI_TIMEOUT_SECONDS=1800
OI_OUTPUT_LIMIT=3500
MULTI_MAX_CHARS_PER_STAGE=1400
CHAIN_WORKSPACE_DIR=workspace_steps
MEMORY_RESET_HOUR=3
MEMORY_TIMEZONE=Asia/Seoul
MEMORY_MAX_ENTRIES=30
RAG_DB_PATH=workspace_steps/memory_rag.sqlite3
RAG_TOP_K=4
RAG_EMBED_DIM=256
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

명령어 없이 일반 텍스트를 보내면 `fast` 전문가를 포함한 **멀티 모델 순차 협업**으로 처리됩니다.

## Open Interpreter 관련 주의
- `/oi`는 실제 로컬 명령을 실행할 수 있으므로 운영 환경에서 접근 가능한 텔레그램 사용자/채팅을 제한하세요.
- 필요 시 `ENABLE_OPEN_INTERPRETER=false`로 즉시 비활성화할 수 있습니다.
- `/oi` 실패 시 기본 명령(`OI_COMMAND_TEMPLATE`) 이후 fallback 명령(`OI_FALLBACK_COMMAND_TEMPLATE`)을 순차 시도합니다.
- Open Interpreter Python API 사용 시 내부 설정은 아래처럼 강제됩니다:
  - `interpreter.llm.model = "ollama/{OI_MODEL}"`
  - `interpreter.llm.api_base = OLLAMA_BASE_URL`
  - `interpreter.llm.supports_functions = False`
  - `interpreter.auto_run = True`
- 일반 질의에서도 Planner가 도구가 필요하다고 판단하면 `/oi` 없이 자동 실행할 수 있습니다.

## 단계별 결과 파일 저장
- 멀티 모델 체인은 각 단계 결과를 워크스페이스 디렉토리에 파일로 저장합니다.
- 기본 디렉토리: `workspace_steps` (환경변수 `CHAIN_WORKSPACE_DIR`로 변경 가능)
- 파일명 형식:
  - `{request_id}_01_plan_{model}.md`
  - `{request_id}_02_specialist_{model}.md`
  - `{request_id}_03_review_{model}.md`
  - `{request_id}_04_synthesis_{model}.md`

## 일 단위 대화 기억(03:00 초기화)
- 각 채팅의 대화를 `workspace_steps/memory/{chat_id}_{bucket_date}.jsonl`에 저장합니다.
- 계획 단계는 오늘의 저장된 기억을 함께 읽어 참고합니다.
- `MEMORY_RESET_HOUR=3` 기준으로 매일 새벽 3시에 새로운 버킷으로 넘어가며, 이전 기억은 자동으로 참조 대상에서 제외됩니다.
- 시간대는 `MEMORY_TIMEZONE`으로 제어합니다(기본 `Asia/Seoul`).

## 장기 기억 검색(RAG)
- 대화는 임베딩 형태로 SQLite 기반 경량 벡터 저장소(`RAG_DB_PATH`)에도 저장됩니다.
- 계획 단계에서 요청과 의미적으로 유사한 과거 기억 상위 `RAG_TOP_K`개를 검색해 프롬프트에 주입합니다.
- 임베딩 차원은 `RAG_EMBED_DIM`으로 조정 가능합니다(기본 256).

## 순환형 보정(Reviewer → Specialist Retry)
- 검토 결과가 심각한 오류/누락으로 판단되면, 검토 피드백을 기반으로 전문가 단계를 자동 1회 재실행합니다.
- 재실행 산출물도 워크스페이스 파일로 저장됩니다.
