from interpreter import interpreter

interpreter.llm.model = "ollama/mixtral"
interpreter.llm.api_base = "http://127.0.0.1:11434"
interpreter.llm.supports_functions = False

# 핵심 추가 사항: 실행 확인(y/n) 절차를 완전히 끄고 즉시 실행합니다.
interpreter.auto_run = True

interpreter.chat()
