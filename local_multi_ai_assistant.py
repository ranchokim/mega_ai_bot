#!/usr/bin/env python3
"""Telegram + Ollama + Open Interpreter 멀티 로컬 AI 비서."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
import importlib
import json
import math
import hashlib
import re
import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class ModelProfile:
    name: str
    role: str
    description: str
    recommended: bool = False


MODEL_PROFILES: Dict[str, ModelProfile] = {
    "fast": ModelProfile(
        name="qwen3:latest",
        role="빠른 일반 대화",
        description="5.2GB급으로 응답속도/품질 균형이 좋아 기본 엔진으로 추천",
        recommended=True,
    ),
    "general": ModelProfile(
        name="llama3.1:8b",
        role="일반 지식/작성",
        description="4.9GB급으로 안정적인 품질",
    ),
    "code": ModelProfile(
        name="qwen3-coder:30b",
        role="코드/디버깅",
        description="고품질 코드 모델(느릴 수 있음)",
    ),
    "reason": ModelProfile(
        name="deepseek-r1:32b",
        role="고난도 추론",
        description="추론 품질 우수(매우 느릴 수 있음)",
    ),
    "fallback": ModelProfile(
        name="phi3:latest",
        role="긴급 경량 대체",
        description="2.2GB급으로 리소스 부족 시 사용",
    ),
}


class BotConfig:
    token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    ollama_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "1800"))
    max_context_tokens: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

    enable_open_interpreter: bool = os.getenv("ENABLE_OPEN_INTERPRETER", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    oi_command_template: str = os.getenv(
        "OI_COMMAND_TEMPLATE",
        'interpreter --yes --offline --message {prompt}',
    )
    oi_fallback_command_template: str = os.getenv(
        "OI_FALLBACK_COMMAND_TEMPLATE",
        'python3 -m interpreter --yes --offline --message {prompt}',
    )
    oi_timeout_seconds: int = int(os.getenv("OI_TIMEOUT_SECONDS", "1800"))
    oi_output_limit: int = int(os.getenv("OI_OUTPUT_LIMIT", "3500"))
    oi_model: str = os.getenv("OI_MODEL", MODEL_PROFILES["fast"].name)
    multi_max_chars_per_stage: int = int(os.getenv("MULTI_MAX_CHARS_PER_STAGE", "1400"))
    workspace_dir: str = os.getenv("CHAIN_WORKSPACE_DIR", "workspace_steps")
    memory_reset_hour: int = int(os.getenv("MEMORY_RESET_HOUR", "3"))
    memory_timezone: str = os.getenv("MEMORY_TIMEZONE", "Asia/Seoul")
    memory_max_entries: int = int(os.getenv("MEMORY_MAX_ENTRIES", "30"))
    rag_db_path: str = os.getenv("RAG_DB_PATH", os.path.join(workspace_dir, "memory_rag.sqlite3"))
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    rag_embed_dim: int = int(os.getenv("RAG_EMBED_DIM", "256"))


def tg_api(method: str, payload: dict) -> dict:
    url = f"https://api.telegram.org/bot{BotConfig.token}/{method}"
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data}")
    return data["result"]


def send_message(chat_id: int, text: str, reply_to: Optional[int] = None) -> dict:
    payload = {"chat_id": chat_id, "text": text}
    if reply_to is not None:
        payload["reply_to_message_id"] = reply_to
    return tg_api("sendMessage", payload)


def edit_message(chat_id: int, message_id: int, text: str) -> None:
    payload = {"chat_id": chat_id, "message_id": message_id, "text": text}
    tg_api("editMessageText", payload)


def get_updates(offset: Optional[int]) -> List[dict]:
    payload = {"timeout": 30, "allowed_updates": ["message"]}
    if offset is not None:
        payload["offset"] = offset
    return tg_api("getUpdates", payload)


def route_task(text: str) -> Tuple[str, str]:
    lowered = text.lower().strip()

    if lowered.startswith("/oi") or lowered.startswith("/open_interpreter"):
        cleaned = text.split(" ", 1)[1].strip() if " " in text else ""
        return "oi", cleaned

    if lowered.startswith("/code") or any(k in lowered for k in ["코드", "bug", "debug", "python", "js", "sql"]):
        return "ollama_code", text.replace("/code", "", 1).strip()
    if lowered.startswith("/reason") or any(k in lowered for k in ["추론", "증명", "논리", "reason"]):
        return "ollama_reason", text.replace("/reason", "", 1).strip()
    if lowered.startswith("/fast"):
        return "ollama_fast", text.replace("/fast", "", 1).strip()
    if lowered.startswith("/general"):
        return "ollama_general", text.replace("/general", "", 1).strip()

    return "ollama_fast", text


def generate_with_ollama(model: str, prompt: str, system_prompt: Optional[str] = None) -> str:
    if system_prompt is None:
        system_prompt = (
            "당신은 한국어 우선의 로컬 AI 비서다.\n"
            "- 답변은 실행 가능한 단계로 제시한다.\n"
            "- 장문 작업은 핵심 요약을 먼저 제공한다.\n"
            "- 불확실하면 가정 사항을 명확히 밝힌다."
        )
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "num_ctx": BotConfig.max_context_tokens,
            "temperature": 0.4,
        },
        "keep_alive": "10m",
    }

    url = f"{BotConfig.ollama_url}/api/chat"
    response = requests.post(url, json=payload, timeout=BotConfig.timeout_seconds)
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "(응답이 비어 있습니다)")


def generate_with_open_interpreter(prompt: str) -> str:
    if not BotConfig.enable_open_interpreter:
        return "Open Interpreter 기능이 비활성화되어 있습니다. (.env에서 ENABLE_OPEN_INTERPRETER=true 설정)"

    # 1) Python API 경로 우선 사용
    try:
        oi_module = importlib.import_module("interpreter")
        oi = oi_module.interpreter
        oi.llm.model = f"ollama/{BotConfig.oi_model}"
        oi.llm.api_base = BotConfig.ollama_url
        oi.llm.supports_functions = False
        oi.auto_run = True

        result = oi.chat(prompt)
        if isinstance(result, str):
            output = result.strip()
        elif isinstance(result, list):
            chunks: List[str] = []
            for item in result:
                if isinstance(item, dict):
                    content = item.get("content")
                    if isinstance(content, str) and content.strip():
                        chunks.append(content.strip())
            output = "\n".join(chunks).strip()
        else:
            output = str(result).strip()

        if not output:
            output = "(Open Interpreter 출력 없음)"

        if len(output) > BotConfig.oi_output_limit:
            output = output[: BotConfig.oi_output_limit] + "\n\n... (출력 잘림)"
        return output
    except Exception as api_exc:
        api_error = str(api_exc)

    # 2) CLI fallback 경로
    command_templates = [
        BotConfig.oi_command_template,
        BotConfig.oi_fallback_command_template,
    ]
    errors: List[str] = [f"python-api 실패: {api_error}"]

    for template in command_templates:
        formatted = template.format(prompt=prompt)
        argv = shlex.split(formatted)
        try:
            completed = subprocess.run(
                argv,
                text=True,
                capture_output=True,
                timeout=BotConfig.oi_timeout_seconds,
            )
        except FileNotFoundError as exc:
            errors.append(f"{' '.join(argv[:3])}... 실행 파일 없음: {exc}")
            continue

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        if completed.returncode == 0:
            combined = stdout if stdout else "(Open Interpreter 출력 없음)"
            if stderr:
                combined += f"\n\n[stderr]\n{stderr}"
            if len(combined) > BotConfig.oi_output_limit:
                combined = combined[: BotConfig.oi_output_limit] + "\n\n... (출력 잘림)"
            return combined

        snippet_out = stdout[:300] if stdout else "stdout 없음"
        snippet_err = stderr[:300] if stderr else "stderr 없음"
        errors.append(
            f"실패(code={completed.returncode}) cmd={' '.join(argv[:5])}... | {snippet_out} | {snippet_err}"
        )

    raise RuntimeError(
        "Open Interpreter 실행 실패. 시도한 명령 모두 실패했습니다.\n"
        + "\n".join(f"- {item}" for item in errors)
    )


def format_models() -> str:
    lines = ["사용 가능한 멀티 비서 프로필:"]
    for key, profile in MODEL_PROFILES.items():
        tag = " (기본)" if profile.recommended else ""
        lines.append(f"- {key}: {profile.name}{tag} | {profile.role}")
    lines.append("- oi: Open Interpreter CLI 실행")
    lines.append("\n명령 예시: /fast 질문, /code 코드질문, /reason 복잡한추론, /oi 쉘작업")
    lines.append("일반/코드/추론 질의는 모델들이 역할을 나눠 순차 협업 후 최종 답을 생성합니다.")
    return "\n".join(lines)


def summarize_for_chain(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) <= BotConfig.multi_max_chars_per_stage:
        return cleaned
    return cleaned[: BotConfig.multi_max_chars_per_stage] + "\n...(중간 출력 생략)"


def get_memory_bucket_date(now: Optional[datetime] = None) -> str:
    tz = ZoneInfo(BotConfig.memory_timezone)
    current = now.astimezone(tz) if now else datetime.now(tz)
    if current.hour < BotConfig.memory_reset_hour:
        current = current - timedelta(days=1)
    return current.strftime("%Y-%m-%d")


def get_memory_file_path(chat_id: int, now: Optional[datetime] = None) -> str:
    bucket = get_memory_bucket_date(now)
    memory_dir = os.path.join(BotConfig.workspace_dir, "memory")
    os.makedirs(memory_dir, exist_ok=True)
    return os.path.join(memory_dir, f"{chat_id}_{bucket}.jsonl")


def init_rag_db() -> None:
    os.makedirs(BotConfig.workspace_dir, exist_ok=True)
    conn = sqlite3.connect(BotConfig.rag_db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                ts TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def tokenize(text: str) -> List[str]:
    return [tok for tok in re.split(r"[^0-9A-Za-z가-힣_]+", text.lower()) if tok]


def embed_text(text: str) -> List[float]:
    vec = [0.0] * BotConfig.rag_embed_dim
    tokens = tokenize(text)
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % BotConfig.rag_embed_dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine_similarity(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def rag_add_memory(chat_id: int, role: str, text: str, ts: str) -> None:
    init_rag_db()
    emb = embed_text(text)
    conn = sqlite3.connect(BotConfig.rag_db_path)
    try:
        conn.execute(
            "INSERT INTO memories(chat_id, ts, role, text, embedding) VALUES (?, ?, ?, ?, ?)",
            (chat_id, ts, role, text, json.dumps(emb)),
        )
        conn.commit()
    finally:
        conn.close()


def rag_search(chat_id: int, query: str, top_k: Optional[int] = None) -> List[dict]:
    init_rag_db()
    k = top_k or BotConfig.rag_top_k
    query_emb = embed_text(query)
    conn = sqlite3.connect(BotConfig.rag_db_path)
    try:
        rows = conn.execute(
            "SELECT ts, role, text, embedding FROM memories WHERE chat_id = ? ORDER BY id DESC LIMIT 1000",
            (chat_id,),
        ).fetchall()
    finally:
        conn.close()

    scored: List[dict] = []
    for ts, role, text, emb_json in rows:
        try:
            emb = json.loads(emb_json)
        except json.JSONDecodeError:
            continue
        score = cosine_similarity(query_emb, emb)
        scored.append({"score": score, "ts": ts, "role": role, "text": text})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


def append_daily_memory(chat_id: int, role: str, text: str) -> None:
    timestamp = datetime.now(ZoneInfo(BotConfig.memory_timezone)).isoformat()
    payload = {
        "ts": timestamp,
        "role": role,
        "text": text.strip(),
    }
    path = get_memory_file_path(chat_id)
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    rag_add_memory(chat_id, role, text.strip(), timestamp)


def load_daily_memory(chat_id: int) -> List[dict]:
    path = get_memory_file_path(chat_id)
    if not os.path.exists(path):
        return []
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if len(rows) > BotConfig.memory_max_entries:
        rows = rows[-BotConfig.memory_max_entries :]
    return rows


def format_daily_memory(chat_id: int, query: str) -> str:
    rows = load_daily_memory(chat_id)
    similar_rows = rag_search(chat_id, query, BotConfig.rag_top_k)
    if not rows and not similar_rows:
        return "(오늘 저장된 대화 기억 및 유사 기억 없음)"
    lines: List[str] = []
    lines.append("[오늘 대화 요약]")
    for row in rows[-BotConfig.memory_max_entries :]:
        role = row.get("role", "unknown")
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"- {role}: {summarize_for_chain(text)}")
    lines.append("\n[유사 과거 기억(RAG)]")
    for item in similar_rows:
        if item["score"] <= 0:
            continue
        lines.append(f"- {item['role']}({item['ts']}): {summarize_for_chain(item['text'])}")
    return "\n".join(lines)


def parse_json_object(raw_text: str) -> Optional[dict]:
    raw = raw_text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


def planner_tool_decision(task: str, memory_context: str, planner: ModelProfile) -> Tuple[bool, str, str]:
    decision_system = (
        "역할: Planner with Tool Choice.\n"
        "Open Interpreter 도구 사용 여부를 스스로 결정하라.\n"
        "JSON만 출력: {\"use_oi\": bool, \"reason\": str, \"oi_task\": str}\n"
        "파일 정리/쉘명령/실제 시스템조작이 필요하면 use_oi=true."
    )
    decision_prompt = f"[기억]\n{memory_context}\n\n[요청]\n{task}"
    raw = generate_with_ollama(planner.name, decision_prompt, decision_system)
    parsed = parse_json_object(raw)
    if not parsed:
        heuristic = any(k in task.lower() for k in ["폴더", "파일", "터미널", "정리", "삭제", "실행", "shell"])
        return heuristic, "heuristic fallback", task
    use_oi = bool(parsed.get("use_oi", False))
    reason = str(parsed.get("reason", "")).strip() or "no reason"
    oi_task = str(parsed.get("oi_task", "")).strip() or task
    return use_oi, reason, oi_task


def review_needs_retry(review_text: str) -> bool:
    lowered = review_text.lower()
    signals = ["치명", "심각", "누락", "오류", "재작성", "수정 필요", "틀림", "불충분"]
    return any(sig in lowered for sig in signals)


def _safe_model_name(model_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in model_name)


def save_chain_stage_result(request_id: str, stage_idx: int, stage_name: str, model_name: str, content: str) -> str:
    os.makedirs(BotConfig.workspace_dir, exist_ok=True)
    safe_model = _safe_model_name(model_name)
    file_name = f"{request_id}_{stage_idx:02d}_{stage_name}_{safe_model}.md"
    file_path = os.path.join(BotConfig.workspace_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as fp:
        fp.write(content.strip() + "\n")
    return file_path


def run_multi_model_chain(
    chat_id: int,
    status_msg_id: int,
    task: str,
    specialist: ModelProfile,
    request_id: str,
) -> str:
    planner = MODEL_PROFILES["fast"]
    verifier = MODEL_PROFILES["general"]
    synthesizer = MODEL_PROFILES["general"]

    planner_system = (
        "역할: 작업 계획가.\n"
        "요청과 오늘의 대화 기억을 분석해서 목표/가정/필요 산출물/실행 순서를 간단히 정리하라.\n"
        "최대 8줄로 작성."
    )
    daily_memory = format_daily_memory(chat_id, task)
    use_oi, tool_reason, oi_task = planner_tool_decision(task, daily_memory, planner)
    tool_output = "(도구 미사용)"
    if use_oi:
        try:
            tool_output = generate_with_open_interpreter(oi_task)
        except Exception as tool_exc:
            tool_output = f"(도구 실행 실패: {tool_exc})"
        save_chain_stage_result(request_id, 0, "tool", BotConfig.oi_model, tool_output)
    planner_prompt = (
        f"[오늘의 대화 기억]\n{daily_memory}\n\n"
        f"[도구 선택]\nuse_oi={use_oi}, reason={tool_reason}\n\n"
        f"[도구 실행 결과]\n{summarize_for_chain(tool_output)}\n\n"
        f"[사용자 요청]\n{task}"
    )
    plan = generate_with_ollama(planner.name, planner_prompt, planner_system)
    plan_path = save_chain_stage_result(request_id, 1, "plan", planner.name, plan)
    edit_message(chat_id, status_msg_id, f"진행중 1/4: 계획 수립 완료 ({planner.name})")

    specialist_system = (
        f"역할: {specialist.role} 전문가.\n"
        "입력된 계획을 바탕으로 핵심 결과를 작성하라.\n"
        "필요시 근거/코드/단계를 포함하라."
    )
    specialist_prompt = f"[사용자 요청]\n{task}\n\n[계획]\n{summarize_for_chain(plan)}"
    specialist_answer = generate_with_ollama(specialist.name, specialist_prompt, specialist_system)
    specialist_path = save_chain_stage_result(request_id, 2, "specialist", specialist.name, specialist_answer)
    edit_message(chat_id, status_msg_id, f"진행중 2/4: 전문가 초안 완료 ({specialist.name})")

    verifier_system = (
        "역할: 검토자.\n"
        "초안의 오류 가능성/누락/위험 요소를 간단히 지적하고 개선안을 제시하라.\n"
        "출력은 '점검결과' 섹션으로만 작성."
    )
    verifier_prompt = (
        f"[사용자 요청]\n{task}\n\n[전문가 초안]\n{summarize_for_chain(specialist_answer)}\n\n"
        f"[도구 결과]\n{summarize_for_chain(tool_output)}"
    )
    review = generate_with_ollama(verifier.name, verifier_prompt, verifier_system)
    review_path = save_chain_stage_result(request_id, 3, "review", verifier.name, review)
    edit_message(chat_id, status_msg_id, f"진행중 3/4: 검토 완료 ({verifier.name})")

    if review_needs_retry(review):
        retry_prompt = (
            f"[사용자 요청]\n{task}\n\n"
            f"[기존 초안]\n{summarize_for_chain(specialist_answer)}\n\n"
            f"[검토 피드백]\n{summarize_for_chain(review)}\n\n"
            f"[도구 결과]\n{summarize_for_chain(tool_output)}"
        )
        specialist_answer = generate_with_ollama(specialist.name, retry_prompt, specialist_system)
        save_chain_stage_result(request_id, 5, "specialist_retry", specialist.name, specialist_answer)
        review = generate_with_ollama(verifier.name, retry_prompt, verifier_system)
        save_chain_stage_result(request_id, 6, "review_retry", verifier.name, review)

    synth_system = (
        "역할: 최종 편집자.\n"
        "계획/초안/검토를 종합하여 최종 답변을 한국어로 작성하라.\n"
        "형식: 1) 핵심요약 2) 실행단계 3) 주의사항.\n"
        "불확실한 정보는 '가정'으로 표시."
    )
    synth_prompt = (
        f"[사용자 요청]\n{task}\n\n"
        f"[계획]\n{summarize_for_chain(plan)}\n\n"
        f"[전문가 초안]\n{summarize_for_chain(specialist_answer)}\n\n"
        f"[검토 결과]\n{summarize_for_chain(review)}\n\n"
        f"[도구 실행 결과]\n{summarize_for_chain(tool_output)}"
    )
    final_answer = generate_with_ollama(synthesizer.name, synth_prompt, synth_system)
    final_path = save_chain_stage_result(request_id, 4, "synthesis", synthesizer.name, final_answer)
    edit_message(chat_id, status_msg_id, f"진행중 4/4: 최종 합성 완료 ({synthesizer.name})")
    send_message(
        chat_id,
        "단계별 결과 파일 저장 완료:\n"
        f"1) {plan_path}\n"
        f"2) {specialist_path}\n"
        f"3) {review_path}\n"
        f"4) {final_path}",
    )
    return final_answer


def handle_ollama(chat_id: int, msg_id: int, task: str, profile: ModelProfile) -> None:
    status_msg = send_message(
        chat_id,
        "진행중: 멀티 모델 협업 파이프라인 시작\n"
        "순차 실행으로 자원 과부하를 줄이며 답변을 합성합니다.",
        reply_to=msg_id,
    )
    status_msg_id = status_msg["message_id"]

    start = time.time()
    request_id = f"{int(start)}_{chat_id}_{msg_id}"
    append_daily_memory(chat_id, "user", task)
    try:
        answer = run_multi_model_chain(chat_id, status_msg_id, task, profile, request_id)
        elapsed = time.time() - start
        edit_message(chat_id, status_msg_id, f"완료: {elapsed:.1f}초\n엔진: Ollama 멀티 모델 순차 협업")
        send_message(chat_id, answer, reply_to=msg_id)
        append_daily_memory(chat_id, "assistant", answer)
    except Exception as exc:
        fallback = MODEL_PROFILES["fallback"]
        try:
            edit_message(chat_id, status_msg_id, f"오류: 협업 파이프라인 실패 → {fallback.name} 단일 응답 재시도")
            answer = generate_with_ollama(fallback.name, task)
            send_message(chat_id, f"(fallback:{fallback.name})\n\n{answer}", reply_to=msg_id)
        except Exception as exc2:
            send_message(chat_id, f"요청 처리 실패: {exc}\nfallback 실패: {exc2}", reply_to=msg_id)


def handle_oi(chat_id: int, msg_id: int, task: str) -> None:
    status_msg = send_message(
        chat_id,
        "진행중: Open Interpreter 작업 시작\n환경 점검 후 실행합니다...",
        reply_to=msg_id,
    )
    status_msg_id = status_msg["message_id"]

    start = time.time()
    try:
        answer = generate_with_open_interpreter(task)
        elapsed = time.time() - start
        edit_message(chat_id, status_msg_id, f"완료: {elapsed:.1f}초\n엔진: Open Interpreter")
        send_message(chat_id, answer, reply_to=msg_id)
    except Exception as exc:
        edit_message(chat_id, status_msg_id, "실패: Open Interpreter 실행 중 오류")
        send_message(chat_id, f"Open Interpreter 오류: {exc}", reply_to=msg_id)


def handle_command(chat_id: int, msg_id: int, text: str) -> None:
    cmd = text.strip()
    if cmd == "/start":
        send_message(
            chat_id,
            "로컬 멀티 AI 비서가 시작되었습니다.\n"
            "- Ollama: /fast /general /code /reason\n"
            "- Open Interpreter: /oi\n"
            "- 모델목록: /models\n"
            "- 상태: /status\n"
            "일반 텍스트는 멀티 모델 순차 협업으로 처리합니다.",
            reply_to=msg_id,
        )
        return

    if cmd == "/models":
        send_message(chat_id, format_models(), reply_to=msg_id)
        return

    if cmd == "/status":
        oi_enabled = "ON" if BotConfig.enable_open_interpreter else "OFF"
        send_message(
            chat_id,
            f"상태: 정상 동작 중\n- Telegram polling: ON\n- Ollama URL: {BotConfig.ollama_url}\n- Open Interpreter: {oi_enabled}",
            reply_to=msg_id,
        )
        return

    route, task = route_task(text)
    if not task.strip():
        send_message(chat_id, "질문 내용을 함께 보내주세요. 예: /oi 현재 디렉터리 파일 요약", reply_to=msg_id)
        return

    if route == "oi":
        handle_oi(chat_id, msg_id, task)
        return

    profile_key = route.replace("ollama_", "")
    profile = MODEL_PROFILES[profile_key]
    handle_ollama(chat_id, msg_id, task, profile)


def main() -> None:
    if not BotConfig.token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN 환경변수가 필요합니다.")

    print("[boot] Telegram polling bot start")
    print(f"[boot] Ollama URL: {BotConfig.ollama_url}")
    print(f"[boot] Open Interpreter enabled: {BotConfig.enable_open_interpreter}")

    offset: Optional[int] = None

    while True:
        try:
            updates = get_updates(offset)
            for update in updates:
                offset = update["update_id"] + 1
                message = update.get("message")
                if not message:
                    continue

                chat_id = message["chat"]["id"]
                msg_id = message["message_id"]
                text = message.get("text", "").strip()

                if not text:
                    send_message(chat_id, "텍스트 메시지만 처리할 수 있습니다.", reply_to=msg_id)
                    continue

                handle_command(chat_id, msg_id, text)
        except requests.RequestException as net_err:
            print(f"[warn] network error: {net_err}")
            time.sleep(3)
        except Exception as err:
            print(f"[warn] loop error: {err}")
            time.sleep(3)


if __name__ == "__main__":
    main()
