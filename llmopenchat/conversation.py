"""UI-independent generation and complete, bounded tool conversation rounds."""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass
from typing import Callable

from .api import MAX_TOOL_ARGUMENT_BYTES, ApiClient, ApiError, StreamEvent, validated_tool_call
from .harness import HarnessError, ToolHarness
from .rag import RagError, RagSession


MAX_TOOL_ROUNDS = 8
MAX_TOOL_CALLS = 16
TOOL_LIMIT_MESSAGE = "Достигнут лимит вызовов инструментов за один запрос. Продолжите новым сообщением."
_CODE_TOOLS = {"list_files", "read_file", "write_file", "search_files", "create_directory"}
_WEB_TOOLS = {"web_fetch", "web_search"}
_MUTATION_TOOLS = {"write_file", "create_directory"}


def _creation_intent(messages: list[dict]) -> bool:
    """Recognize concrete creation requests, keeping examples and pasted code in chat."""
    prompt = next((message.get("content", "") for message in reversed(messages)
                   if message.get("role") == "user"), "")
    if not isinstance(prompt, str) or "```" in prompt or "~~~" in prompt:
        return False
    prompt = prompt.casefold()
    if re.search(r"\b(?:только\s+в\s+чате|без\s+сохранения|не\s+(?:создавай|записывай|сохраняй|создавать|записывать|сохранять)\s+файл\w*|(?:do not|don't)\s+(?:create|write|save)\s+files?|chat only|only in chat|without saving)\b", prompt):
        return False
    if re.search(r"\b(?:пример\w*|объясн\w*|объясни|расскаж\w*|поясн\w*|explain|describe|examples?|samples?)\b", prompt):
        return False
    verb = re.search(r"\b(?:напиши|написать|создай|создать|сделай|сделать|сгенерируй|реализуй|запиши|записать|сохрани|сохранить|write|create|build|implement|generate|save)\b", prompt)
    target = re.search(r"\b(?:скрипт\w*|программ\w*|проект\w*|файл\w*|приложени\w*|утилит\w*|scripts?|programs?|projects?|files?|applications?|apps?|hello\s*world)\b|\b[\w.-]+\.(?:cs|csproj|py|js|ts|html|json|sh|ps1|go|rs|java|cpp)\b", prompt)
    return verb is not None and target is not None


def _execution_intent(messages: list[dict]) -> bool:
    prompt = next((message.get("content", "") for message in reversed(messages) if message.get("role") == "user"), "")
    if not isinstance(prompt, str) or "```" in prompt or re.search(r"\b(?:пример\w*|объясн\w*|объясни|examples?|explain|только\s+в\s+чате|не\s+(?:запускай|выполняй|запускать|выполнять)|(?:don't|do not)\s+(?:run|execute)|only\s+in\s+chat)\b", prompt.casefold()):
        return False
    return bool(re.search(r"\b(?:запусти|запустить|выполни|выполнить|run|execute)\b", prompt.casefold())
                and re.search(r"\b(?:powershell|скрипт\w*|scripts?)\b|\b[\w.-]+\.ps1\b", prompt.casefold()))


def _runtime_instruction(code: bool, web: bool, repository: str | None, corrective: bool, powershell: bool = False) -> str:
    parts = ["Tools are enabled only by the human-controlled menu for this chat. Each call needs fresh human approval; tool availability is not permission to execute."]
    if code:
        parts.append("The selected repository root is " + json.dumps(repository, ensure_ascii=False) + ". All code tool paths must be relative to this root; never pass an absolute path or change the selected root.")
        parts.append("When asked to create a program, script, project or file, create the actual files with write_file/create_directory tools rather than only describing code in the reply. For explanation or example-only requests, answer in chat. Honor explicit chat-only or do-not-save requests. You may inspect existing files first. Report files as created only after successful tool results. Denied or failed calls are not successful changes; do not claim they succeeded.")
        parts.append("For a project request, write its runnable project metadata, dependencies and entry point as actual files (for example a C# project needs a .csproj file and Program.cs). For a script-only request, create the actual script file. Do not substitute only a code example for the requested project.")
    if web:
        parts.append("Use web tools only when the task needs public web information. Every web request also requires approval.")
    if powershell:
        parts.append("Restricted PowerShell is enabled for approved .ps1 files in the selected repository " + json.dumps(repository, ensure_ascii=False) + ". Use powershell_run with a relative script path when asked to run a script. It supports only basic arithmetic/control flow and Write-Output/Write-Host/Error/Warning/Verbose/Debug. Filesystem, network, external programs, .NET APIs, dynamic commands and C# compilation/execution are unavailable. Writing a script and running it require separate approvals. Claim execution only after a successful powershell_run result with exit_code=0.")
    if corrective:
        parts.append("This is one corrective attempt: the previous reply did not attempt the requested action. Use the appropriate available tools to create the requested files or run the requested PowerShell script now. If you cannot, clearly state what was not performed.")
    return "\n".join(parts)


class GenerationCancelled(ApiError):
    """The UI cancelled a generation; complete tool results remain in history."""


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    usage: dict | None
    finish: str | None
    elapsed: float


def initial_messages(config: dict) -> list[dict]:
    return [{"role": "system", "content": config["system_prompt"]}] if config["system_prompt"] else []


def generate_response(
    client: ApiClient,
    messages: list[dict],
    harness: ToolHarness | None = None,
    on_event: Callable[[StreamEvent], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    *,
    rag: RagSession | None = None,
) -> GenerationResult:
    """Run approved tools, keeping runtime instructions out of saved history."""
    start = time.perf_counter()
    usage = None
    finish = None
    answer: list[str] = []
    call_count = 0
    schemas = harness.schemas() if harness is not None else []
    names = {schema["function"]["name"] for schema in schemas}
    settings = getattr(harness, "settings", None)
    code_enabled = bool(getattr(settings, "code_enabled", names & _CODE_TOOLS))
    web_enabled = bool(getattr(settings, "web_enabled", names & _WEB_TOOLS))
    powershell_enabled = bool(getattr(settings, "powershell_enabled", "powershell_run" in names))
    root = getattr(settings, "repository", None)
    repository = str(root) if root is not None else None
    creation_task = code_enabled and _creation_intent(messages)
    execution_task = powershell_enabled and _execution_intent(messages)
    corrective_used = False
    corrective_pending = False
    mutation_attempted = False
    execution_attempted = False
    mutation_uncertain = False
    had_failure = False
    result_slots: list[tuple[dict, dict]] = []
    used_ids = {
        call["id"]
        for message in messages if isinstance(message, dict) and isinstance(message.get("tool_calls"), list)
        for call in message["tool_calls"] if isinstance(call, dict) and isinstance(call.get("id"), str)
    }

    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise GenerationCancelled("Генерация отменена.")

    def emit(kind: str, value: dict) -> None:
        if on_event is not None:
            on_event(StreamEvent(kind, copy.deepcopy(value)))

    def status(phase: str) -> None:
        emit("harness_status", {"phase": phase, "code_enabled": code_enabled,
             "web_enabled": web_enabled, "powershell_enabled": powershell_enabled,
             "repository": repository, "tool_count": len(schemas)})

    def parsed_result(result: dict) -> dict:
        try:
            value = json.loads(result["content"])
        except (ValueError, TypeError, RecursionError):
            value = None
        if not isinstance(value, dict) or not isinstance(value.get("status"), str) or value["status"] not in {"ok", "denied", "error"}:
            return {"status": "error", "error": "Инструмент вернул неверный формат результата."}
        return value

    def summary() -> dict:
        counts = {"ok": 0, "denied": 0, "error": 0}
        files, directories, scripts = [], [], []
        for call, result in result_slots:
            value = parsed_result(result)
            counts[value["status"]] += 1
            if value["status"] != "ok":
                continue
            name = call["function"]["name"]
            arguments = json.loads(call["function"]["arguments"])
            if name == "powershell_run" and value.get("exit_code") == 0:
                path = value.get("path", arguments.get("path"))
                if isinstance(path, str) and path not in scripts:
                    scripts.append(path)
            if name == "write_file":
                path = value.get("path", arguments.get("path"))
                if isinstance(path, str) and path not in files:
                    files.append(path)
            created = value.get("directories_created", []) if name in _MUTATION_TOOLS else []
            if name == "create_directory" and not created and value.get("created") is True:
                created = [value.get("path", arguments.get("path"))]
            if isinstance(created, list):
                for path in created:
                    if isinstance(path, str) and path not in directories:
                        directories.append(path)
        return {"requested": len(result_slots), "succeeded": counts["ok"], "denied": counts["denied"],
                "failed": counts["error"], "files_written": files, "directories_created": directories,
                "no_changes": not files and not directories and not mutation_uncertain,
                "uncertain_changes": mutation_uncertain, "scripts_run": scripts}

    status("ready" if schemas else "idle")
    try:
        check_cancelled()
        rag_context = None
        if rag is not None and rag.enabled:
            try:
                rag_context = rag.prepare(messages)
            except RagError as error:
                raise ApiError(f"RAG: {error}") from error
            check_cancelled()
            emit("rag_sources", {"sources": rag_context.sources})
            if not rag_context.sources:
                text = ("В базе знаний не найдены подходящие фрагменты для этого вопроса. "
                        "Добавьте документы через /rag add ПУТЬ, уточните вопрос или выключите режим: /rag off.")
                if on_event is not None:
                    on_event(StreamEvent("content", text))
                status("complete")
                return GenerationResult(text, None, "stop", time.perf_counter() - start)
        for round_index in range(MAX_TOOL_ROUNDS + 1):
            check_cancelled()
            answer, calls = [], []
            finish = None
            schemas = harness.schemas() if harness is not None else []
            outbound = list(messages)
            if rag_context is not None:
                position = next((index for index, message in enumerate(outbound)
                                 if message["role"] != "system"), len(outbound))
                outbound.insert(position, {"role": "system", "content": rag_context.instruction})
            if schemas:
                instruction = _runtime_instruction(code_enabled, web_enabled, repository, corrective_pending, powershell_enabled)
                position = next((index for index, message in enumerate(outbound) if message["role"] != "system"), len(outbound))
                outbound.insert(position, {"role": "system", "content": instruction})
            force_tool = (creation_task or execution_task) and (round_index == 0 or corrective_pending)
            if force_tool and schemas:
                stream = client.stream_chat(outbound, tools=schemas, tool_choice="required")
            else:
                stream = client.stream_chat(outbound, tools=schemas) if schemas else client.stream_chat(outbound)
            corrective_pending = False
            status("requesting")
            try:
                for event in stream:
                    check_cancelled()
                    if event.kind == "content":
                        answer.append(event.value)
                    elif event.kind == "usage":
                        usage = event.value
                    elif event.kind == "finish":
                        finish = event.value
                    elif event.kind == "tool_call":
                        calls.append(event.value)
                    if on_event is not None:
                        on_event(event)
                    check_cancelled()
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            if not calls:
                if (creation_task and not mutation_attempted) or (execution_task and not execution_attempted):
                    if not mutation_attempted and not execution_attempted and not had_failure and not corrective_used and round_index < MAX_TOOL_ROUNDS:
                        corrective_used = corrective_pending = True
                        status("retrying")
                        continue
                    answer = ["Файлы не созданы: модель не выполнила запись через инструменты. Изменений в репозитории нет."] if creation_task and not mutation_attempted else ["Скрипт не запущен: модель не выполнила вызов PowerShell."]
                if not answer:
                    raise ApiError("Модель не выдала текст ответа. Увеличьте max_tokens и проверьте журнал сервера.")
                break
            if not schemas:
                raise ApiError("Модель запросила инструменты, отключённые в меню этого чата.")
            if finish != "tool_calls":
                raise ApiError("Незавершённый запрос инструмента не выполнен.")
            if len(calls) > MAX_TOOL_CALLS:
                raise ApiError("Неверный список вызовов инструментов (максимум 16 вызовов).")
            allowed_names = {schema["function"]["name"] for schema in schemas}
            clean_calls = []
            argument_bytes = 0
            for value in calls:
                try:
                    call = validated_tool_call(value)
                except (ValueError, RecursionError) as error:
                    raise ApiError("Неверный запрос инструмента не выполнен.") from error
                if call["id"] in used_ids:
                    raise ApiError("Неверный или повторный запрос инструмента не выполнен.")
                if call["function"]["name"] not in allowed_names:
                    raise ApiError("Модель запросила инструмент, который не подключён в этом чате.")
                argument_bytes += len(call["function"]["arguments"].encode("utf-8"))
                if argument_bytes > MAX_TOOL_ARGUMENT_BYTES:
                    raise ApiError("Аргументы инструментов слишком большие (максимум 256 КБ).")
                used_ids.add(call["id"])
                clean_calls.append(call)
            assistant = {"role": "assistant", "content": "".join(answer), "tool_calls": copy.deepcopy(clean_calls)}
            results = [{"role": "tool", "tool_call_id": call["id"], "content": json.dumps({
                "status": "error", "error": "Вызов не выполнен: генерация остановлена.",
            }, ensure_ascii=False)} for call in clean_calls]
            # Commit complete protocol rounds before any cancellation or execution.
            messages.extend([assistant, *results])
            result_slots.extend(zip(clean_calls, results))
            if round_index == MAX_TOOL_ROUNDS or call_count + len(clean_calls) > MAX_TOOL_CALLS:
                answer = [TOOL_LIMIT_MESSAGE]
                had_failure = True
                break
            for call, result in zip(clean_calls, results):
                check_cancelled()
                function = call["function"]
                emit("tool_start", {"id": call["id"], "name": function["name"],
                                    "arguments": json.loads(function["arguments"])})
                check_cancelled()
                mutation = function["name"] in _MUTATION_TOOLS
                mutation_attempted = mutation_attempted or mutation
                execution_attempted = execution_attempted or function["name"] == "powershell_run"
                try:
                    result["content"] = harness.execute(function["name"], function["arguments"])
                except KeyboardInterrupt:
                    mutation_uncertain = mutation_uncertain or mutation
                    result["content"] = json.dumps({"status": "error", "error":
                        "Выполнение прервано. Действие могло быть выполнено частично; проверьте состояние."}, ensure_ascii=False)
                    value = parsed_result(result)
                    emit("tool_result", {"id": call["id"], "name": function["name"], "status": "error", "result": value})
                    raise
                except HarnessError as error:
                    result["content"] = json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False)
                value = parsed_result(result)
                if function["name"] == "powershell_run" and value["status"] == "ok" and value.get("exit_code") != 0:
                    value = {**value, "status": "error"}
                result["content"] = json.dumps(value, ensure_ascii=False)
                had_failure = had_failure or value["status"] != "ok"
                mutation_uncertain = mutation_uncertain or (mutation and value["status"] == "error")
                emit("tool_result", {"id": call["id"], "name": function["name"], "status": value["status"], "result": value})
                call_count += 1
                check_cancelled()
            emit("tool_round", {"round": round_index + 1, "calls": len(clean_calls)})
        progress = summary()
        if creation_task and not progress["files_written"] and mutation_attempted:
            if progress["no_changes"]:
                answer = ["Файлы не созданы и не изменены через инструменты."]
            elif mutation_uncertain:
                answer = ["Запись файлов не подтверждена: инструмент сообщил об ошибке. Проверьте репозиторий; действие могло быть выполнено частично."]
            elif progress["directories_created"]:
                answer = ["Файлы не записаны; созданы только каталоги: " + ", ".join(progress["directories_created"]) + "."]
        if execution_task and execution_attempted and not progress["scripts_run"]:
            answer = ["Выполнение скрипта не подтверждено: вызов был отклонён или завершился ошибкой."]
        if rag_context is not None and rag_context.sources:
            footer = "\n\nИсточники RAG:\n" + "\n".join(
                f"[{source['id']}] {source['source']} · {source['location']}"
                for source in rag_context.sources
            )
            answer.append(footer)
            if on_event is not None:
                on_event(StreamEvent("content", footer))
        status("complete")
        return GenerationResult("".join(answer), usage, finish, time.perf_counter() - start)
    except BaseException:
        status("error")
        raise
    finally:
        emit("harness_summary", summary())
