"""Chat-local harness telemetry derived from host events, never model prose."""

from __future__ import annotations

from dataclasses import dataclass, field

from .harness import HarnessSettings


@dataclass
class HarnessStatus:
    settings: HarnessSettings = field(default_factory=HarnessSettings)
    phase: str = "idle"
    last_tool: str = ""
    last_result: str = ""
    summary: dict | None = None

    def configure(self, settings: HarnessSettings) -> None:
        if settings != self.settings:
            self.phase = "ready" if settings.code_enabled or settings.web_enabled or settings.powershell_enabled else "idle"
            self.last_tool = self.last_result = ""
            self.summary = None
        self.settings = settings

    def update(self, kind: str, value: dict) -> None:
        if kind == "harness_status":
            self.phase = value.get("phase", self.phase)
            if self.phase == "ready":
                self.summary = None
                self.last_tool = self.last_result = ""
        elif kind == "tool_start":
            self.last_tool = value.get("name", "")
            self.last_result = ""
            self.phase = "validating"
        elif kind == "tool_result":
            self.last_tool = value.get("name", "")
            self.last_result = value.get("status", "error")
            self.phase = "result"
        elif kind == "harness_summary":
            self.summary = value

    @property
    def enabled(self) -> bool:
        return self.settings.code_enabled or self.settings.web_enabled or self.settings.powershell_enabled

    def connection(self) -> str:
        if not self.enabled:
            return "Харнес: выключен · F3 — подключить"
        names = []
        if self.settings.code_enabled:
            names.append("код")
        if self.settings.web_enabled:
            names.append("веб")
        if self.settings.powershell_enabled:
            names.append("PowerShell")
        return "Харнес: " + "+".join(names) + " подключён"

    def activity(self) -> str:
        if not self.enabled:
            return ""
        if self.phase == "requesting":
            return "отправка инструментов API; ждём модель"
        if self.phase == "retrying":
            return "модель не вызвала нужный инструмент; повтор запроса"
        if self.phase == "approval":
            return f"{self.last_tool}: ждёт вашего подтверждения"
        if self.phase == "executing":
            return f"{self.last_tool}: выполняется"
        if self.phase == "validating":
            return f"{self.last_tool}: проверка запроса"
        if self.phase == "error" and self.summary is None:
            return "ошибка / прервано" + (f" · {self.last_tool}: {self.last_result}" if self.last_result else "")
        if self.summary is not None:
            count = self.summary.get("requested", 0)
            prefix = "ошибка / прервано · " if self.phase == "error" else ""
            if not count:
                return prefix + "модель не вызвала инструменты; изменений на диске нет"
            counts = (f"вызовов: {count} · успешно: {self.summary.get('succeeded', 0)}"
                    f" · отказов: {self.summary.get('denied', 0)} · ошибок: {self.summary.get('failed', 0)}"
                    f" · файлов записано: {len(self.summary.get('files_written', []))}"
                    f" · успешно выполнено скриптов: {len(self.summary.get('scripts_run', []))}")
            return prefix + counts + (" · возможны частичные изменения" if self.summary.get("uncertain_changes") else "")
        if self.last_result:
            result = {"ok": "успешно", "denied": "отклонён", "error": "ошибка"}.get(self.last_result, "ошибка")
            return f"{self.last_tool}: {result}"
        return "вызовов ещё нет"

    def compact(self) -> str:
        parts = [self.connection()]
        if self.enabled:
            parts.append(self.activity())
        if self.settings.code_enabled or self.settings.powershell_enabled:
            parts.append(f"репозиторий: {self.settings.repository}")
        return " · ".join(parts)

    def details(self) -> str:
        lines = [self.connection()]
        lines.append("Веб: " + ("включён" if self.settings.web_enabled else "выключен"))
        lines.append("Код: " + ("включён" if self.settings.code_enabled else "выключен"))
        lines.append("PowerShell: " + ("включён" if self.settings.powershell_enabled else "выключен"))
        if self.settings.code_enabled or self.settings.powershell_enabled:
            lines.append(f"Граница файлового доступа: {self.settings.repository}")
        if self.enabled:
            lines.append("Активность: " + self.activity())
            lines.append("Каждый вызов требует отдельного подтверждения.")
        if self.summary is not None:
            for path in self.summary.get("files_written", []):
                lines.append("Записан файл: " + path)
            for path in self.summary.get("directories_created", []):
                lines.append("Создан каталог: " + path)
            for path in self.summary.get("scripts_run", []):
                lines.append("Выполнен скрипт: " + path)
        return "\n".join(lines)
