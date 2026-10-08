"""Offline dataset commands and an isolated SFT training process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Callable

from .runtime import _WindowsJob


def add_sft_parser(commands) -> None:
    parser = commands.add_parser("sft", help="Датасеты и LoRA/QLoRA-дообучение")
    commands = parser.add_subparsers(dest="sft_command")
    template = commands.add_parser("init", help="Создать шаблон датасета JSONL")
    template.add_argument("path", type=Path)
    validate = commands.add_parser("validate", help="Проверить датасет без запуска модели")
    validate.add_argument("path", type=Path)
    session = commands.add_parser("from-session", help="Импортировать историю как кандидатов для проверки")
    session.add_argument("path", type=Path)
    session.add_argument("--output", type=Path, required=True)
    session.add_argument("--task", default="chat")
    prepare = commands.add_parser("prepare", help="Подготовить train/validation и задание обучения")
    prepare.add_argument("path", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--base-model", required=True, help="Исходный HF-репозиторий или каталог Transformers, не GGUF")
    prepare.add_argument("--revision", help="Версия исходной модели; для воспроизводимости укажите commit SHA")
    prepare.add_argument("--method", choices=("lora", "qlora"), default="qlora")
    prepare.add_argument("--validation-ratio", type=float, default=0.1)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--max-length", type=int, default=2048)
    prepare.add_argument("--epochs", type=float, default=3.0)
    prepare.add_argument("--learning-rate", type=float, default=0.0001)
    prepare.add_argument("--lora-rank", type=int, default=16)
    prepare.add_argument("--lora-alpha", type=int, default=32)
    prepare.add_argument("--lora-dropout", type=float, default=0.05)
    check = commands.add_parser("check", help="Проверить отдельное окружение обучения")
    check.add_argument("--python", type=Path, help="Python окружения с training-зависимостями")
    train = commands.add_parser("train", help="Запустить обучение в отдельном процессе; Ctrl+C — остановить")
    train.add_argument("job", type=Path)
    train.add_argument("--python", type=Path, help="Python окружения с training-зависимостями")


def _path(value: Path, root: Path) -> Path:
    value = value.expanduser()
    return (value if value.is_absolute() else root / value).resolve()


def launch_training(root: Path, *, job: Path | None = None, python: Path | None = None,
                    emit: Callable[[str], None] = print) -> int:
    """Stream a child process without shell expansion or training imports here."""
    executable = str(_path(python, root)) if python is not None else sys.executable
    argv = [executable, "-u", "-m", "llmopenchat.sft_train"]
    if job is None:
        argv.append("--check")
    else:
        job = _path(job, root)
        if not job.is_file():
            raise ValueError(f"Задание обучения не найдено: {job}")
        argv.append(str(job))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root) + (os.pathsep + environment["PYTHONPATH"]
                                              if environment.get("PYTHONPATH") else "")
    environment["PYTHONIOENCODING"] = "utf-8"
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    guard = _WindowsJob() if os.name == "nt" else None
    process = None
    try:
        process = subprocess.Popen(argv, cwd=root, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", creationflags=flags)
        if guard is not None:
            guard.assign(process.pid)
        assert process.stdout is not None
        for line in process.stdout:
            emit(line.rstrip("\r\n"))
        return process.wait()
    except KeyboardInterrupt:
        emit("Останавливаю обучение…")
        if process is None:
            return 130
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        return 130
    finally:
        try:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                if process.stdout is not None:
                    process.stdout.close()
        finally:
            if guard is not None:
                guard.close()


def dispatch_sft(args: argparse.Namespace, root: Path, *, emit=print, console=None) -> int:
    from .sft import from_session, prepare_dataset, validate_dataset, write_template

    action = args.sft_command
    if action is None:
        if console is None:
            raise ValueError("Для меню нужен терминал; используйте sft --help.")
        return sft_menu(root, console)
    if action == "init":
        path = write_template(_path(args.path, root))
        emit(f"Создан шаблон: {path}\nПроверьте примеры и отметьте качественные status: approved.")
    elif action == "validate":
        emit(json.dumps(validate_dataset(_path(args.path, root)), ensure_ascii=False, indent=2))
    elif action == "from-session":
        path = from_session(_path(args.path, root), _path(args.output, root), task=args.task)
        emit(f"Кандидаты сохранены: {path}\nСтатус draft: исправьте и проверьте ответы перед обучением.")
    elif action == "prepare":
        base_model = args.base_model
        if (root / base_model).is_dir():
            base_model = str((root / base_model).resolve())
        options = {name: getattr(args, name) for name in (
            "revision", "method", "validation_ratio", "seed", "max_length", "epochs",
            "learning_rate", "lora_rank", "lora_alpha", "lora_dropout",
        )}
        path = prepare_dataset(_path(args.path, root), _path(args.output, root),
                               base_model, **options)
        emit(f"Задание готово: {path}\nПроверьте job.json; запуск: sft train \"{path}\" --python ПУТЬ_К_PYTHON")
    elif action in {"check", "train"}:
        return launch_training(root, job=getattr(args, "job", None), python=args.python, emit=emit)
    return 0


def sft_menu(root: Path, console) -> int:
    from .menu import read_choice, say
    from .sft import from_session, prepare_dataset, validate_dataset, write_template

    def emit(text):
        say(console, text)

    def read_path(prompt):
        value = read_choice(prompt).strip().strip('"')
        if not value:
            raise ValueError("Путь не указан.")
        return _path(Path(value), root)

    while True:
        emit("\nSFT · дообучение\n1 — Создать шаблон датасета\n2 — Кандидаты из истории чата\n"
             "3 — Проверить датасет\n4 — Подготовить задание обучения\n"
             "5 — Проверить Python окружения обучения\n6 — Запустить задание обучения\n0 — Назад")
        choice = read_choice()
        if choice == "0":
            return 0
        try:
            if choice == "1":
                emit(str(write_template(read_path("Новый JSONL-файл > "))))
                emit("Отредактируйте примеры; проверенные ответы помечайте status: approved.")
            elif choice == "2":
                source = read_path("Файл истории чата > ")
                emit(str(from_session(source, read_path("Новый JSONL-файл > "))))
                emit("Импортированы кандидаты draft. Проверьте и исправьте ответы.")
            elif choice == "3":
                emit(json.dumps(validate_dataset(read_path("Датасет > ")), ensure_ascii=False, indent=2))
            elif choice == "4":
                dataset = read_path("Датасет > ")
                base_model = read_choice("Исходная HF-модель или каталог Transformers > ").strip().strip('"')
                if (root / base_model).is_dir():
                    base_model = str((root / base_model).resolve())
                method = read_choice("Метод: qlora (Enter, NVIDIA CUDA) или lora > ").strip() or "qlora"
                output = read_path("Новая папка задания > ")
                job = prepare_dataset(dataset, output, base_model, method=method)
                emit(f"Задание: {job}\nОбучение запускается отдельным пунктом меню.")
            elif choice in {"5", "6"}:
                job = read_path("Файл job.json > ") if choice == "6" else None
                value = read_choice("Python окружения обучения (Enter — текущий) > ").strip().strip('"')
                status = launch_training(root, job=job, python=Path(value) if value else None, emit=emit)
                if status:
                    emit(f"Процесс завершён с кодом {status}.")
            else:
                emit("Введите 1, 2, 3, 4, 5, 6 или 0.")
        except KeyboardInterrupt:
            emit("Действие отменено.")
        except (OSError, ValueError) as error:
            emit(f"Ошибка SFT: {error}")
