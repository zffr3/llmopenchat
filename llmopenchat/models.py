"""A small, reproducible package manager for compatible local text GGUF models.

Only explicitly selected quantizations are downloaded. Installs never change the
active config, and existing files (including interrupted downloads) are preserved
on errors. The legacy installer is imported as an unowned, read-only package.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import sys
from typing import Callable
from urllib.parse import quote, urlsplit
import uuid

from .config import DEFAULT_CONFIG, write_config
from .download import DownloadError, _digest, download_file
from .setup import SetupError, _install_runtime, _sha256


_GIB = 1024**3
_COMMIT = re.compile(r"[0-9a-fA-F]{40}")
_HASH = re.compile(r"[0-9a-fA-F]{64}")
_QUANT = re.compile(r"(?<![A-Za-z0-9])(?:IQ[1-4](?:_[A-Z0-9]+)+|Q[2-8](?:_[A-Z0-9]+)*|BF16|F16|F32|FP16)(?![A-Za-z0-9])", re.I)
_SPLIT = re.compile(r"^(.*?)-(\d{5})-of-(\d{5})\.gguf$", re.I)
_MARKER = ".llmopenchat-package.json"

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


class ModelError(RuntimeError):
    """An actionable catalogue, integrity, or model management error."""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    repo_id: str
    parameters_b: float
    description: str
    family: str
    censorship: str
    focus: str
    source_url: str
    revision: str
    default_filename: str
    default_size: int
    default_sha256: str
    default_quantization: str = "Q4_K_M"
    architecture: str = "deepseek2"
    context_length: int = 202752


_ORIGINAL = "Оригинальное выравнивание Z.ai; степень отказов не измерена."
_ABLITERATED = "Автор заявляет снижение отказов (abliteration); эффект зависит от запроса."
_INSTRUCT = "Исходное instruct-выравнивание автора; степень отказов не измерена."
CATALOG = (
    ModelSpec(
        "glm-4.7-flash", "GLM-4.7-Flash", "lmstudio-community/GLM-4.7-Flash-GGUF", 29.943,
        "Официальная Flash: универсальный чат, код и рассуждения. Около 30B параметров всего, 3B активны на токен.",
        "flash", _ORIGINAL, "чат; код / программирование; рассуждения", "https://huggingface.co/zai-org/GLM-4.7-Flash",
        "b58019be521f54a64160968f3c37a55174b1f186", "GLM-4.7-Flash-Q4_K_M.gguf", 18132721120,
        "a53ccc46b2a48e4b29b97f1d5828902a9d8701e74c3f7c2f5331c136b66c728b",
    ),
    ModelSpec(
        "glm-4.7-flash-huihui", "Huihui GLM-4.7-Flash abliterated", "mradermacher/Huihui-GLM-4.7-Flash-abliterated-GGUF", 29.943,
        "Flash с abliteration от Huihui: обычный чат и код, со снижением отказов по заявлению автора.",
        "flash", _ABLITERATED, "чат; код / программирование", "https://huggingface.co/huihui-ai/Huihui-GLM-4.7-Flash-abliterated",
        "2d925ab0fdaa279a87485f4786ca22661bf030b2", "Huihui-GLM-4.7-Flash-abliterated.Q4_K_M.gguf", 18132722048,
        "ca247d439725435ad5addc43926d311d0b0cd97cb65a32eb7d5b3ac479bf434d",
    ),
    ModelSpec(
        "glm-4.7-flash-neo-code", "GLM-4.7-Flash Heretic NEO CODE", "DavidAU/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF", 29.943,
        "Heretic-вариант Flash с калибровкой квантования на коде и 16-битным выходным тензором. CODE-imatrix не означает подтвержденное DevOps-дообучение.",
        "flash", "Автор заявляет снижение отказов (Heretic); независимой шкалы цензуры нет.",
        "код / программирование; чат", "https://huggingface.co/DavidAU/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF",
        "4395b80e6ad892a21bb668dfab4e24c9813e4331", "GLM-4.7-Flash-Uncen-Hrt-NEO-CODE-MAX-imat-D_AU-Q4_K_M.gguf", 18506911680,
        "6eb0adad2033705c1b107c6303673a6bcff6cf6df8c798bc3dcaa5879fffd287",
    ),
    ModelSpec(
        "glm-4-32b", "GLM-4-32B-0414", "lmstudio-community/GLM-4-32B-0414-GGUF", 32.566,
        "Плотная GLM-4 32B для диалога, инженерного кода и документов. Обычно медленнее MoE Flash при частичной выгрузке на CPU.",
        "glm4", _ORIGINAL, "чат; код / программирование; документы", "https://huggingface.co/zai-org/GLM-4-32B-0414",
        "20621a295c480ce7037bed7808533dc356a386bb", "GLM-4-32B-0414-Q4_K_M.gguf", 19680022112,
        "2080ce59fc090f61efd55b54235810d30ec5b1e1be60bd024ebc6ef1a835c733", architecture="glm4", context_length=32768,
    ),
    ModelSpec(
        "glm-4-32b-abliterated", "GLM-4-32B-0414 abliterated", "mradermacher/GLM-4-32B-0414-abliterated-GGUF", 32.566,
        "Плотная GLM-4 32B с abliteration; автор квантования указывает модификацию Huihui и снижение отказов.",
        "glm4", _ABLITERATED, "чат; код / программирование", "https://huggingface.co/mradermacher/GLM-4-32B-0414-abliterated-GGUF",
        "de50b0f2a7ef06611cd4becf1391b5ba3f7bb719", "GLM-4-32B-0414-abliterated.Q4_K_M.gguf", 19680022976,
        "a2072c1d0662deea26490998fb5a955746b4c84f046340ac9cdc07618bc47da5", architecture="glm4", context_length=32768,
    ),
    ModelSpec(
        "glm-z1-32b", "GLM-Z1-32B-0414", "unsloth/GLM-Z1-32B-0414-GGUF", 32.566,
        "Reasoning-версия 32B для математики, кода и логики. Рассуждения могут требовать много токенов; отключение thinking у Flash сюда не переносится.",
        "glm4", _ORIGINAL, "рассуждения; математика; код / программирование", "https://huggingface.co/zai-org/GLM-Z1-32B-0414",
        "1a27b7404359fca3d63cfbef2f95cd7c43c2e945", "GLM-Z1-32B-0414-Q4_K_M.gguf", 19680023136,
        "80c6c4d6456255a63e059aa67b8529a6894f86b43377f7a287eeed8d7cc95214", architecture="glm4", context_length=32768,
    ),
    ModelSpec(
        "glm-4.7-flash-coder", "GLM-4.7-Flash Coder", "ngquocvinh/GLM-4.7-Flash-Coder-GGUF", 29.943,
        "Дообучена White Circle на успешных задачах программирования и исправлении кода. Полный агентный режим требует инструментов, которых в чат-клиенте пока нет.",
        "flash", "Автор не заявляет снятие выравнивания; степень отказов не измерена.",
        "код / программирование; исправление ошибок", "https://huggingface.co/whitecircle/GLM-4.7-Flash-Coder",
        "4822186c67c9fd2f7e5f11095f0df41b57ef5dab", "GLM-4.7-Flash-Coder-Q4_K_M.gguf", 18132722464,
        "6b11aaf453d8d8e42bfb3cc407bce11c1b09dba00efb91b3664751a22ee44020",
    ),
    ModelSpec(
        "glm-4.7-flash-reap-23b", "GLM-4.7-Flash REAP 23B A3B", "unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF", 22.996,
        "Flash с удалением 25% экспертов методом REAP от Cerebras: 23B всего / 3B активны. Q4 занимает около 13.1 ГиБ; остается память для контекста.",
        "flash", _ORIGINAL, "чат; код / программирование; меньше памяти", "https://huggingface.co/cerebras/GLM-4.7-Flash-REAP-23B-A3B",
        "983a65c63bc7bd37353e5246abc8005b3f72d5ec", "GLM-4.7-Flash-REAP-23B-A3B-Q4_K_M.gguf", 14113838816,
        "038e930ee1e050dca7732a7a2c768a4b0f83f5655add1acb6c7380852b9eef68",
    ),
    ModelSpec(
        "qwen3-0.6b", "Qwen3-0.6B", "unsloth/Qwen3-0.6B-GGUF", 0.596,
        "Самая компактная Qwen для простого анализа текста: классификация, метки и короткие сводки. Поддерживает русский; сложные документы лучше поручать более крупной модели. Для быстрого ответа добавьте /no_think в запрос.",
        "gguf", _INSTRUCT, "анализ текста; классификация; сводки; русский; лёгкий чат", "https://huggingface.co/Qwen/Qwen3-0.6B",
        "50968a4468ef4233ed78cd7c3de230dd1d61a56b", "Qwen3-0.6B-Q4_K_M.gguf", 396705472,
        "ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a", architecture="qwen3", context_length=40960,
    ),
    ModelSpec(
        "lfm2-1.2b-extract", "LFM2-1.2B-Extract", "LiquidAI/LFM2-1.2B-Extract-GGUF", 1.17,
        "Специалист по извлечению данных из текста в JSON, XML и YAML. Лучше задавать схему в system prompt, температуру 0 и новый чат для каждого документа. Русский не заявлен; лицензия LFM 1.0.",
        "gguf", _INSTRUCT, "анализ текста; извлечение данных; JSON; XML; YAML; английский", "https://huggingface.co/LiquidAI/LFM2-1.2B-Extract",
        "ef65f6005f6a4de8a8e7a60279242b1c96be229a", "LFM2-1.2B-Extract-Q4_K_M.gguf", 730894048,
        "09b60b507ee7d1698b2b4dfce184c75083d7790c7701910ed60afa2801024702", architecture="lfm2", context_length=128000,
    ),
    ModelSpec(
        "lfm2-1.2b-rag", "LFM2-1.2B-RAG", "LiquidAI/LFM2-1.2B-RAG-GGUF", 1.17,
        "Специалист по ответам на вопросы по предоставленным документам и контексту. Текст нужно вставить в запрос: индекс и поиск по файлам модель сама не добавляет. Рекомендуется температура 0; русский не заявлен; лицензия LFM 1.0.",
        "gguf", _INSTRUCT, "анализ текста; документы; вопросы по тексту; RAG; английский", "https://huggingface.co/LiquidAI/LFM2-1.2B-RAG",
        "d8628e893f593f5649ef4f06f76fddaa5bacaa0d", "LFM2-1.2B-RAG-Q4_K_M.gguf", 730894048,
        "5e4d123cd76dd38a1b55f86a5e1f5fa579e452ff89fa636709edbecd3513db0a", architecture="lfm2", context_length=128000,
    ),
    ModelSpec(
        "qwen2.5-coder-1.5b", "Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", 1.54,
        "Компактный специалист по коду: небольшие скрипты, объяснение и исправление ошибок. Instruct-версия на 1.54B с лицензией Apache 2.0; для сложных проектов лучше более крупная модель.",
        "gguf", _INSTRUCT, "код / программирование; скрипты; исправление ошибок", "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "f86cb2c1fa58255f8052cc32aeede1b7482d4361", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 1117320768,
        "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046", architecture="qwen2", context_length=32768,
    ),
    ModelSpec(
        "qwen3-1.7b", "Qwen3-1.7B", "unsloth/Qwen3-1.7B-GGUF", 1.721,
        "Лёгкая многоязычная модель для анализа текста, сводок и извлечения фактов. Подходит для русского текста и переводов; /no_think отключает длинные рассуждения.",
        "gguf", _INSTRUCT, "анализ текста; сводки; извлечение данных; перевод; русский", "https://huggingface.co/Qwen/Qwen3-1.7B",
        "d7f544eead698dbd1f15126ef60b45a1e1933222", "Qwen3-1.7B-Q4_K_M.gguf", 1107409472,
        "b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897", architecture="qwen3", context_length=40960,
    ),
    ModelSpec(
        "smollm3-3b", "SmolLM3-3B", "unsloth/SmolLM3-3B-GGUF", 3.075,
        "Компактная модель Hugging Face для сводок, понимания текста и рассуждений. Основные языки: английский, французский, испанский, немецкий, итальянский и португальский; русский слабее. Для обычного ответа задайте /no_think в system prompt.",
        "gguf", _INSTRUCT, "анализ текста; сводки; рассуждения; европейские языки", "https://huggingface.co/HuggingFaceTB/SmolLM3-3B",
        "a7bc17204c8a326d6bd6e466e076959eddae2025", "SmolLM3-3B-Q4_K_M.gguf", 1915306528,
        "4de907d2d388a5508fb7cb443a06effe14cce3518b0a78d3bdd9e74d9edce989", architecture="smollm3", context_length=65536,
    ),
    ModelSpec(
        "granite-4.0-micro", "Granite-4.0-Micro", "ibm-granite/granite-4.0-micro-GGUF", 3.403,
        "Компактная IBM Granite для сводок, классификации и извлечения данных из документов. Поддерживает ответы по контексту и вызовы функций; русский не входит в заявленные языки. Лицензия Apache 2.0.",
        "gguf", _INSTRUCT, "анализ текста; классификация; сводки; извлечение данных; RAG; инструменты; английский", "https://huggingface.co/ibm-granite/granite-4.0-micro",
        "ec48475f0c811d812fbfb61975717a9c36eeb652", "granite-4.0-micro-Q4_K_M.gguf", 2099502528,
        "97c417dcc0534b0737c74016fb2af083cb17c3b51eaac621192d23961b7024eb", architecture="granite", context_length=131072,
    ),
    ModelSpec(
        "phi-4-mini-instruct", "Phi-4-mini-instruct", "lmstudio-community/Phi-4-mini-instruct-GGUF", 3.836,
        "Компактная модель Microsoft для логики, математики и анализа текста. Русский входит в заявленные языки, но автор отмечает более высокое качество английского. Поддерживает вызовы функций; лицензия MIT.",
        "gguf", _INSTRUCT, "анализ текста; рассуждения; математика; извлечение данных; русский; инструменты", "https://huggingface.co/microsoft/Phi-4-mini-instruct",
        "009f2b81869a0afd6b9aae23d29fa7b7ee2dcac9", "Phi-4-mini-instruct-Q4_K_M.gguf", 2491874400,
        "3c4d3cbdf3006d81444f6c7a5a56eb93d8e0f0e2ba5963b8ab62f9fd42604233", architecture="phi3", context_length=131072,
    ),
    ModelSpec(
        "qwen3-4b-instruct-2507", "Qwen3-4B-Instruct-2507", "unsloth/Qwen3-4B-Instruct-2507-GGUF", 4.022,
        "Основной компактный выбор для анализа русского текста: сводки, сравнение документов и извлечение данных. Версия 2507 отвечает без блоков thinking; нативный контекст 256K, в клиенте по умолчанию 8192.",
        "gguf", _INSTRUCT, "анализ текста; документы; сводки; извлечение данных; перевод; русский", "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507",
        "a06e946bb6b655725eafa393f4a9745d460374c9", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", 2497281120,
        "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597", architecture="qwen3", context_length=262144,
    ),
    ModelSpec(
        "qwen3-8b", "Qwen3-8B", "Qwen/Qwen3-8B-GGUF", 8.191,
        "Многоязычная Qwen для более сложного анализа текста, логики, математики и кода. Поддерживает русский и переключаемые рассуждения; для быстрого ответа добавьте /no_think.",
        "gguf", _INSTRUCT, "анализ текста; рассуждения; математика; код / программирование; перевод; русский", "https://huggingface.co/Qwen/Qwen3-8B",
        "7c41481f57cb95916b40956ab2f0b139b296d974", "Qwen3-8B-Q4_K_M.gguf", 5027783488,
        "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785", architecture="qwen3", context_length=40960,
    ),
    ModelSpec(
        "gpt-oss-20b", "gpt-oss-20B", "bartowski/openai_gpt-oss-20b-GGUF", 20.915,
        "MoE OpenAI для рассуждений, общего чата и структурированных ответов: 20.9B всего / около 3.6B активны. Используется встроенный Harmony-шаблон; эксперты остаются в MXFP4 даже в выбранной Q4_K_M. Рассуждения расходуют лимит ответа.",
        "gguf", _INSTRUCT, "рассуждения; математика; чат; JSON; извлечение данных; инструменты", "https://huggingface.co/openai/gpt-oss-20b",
        "e39ba3aa000c47c83dacdc9e1ca2c9dd0808c205", "openai_gpt-oss-20b-Q4_K_M.gguf", 11673418816,
        "86a21df11afa5a40031ec1974e368ae0ab561ee3995f4d08ff432e8b2b7af9fc", architecture="gpt-oss", context_length=131072,
    ),
    ModelSpec(
        "eurollm-22b-instruct-2512", "EuroLLM-22B-Instruct-2512", "bartowski/utter-project_EuroLLM-22B-Instruct-2512-GGUF", 22.637,
        "Многоязычная модель для переводов и работы с текстом на 35 языках, включая русский и украинский. Дообучение EuroBlocks ориентировано на следование инструкциям и машинный перевод; лицензия Apache 2.0.",
        "gguf", _INSTRUCT, "перевод; русский; украинский; европейские языки; анализ текста; чат", "https://huggingface.co/utter-project/EuroLLM-22B-Instruct-2512",
        "1dbd312ffa10e83e5faa855e3727cbf4abc45f08", "utter-project_EuroLLM-22B-Instruct-2512-Q4_K_M.gguf", 13658576160,
        "2a222374c4adacd55b55795e2f9dca42a2f100d5a2d5858442f928c4c8bdf5e7", architecture="llama", context_length=32768,
    ),
    ModelSpec(
        "mistral-small-3.2-24b", "Mistral-Small-3.2-24B-Instruct-2506", "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF", 23.572,
        "Mistral Small для документов, сводок, точного следования инструкциям и JSON-ответов. Поддерживает русский и вызовы функций. Устанавливается текстовый GGUF; обработка изображений в этом клиенте не подключается.",
        "gguf", _INSTRUCT, "анализ текста; документы; сводки; JSON; русский; инструменты; чат", "https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "b750ec2299225e492f1bd27cab88a0a595fa848f", "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf", 14333922848,
        "a3cc56310807ed0d145eaf9f018ccda9ae7ad8edb41ec870aa2454b0d4700b3c", architecture="llama", context_length=131072,
    ),
    ModelSpec(
        "qwen3-30b-a3b-instruct-2507", "Qwen3-30B-A3B-Instruct-2507", "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF", 30.532,
        "Универсальная MoE для анализа текста, документов, извлечения данных и переводов: 30.5B всего / 3.3B активны. Версия 2507 работает без thinking; нативный контекст 256K, клиент по умолчанию использует 8192.",
        "gguf", _INSTRUCT, "анализ текста; документы; сводки; извлечение данных; JSON; перевод; чат", "https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507",
        "eea7b2be5805a5f151f8847ede8e5f9a9284bf77", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf", 18556686752,
        "6c997b8af17debdfb01d890214400ccbab00db6acc0ba8da5de1cc906c4774d0", architecture="qwen3moe", context_length=262144,
    ),
    ModelSpec(
        "qwen3-coder-30b-a3b", "Qwen3-Coder-30B-A3B-Instruct", "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", 30.532,
        "MoE для программирования, исправления ошибок и понимания репозиториев: 30.5B всего / 3.3B активны. Поддерживает вызовы инструментов и контекст 256K; доступ к файлам включается через инструменты клиента.",
        "gguf", _INSTRUCT, "код / программирование; репозитории; исправление ошибок; инструменты", "https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "b17cb02dd882d5b6ab62fc777ad2995f19668350", "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", 18556689568,
        "fadc3e5f8d42bf7e894a785b05082e47daee4df26680389817e2093056f088ad", architecture="qwen3moe", context_length=262144,
    ),
    ModelSpec(
        "qwen2.5-coder-32b", "Qwen2.5-Coder-32B-Instruct", "unsloth/Qwen2.5-Coder-32B-Instruct-GGUF", 32.764,
        "Плотная модель для программирования, объяснения и исправления сложного кода. GGUF рассчитан на контекст 32K; заявленные 128K исходной модели требуют отдельной настройки YaRN. Часть весов на GPU 16 ГБ будет размещаться в ОЗУ.",
        "gguf", _INSTRUCT, "код / программирование; скрипты; исправление ошибок; объяснение кода", "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct",
        "638ed91307012a4234ffbec397be520624b4f8f3", "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf", 19851335840,
        "77691d25d120a4708c4b301784e3527d2738024b1cddb7ccc08b8bfdc018aaa3", architecture="qwen2", context_length=32768,
    ),
    ModelSpec(
        "gemma-4-31b-it", "Gemma-4-31B-it", "unsloth/gemma-4-31B-it-GGUF", 30.697,
        "Google Gemma 4 для анализа документов, многоязычного чата и рассуждений. Текстовый GGUF содержит около 30.7B параметров; отдельный vision-проектор не устанавливается. Нативный контекст 256K; лицензия Apache 2.0.",
        "gguf", _INSTRUCT, "анализ текста; документы; сводки; рассуждения; перевод; чат", "https://huggingface.co/google/gemma-4-31B-it",
        "c1ac76e99d5513b141e8adde7288b85c3f9c32ec", "gemma-4-31B-it-Q4_K_M.gguf", 18323733440,
        "38bd64c852c4b460434cc7162fa9bdcf242faf86502581a754cb72956bb17f84", architecture="gemma4", context_length=262144,
    ),
    ModelSpec(
        "qwen3-30b-a3b-thinking-2507", "Qwen3-30B-A3B-Thinking-2507", "unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF", 30.532,
        "MoE для сложных рассуждений, математики и логических задач: 30.5B всего / 3.3B активны. Поддерживает только thinking: /no_think его не отключает. Длинные решения могут требовать увеличения max_tokens.",
        "gguf", _INSTRUCT, "рассуждения; математика; логика; анализ текста; код / программирование", "https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507",
        "a9b37aaac12b2bd0098783a443429543dd76a14d", "Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf", 18556686752,
        "b7380c816fca5a03b746be3d42773fb9a1c4b3bb4b3a1f3d9c2aeb10a720cba6", architecture="qwen3moe", context_length=262144,
    ),
    ModelSpec(
        "deepseek-r1-distill-qwen-32b", "DeepSeek-R1-Distill-Qwen-32B", "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF", 32.764,
        "Плотная дистилляция DeepSeek R1 для математики, логики и рассуждений над кодом. Автор рекомендует температуру 0.6 и инструкции в пользовательском сообщении без system prompt. Рассуждения могут быстро расходовать max_tokens.",
        "gguf", _INSTRUCT, "рассуждения; математика; логика; код / программирование", "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "1938d05cc893a60f37be1dc16e7465038f4fca63", "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf", 19851335584,
        "ca171ca03554ee20cf67ad6b540610ae7eabb95af00c0abd36bb73542e140fb5", architecture="qwen2", context_length=131072,
    ),
    ModelSpec(
        "nemotron-super-49b-v1.5", "Llama-3.3-Nemotron-Super-49B-v1.5", "bartowski/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-GGUF", 49.867,
        "Крупная NVIDIA Nemotron для сложных рассуждений, кода и вызовов функций. Основной язык — английский; русский не заявлен. Для режима без рассуждений задайте /no_think в system prompt. Q4 весит около 28.14 ГиБ и требует значительной доли ОЗУ на GPU 16 ГБ.",
        "gguf", _INSTRUCT, "рассуждения; код / программирование; инструменты; RAG; английский; чат", "https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
        "98fc9722ebffe74e41685c477cf2982012d3f0ad", "nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-Q4_K_M.gguf", 30215579136,
        "eb619df799350250d51148874e6033f0b395b6b867291644e03225a71bda8c01", architecture="deci", context_length=131072,
    ),
)


def _get(value, key: str, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _portable(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ModelError("Недопустимый путь файла модели.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"..", "."} or any(char in part for char in ':*?<>|"') or any(ord(char) < 32 for char in part) for part in path.parts):
        raise ModelError(f"Путь выходит за каталог моделей: {value}")
    # Windows device names and trailing dots/spaces cannot safely represent HF paths.
    if any(re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part) or part.endswith((".", " ")) for part in path.parts):
        raise ModelError(f"Недопустимое имя файла для Windows: {value}")
    return path


class ModelManager:
    def __init__(self, root: Path, emit: Callable[[str], None] = print):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ModelError(f"Каталог проекта не существует: {self.root}")
        self.emit = emit
        self.manifest_path = self._state_path("models.json")

    def _state_path(self, filename: str) -> Path:
        candidate = self.root / ".local" / filename
        for path in (candidate, candidate.parent):
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ModelError("Ссылки и junction в реестре моделей не допускаются.")
        return self._inside(candidate)

    @contextmanager
    def _mutation(self):
        lock = self._state_path("models.lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        stream = lock.open("a+b")
        try:
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise ModelError("Менеджер моделей уже выполняет операцию в другом клиенте; дождитесь завершения.") from exc
        try:
            yield
        finally:
            try:
                stream.seek(0)
                if sys.platform == "win32":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()

    def _inside(self, candidate: Path, parent: Path | None = None) -> Path:
        resolved = candidate.resolve()
        try:
            resolved.relative_to((parent or self.root).resolve())
        except ValueError as exc:
            raise ModelError(f"Путь выходит за разрешенный каталог: {candidate}") from exc
        return resolved

    def _path(self, value: str) -> Path:
        return self._inside(self.root.joinpath(*_portable(value).parts))

    def _owned_path(self, value: str) -> Path:
        portable = _portable(value)
        if portable.parts[:2] != ("models", "managed") or len(portable.parts) < 5:
            raise ModelError("Менеджер может менять файлы только в models/managed/<модель>/<квантование>/<ревизия>.")
        candidate = self.root.joinpath(*portable.parts)
        for path in [candidate, *candidate.parents]:
            if path == self.root:
                break
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ModelError("Ссылки и junction в управляемом каталоге моделей не допускаются.")
        return self._inside(candidate)

    def catalog(self) -> tuple[ModelSpec, ...]:
        return CATALOG

    catalogue = catalog

    def _spec(self, model_id: str) -> ModelSpec:
        spec = next((item for item in self.catalog() if item.id == model_id), None)
        if spec is None:
            raise ModelError(f"Модель не найдена в каталоге: {model_id}")
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", spec.id) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", spec.repo_id):
            raise ModelError("Некорректная запись каталога моделей.")
        if not 0 < spec.parameters_b <= 60 or not _COMMIT.fullmatch(spec.revision):
            raise ModelError("Каталог допускает модели до 60B с закрепленной ревизией.")
        return spec

    def _metadata(self, spec: ModelSpec):
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ModelError("Установите зависимости: python -m pip install -r requirements.txt") from exc
        self.emit(f"Метаданные Hugging Face: {spec.repo_id}…")
        try:
            info = HfApi().model_info(spec.repo_id, revision=spec.revision, files_metadata=True)
        except Exception as exc:
            raise ModelError(f"Не удалось получить метаданные Hugging Face для {spec.repo_id}: {exc}") from exc
        revision = _get(info, "sha")
        if not isinstance(revision, str) or revision.lower() != spec.revision.lower():
            raise ModelError("Hugging Face вернул другую ревизию; загрузка остановлена.")
        gguf = _get(info, "gguf", {}) or {}
        parameters = _get(gguf, "total")
        if parameters is not None and (isinstance(parameters, bool) or not isinstance(parameters, (int, float)) or not 0 < parameters <= 60_000_000_000):
            raise ModelError("Метаданные модели не подтверждают лимит 60B параметров.")
        architecture = _get(gguf, "architecture")
        if architecture is not None and architecture != spec.architecture:
            raise ModelError(f"Неподдерживаемая архитектура GGUF: {architecture}; ожидалась {spec.architecture}.")
        return info

    def _groups(self, spec: ModelSpec, info) -> dict[str, list[dict]]:
        groups: dict[tuple[str, str], list[dict]] = {}
        for sibling in _get(info, "siblings", []) or []:
            filename = _get(sibling, "rfilename", "")
            if not isinstance(filename, str) or not filename.lower().endswith(".gguf"):
                continue
            match = _QUANT.search(PurePosixPath(filename).name)
            if not match:
                continue
            _portable(filename)
            quantization = match.group().upper()
            size, digest = _get(sibling, "size"), _get(_get(sibling, "lfs"), "sha256")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or not isinstance(digest, str) or not _HASH.fullmatch(digest):
                # Unknown-size or non-LFS artifacts cannot be safely installed.
                continue
            split = _SPLIT.fullmatch(filename)
            base = split.group(1) if split else filename
            item = {"filename": filename, "size": size, "sha256": digest.lower()}
            if split:
                item.update(shard=int(split.group(2)), shard_count=int(split.group(3)))
            groups.setdefault((quantization, base), []).append(item)
        available: dict[str, list[dict]] = {}
        ambiguous = set()
        for (quant, _), files in groups.items():
            files.sort(key=lambda file: file.get("shard", 0))
            count = files[0].get("shard_count")
            if count is not None and (count < 1 or any(file.get("shard_count") != count for file in files) or [file["shard"] for file in files] != list(range(1, count + 1))):
                continue
            if quant in available:
                ambiguous.add(quant)
            else:
                available[quant] = files
        for quant in ambiguous:
            del available[quant]
        default = available.get(spec.default_quantization)
        if default and spec.default_filename:
            selected = next((file for file in default if file["filename"] == spec.default_filename), None)
            if selected is None or selected["size"] != spec.default_size or selected["sha256"] != spec.default_sha256:
                raise ModelError("Метаданные закрепленной Q4-модели не совпали с каталогом.")
        return available

    def available_quantizations(self, model_id: str) -> list[dict]:
        spec = self._spec(model_id)
        groups = self._groups(spec, self._metadata(spec))
        return [{"quantization": quant, "size": sum(file["size"] for file in files), "total_size": sum(file["size"] for file in files), "files": copy.deepcopy(files)} for quant, files in sorted(groups.items())]

    quantizations = available_quantizations

    def inspect(self, model_id: str, quant: str = "Q4_K_M") -> dict:
        spec = self._spec(model_id)
        if not isinstance(quant, str) or not _QUANT.fullmatch(quant):
            raise ModelError("Неверное обозначение квантования GGUF.")
        quant = quant.upper()
        files = self._groups(spec, self._metadata(spec)).get(quant)
        if not files:
            raise ModelError(f"Полный однозначный набор {quant} не найден в {spec.repo_id}.")
        package_id = f"{spec.id}@{quant}"
        directory = f"models/managed/{spec.id}/{quant}/{spec.revision}"
        selected = [{**file, "path": f"{directory}/{file['filename']}"} for file in files]
        for file in selected:
            self._path(file["path"])
        total_size = sum(file["size"] for file in selected)
        return {
            "id": package_id, "model_id": spec.id, "name": spec.name, "repo_id": spec.repo_id,
            "quantization": quant, "revision": spec.revision, "files": selected,
            "size": total_size, "total_size": total_size, "model_path": selected[0]["path"],
            "owned_dir": directory, "family": spec.family, "source_url": spec.source_url,
            "parameters_b": spec.parameters_b, "context_length": spec.context_length,
            "managed": True,
        }

    def _load(self) -> dict:
        self.manifest_path = self._state_path("models.json")
        if not self.manifest_path.exists():
            return {"version": 1, "packages": {}}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("packages"), dict):
                raise ValueError("format")
            for key, record in data["packages"].items():
                if not isinstance(record, dict) or record.get("id") != key:
                    raise ValueError("package")
                self._validate_record(record)
            return data
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ModelError("Реестр .local/models.json поврежден; сохраните его копию и восстановите записи перед управлением моделями.") from exc

    def _save(self, data: dict) -> None:
        self.manifest_path = self._state_path("models.json")
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._inside(self.manifest_path.with_name(f"models.{uuid.uuid4().hex}.tmp"))
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.manifest_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _validate_record(self, record: dict) -> None:
        for key in ("id", "model_id", "name", "repo_id", "quantization", "revision", "model_path", "executable", "family"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ModelError(f"Неполная запись реестра моделей: {key}.")
        if record.get("family") not in {"flash", "glm4", "gguf"} or not isinstance(record.get("managed"), bool):
            raise ModelError("Неполная запись реестра моделей: family/managed.")
        files = record.get("files")
        if not isinstance(files, list) or not files or any(not isinstance(file, dict) for file in files):
            raise ModelError("Неполная запись реестра моделей: files.")
        if record["model_path"] != files[0].get("path"):
            raise ModelError("Путь запуска модели отличается от проверенного файла.")
        self._path(record["executable"])
        if record["managed"]:
            if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", record["model_id"]) or not _QUANT.fullmatch(record["quantization"]) or not _COMMIT.fullmatch(record["revision"]):
                raise ModelError("Некорректный идентификатор управляемого пакета.")
            expected = f"models/managed/{record['model_id']}/{record['quantization']}/{record['revision']}"
            if record.get("owned_dir") != expected or record["id"] != f"{record['model_id']}@{record['quantization']}":
                raise ModelError("Каталог модели не соответствует идентификатору пакета.")
            directory = self._owned_path(expected)
        paths = set()
        for file in files:
            path = self._path(file.get("path"))
            if path in paths:
                raise ModelError("В реестре моделей повторяется файл.")
            paths.add(path)
            size, digest = file.get("size"), file.get("sha256")
            if record["managed"]:
                self._inside(self._owned_path(file["path"]), directory)
                if not isinstance(file.get("filename"), str) or f"{expected}/{file['filename']}" != file["path"]:
                    raise ModelError("Файл не соответствует пути внутри пакета.")
                if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or not isinstance(digest, str) or not _HASH.fullmatch(digest):
                    raise ModelError("У управляемого файла нет корректного размера и SHA256.")
            elif (size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0)) or (digest is not None and (not isinstance(digest, str) or not _HASH.fullmatch(digest))):
                raise ModelError("Некорректные метаданные ранее установленной модели.")
        if record["managed"] and record.get("total_size") != sum(file["size"] for file in files):
            raise ModelError("Суммарный размер пакета в реестре неверен.")
        context = record.get("context_length")
        if context is not None and (isinstance(context, bool) or not isinstance(context, int) or context <= 0):
            raise ModelError("Некорректное ограничение контекста в реестре моделей.")
        parameters = record.get("parameters_b")
        if parameters is not None and (isinstance(parameters, bool) or not isinstance(parameters, (int, float)) or not 0 < parameters <= 60):
            raise ModelError("Некорректное число параметров модели в реестре.")

    def _legacy(self, config: dict | None) -> dict | None:
        lock_path = self._inside(self.root / ".local" / "install.json")
        lock = None
        if lock_path.is_file():
            try:
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
                if not isinstance(lock, dict) or not lock.get("model_path"):
                    lock = None
            except (OSError, ValueError):
                pass
        if lock is None and config and config.get("backend") == "llama_cpp" and not config.get("model_package"):
            server = config.get("server", {})
            lock = {"model_path": server.get("model_path"), "executable": server.get("executable"), "repo_id": config.get("target_model", config.get("model")), "model": {}}
        if not lock or not lock.get("model_path"):
            return None
        try:
            model_path = lock["model_path"]
            self._path(model_path)
            executable = lock.get("executable", "")
            if executable:
                self._path(executable)
            provenance = lock.get("model") or {}
            compatibility = provenance.get("llama_cpp_compatibility") or {}
            repo = lock.get("repo_id", "local")
            file = {"path": model_path, "filename": PurePosixPath(model_path).name,
                    "size": compatibility.get("size", provenance.get("size")),
                    "sha256": compatibility.get("sha256", provenance.get("sha256"))}
            family = "flash" if "4.7" in str(repo) else "glm4"
            key = "legacy-" + hashlib.sha256(model_path.encode("utf-8")).hexdigest()[:12]
            record = {"id": key, "model_id": "legacy-local", "name": f"Локальная: {repo}", "repo_id": repo,
                      "quantization": provenance.get("quantization", "Q4_K_M" if "Q4_K_M" in model_path.upper() else "GGUF"),
                      "files": [file], "model_path": model_path, "executable": executable,
                      "size": file.get("size") or 0, "total_size": file.get("size") or 0, "managed": False,
                      "family": family, "source_url": provenance.get("source", ""),
                      "revision": provenance.get("revision", "legacy")}
            if lock.get("chat_template_file"):
                record["chat_template_file"] = lock["chat_template_file"]
            self._validate_record(record)
            return record
        except (ModelError, TypeError, AttributeError):
            return None

    def _state(self, record: dict, verify: bool = False) -> str:
        try:
            self._validate_record(record)
            if record.get("managed"):
                directory = self._owned_path(record["owned_dir"])
                if not directory.is_dir():
                    return "missing"
                marker = self._owned_path(f"{record['owned_dir']}/{_MARKER}")
                if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != self._identity(record):
                    return "corrupt"
            if not record.get("executable") or not self._path(record["executable"]).is_file():
                return "missing"
            for file in record["files"]:
                path = self._path(file["path"])
                if not path.is_file():
                    return "missing"
                if file.get("size") is not None and path.stat().st_size != file["size"]:
                    return "corrupt"
                if verify and file.get("sha256") and _digest(path, self.emit) != file["sha256"]:
                    return "corrupt"
            return "ready"
        except (OSError, ModelError, TypeError, ValueError):
            return "corrupt"

    def _active(self, record: dict, config: dict | None) -> bool:
        if not config or config.get("backend") != "llama_cpp":
            return False
        path = config.get("server", {}).get("model_path")
        if path:
            try:
                return self._path(path) == self._path(record["model_path"])
            except ModelError:
                pass
        return config.get("model_package") == record["id"] if not path else False

    def installed(self, config: dict | None = None, verify: bool = False) -> list[dict]:
        records = copy.deepcopy(list(self._load()["packages"].values()))
        legacy = self._legacy(config)
        if legacy:
            records = [record for record in records if record["id"] != legacy["id"]]
            records.append(legacy)
        for record in records:
            record["status"] = record["state"] = self._state(record, verify)
            record["active"] = self._active(record, config)
        return records

    def _identity(self, record: dict) -> dict:
        keys = ["id", "repo_id", "revision", "quantization", "files", "model_path"]
        if record.get("managed"):
            keys.append("owned_dir")
        return {key: copy.deepcopy(record[key]) for key in keys}

    def _prepare(self, plan: dict) -> Path:
        directory = self._owned_path(plan["owned_dir"])
        marker = self._owned_path(f"{plan['owned_dir']}/{_MARKER}")
        expected = self._identity(plan)
        if directory.exists():
            try:
                saved = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ModelError(f"Каталог {directory} не принадлежит менеджеру моделей; переименуйте его.") from exc
            if saved != expected:
                raise ModelError("В каталоге модели другая установка; существующие файлы сохранены.")
        else:
            directory.mkdir(parents=True, exist_ok=False)
            with marker.open("x", encoding="utf-8") as stream:
                json.dump(expected, stream, ensure_ascii=False, indent=2)
        return directory

    def _url(self, plan: dict, file: dict) -> str:
        return f"https://huggingface.co/{plan['repo_id']}/resolve/{plan['revision']}/{quote(file['filename'], safe='/')}?download=true"

    def _needed(self, plan: dict) -> int:
        required = 2 * _GIB  # Runtime archive/extraction and a working-space margin.
        for file in plan["files"]:
            path = self._path(file["path"])
            if path.is_file() and path.stat().st_size == file["size"]:
                continue
            reserved = 0
            partial = self._inside(path.with_name(path.name + ".partial"))
            checkpoint = self._inside(path.with_name(path.name + ".partial.json"))
            try:
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                if saved.get("size") == file["size"] and saved.get("sha256") == file["sha256"] and saved.get("url") == self._url(plan, file) and partial.is_file():
                    reserved = min(partial.stat().st_size, file["size"])
            except (OSError, ValueError, AttributeError):
                pass
            required += file["size"] - reserved
        return required

    def install(self, model_id: str, quant: str = "Q4_K_M", emit: Callable[[str], None] | None = None) -> dict:
        with self._mutation():
            return self._install(model_id, quant, emit)

    def _install(self, model_id: str, quant: str, emit: Callable[[str], None] | None) -> dict:
        if os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}:
            raise ModelError("Установка рассчитана на Windows x64 с Vulkan; модели GGUF нельзя запустить этим установщиком на другой платформе.")
        plan = self.inspect(model_id, quant)
        data = self._load()
        existing = data["packages"].get(plan["id"])
        if existing and self._identity(existing) != self._identity(plan):
            raise ModelError("Этот пакет уже установлен с другой ревизией. Удалите его явно перед установкой другой версии.")
        progress = emit or self.emit
        free, needed = shutil.disk_usage(self.root).free, self._needed(plan)
        progress(f"{plan['name']} / {plan['quantization']}: {plan['total_size'] / _GIB:.2f} ГиБ. Свободно {free / _GIB:.1f} ГиБ.")
        if free < needed:
            raise ModelError(f"Недостаточно места: нужно еще около {needed / _GIB:.1f} ГиБ, свободно {free / _GIB:.1f} ГиБ.")
        self._prepare(plan)
        try:
            for file in plan["files"]:
                path = self._owned_path(file["path"])
                self._inside(path, self._path(plan["owned_dir"]))
                # These files are also touched by the downloader and must be guarded.
                for suffix in (".partial", ".partial.json"):
                    self._inside(self._owned_path(file["path"] + suffix), self._path(plan["owned_dir"]))
                progress(f"Файл: {file['filename']}")
                download_file(self._url(plan, file), path, file["size"], file["sha256"], emit=progress, workers=16)
                if path.stat().st_size != file["size"]:
                    raise ModelError(f"Размер или SHA256 модели не совпал: {file['filename']}.")
            executable = self._inside(_install_runtime(self.root, progress))
            if not executable.is_file():
                raise ModelError("Установка среды не создала llama-server.exe.")
            record = {**plan, "executable": executable.relative_to(self.root).as_posix()}
            data["packages"][record["id"]] = record
            self._save(data)
        except (DownloadError, SetupError, OSError) as exc:
            raise ModelError(f"Установка остановлена: {exc}. Повторный запуск продолжит проверенную загрузку.") from exc
        progress("Модель установлена. Выберите «Активировать», чтобы использовать ее в новом чате.")
        return {**copy.deepcopy(record), "status": "ready", "state": "ready", "active": False}

    def activate(self, key: str, config: dict, config_path: Path | None = None) -> dict:
        with self._mutation():
            return self._activate(key, config, config_path)

    def _activate(self, key: str, config: dict, config_path: Path | None) -> dict:
        data = self._load()
        legacy = self._legacy(config)
        record = data["packages"].get(key)
        if legacy and legacy["id"] == key:
            record = legacy
        if record is None:
            raise ModelError(f"Установленный пакет не найден: {key}")
        self.emit("Проверка выбранной модели перед активацией…")
        state = self._state(record, verify=True)
        if state != "ready":
            raise ModelError("Модель или среда отсутствует либо повреждена; активация отменена.")
        updated = copy.deepcopy(config)
        updated.update(backend="llama_cpp", target_model=record["repo_id"], model_package=record["id"])
        try:
            endpoint = urlsplit(str(updated.get("base_url", "")))
            compatible = endpoint.scheme == "http" and endpoint.hostname in {"127.0.0.1", "localhost"} and endpoint.path == "/v1" and not endpoint.query and not endpoint.fragment and endpoint.username is None and endpoint.password is None and (endpoint.port or 80) > 0
        except ValueError:
            compatible = False
        if not compatible:
            updated["base_url"] = DEFAULT_CONFIG["base_url"]
        identity = json.dumps(self._identity(record), sort_keys=True).encode("utf-8")
        updated["model"] = "glm-package-" + hashlib.sha256(identity).hexdigest()[:20]
        server = updated["server"]
        server.update(model_path=record["model_path"], executable=record["executable"])
        template_options = {"--chat-template", "--chat-template-file", "--chat-template-kwargs"}
        old_args = server.get("extra_args", [])
        new_args = []
        index = 0
        while index < len(old_args):
            argument = old_args[index]
            option = argument.split("=", 1)[0].replace("_", "-")
            if option in template_options:
                index += 1 if "=" in argument else 2
            else:
                new_args.append(argument)
                index += 1
        request_extra = copy.deepcopy(updated.get("request_extra", {}))
        if record.get("family") == "flash":
            from .gguf_compat import TEMPLATE_SHA256
            template = "llmopenchat/templates/GLM-4.7-Flash.jinja"
            template_path = self._path(template)
            if not template_path.is_file() or _sha256(template_path) != TEMPLATE_SHA256:
                raise ModelError("Шаблон GLM-4.7-Flash отсутствует или изменен; восстановите файл проекта.")
            new_args.extend(["--chat-template-file", template])
            kwargs = request_extra.setdefault("chat_template_kwargs", {})
            if not isinstance(kwargs, dict):
                raise ModelError("request_extra.chat_template_kwargs должен быть JSON-объектом перед активацией Flash.")
            kwargs.setdefault("enable_thinking", False)
        else:
            request_extra.pop("chat_template_kwargs", None)
        server["extra_args"] = new_args
        if record.get("context_length"):
            server["context_size"] = min(server["context_size"], record["context_length"])
        updated["request_extra"] = request_extra
        # Retain the previous legacy profile even if its config was its only source.
        changed = False
        if legacy and data["packages"].get(legacy["id"]) != legacy:
            data["packages"][legacy["id"]] = legacy
            changed = True
        if changed:
            self._save(data)
        if config_path is not None:
            write_config(Path(config_path), updated)
        return updated

    def remove(self, key: str, config: dict | None = None) -> None:
        with self._mutation():
            return self._remove(key, config)

    def _remove(self, key: str, config: dict | None) -> None:
        data = self._load()
        record = data["packages"].get(key)
        if record is None:
            raise ModelError(f"Установленный пакет не найден: {key}")
        if not record.get("managed"):
            raise ModelError("Ранее установленные и вручную добавленные модели не удаляются менеджером.")
        if config is None:
            config_path = self._inside(self.root / "config.json")
            if config_path.is_file():
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
                except (OSError, ValueError) as exc:
                    raise ModelError("Не удалось проверить активную модель в config.json; удаление отменено.") from exc
                if not isinstance(config, dict):
                    raise ModelError("config.json поврежден; удаление отменено.")
        if self._active(record, config):
            raise ModelError("Нельзя удалить активную модель. Сначала активируйте другую модель.")
        directory = self._owned_path(record["owned_dir"])
        managed_root = self._path("models/managed")
        self._inside(directory, managed_root)
        if directory == managed_root:
            raise ModelError("Нельзя удалять общий каталог моделей.")
        if not directory.exists():
            del data["packages"][key]
            self._save(data)
            return
        marker = self._owned_path(f"{record['owned_dir']}/{_MARKER}")
        try:
            saved = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelError("Нет подтверждения владения файлами; удаление отменено.") from exc
        if saved != self._identity(record):
            raise ModelError("Метаданные владения моделью изменены; удаление отменено.")
        # Validate every path before deleting anything. Unknown files are preserved.
        targets = []
        parents = {directory}
        for file in record["files"]:
            path = self._inside(self._owned_path(file["path"]), directory)
            parents.update(path.parents)
            targets.extend([path, self._inside(self._owned_path(file["path"] + ".partial"), directory), self._inside(self._owned_path(file["path"] + ".partial.json"), directory)])
        for path in targets:
            if path.exists() and not path.is_file():
                raise ModelError(f"Вместо файла найден каталог; удаление отменено: {path}")
        try:
            for path in targets:
                path.unlink(missing_ok=True)
            del data["packages"][key]
            self._save(data)
            marker.unlink()
            for parent in sorted(parents, key=lambda path: len(path.parts), reverse=True):
                if parent == directory or directory in parent.parents:
                    try:
                        parent.rmdir()
                    except OSError:
                        pass
        except OSError as exc:
            raise ModelError(f"Не удалось удалить пакет: {exc}. Запись сохранена для повторной попытки.") from exc
