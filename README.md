# llmopenchat

Консольный чат и Telegram-бот для локальных GGUF-моделей и OpenAI-совместимых API. Клиент поддерживает потоковые ответы, историю диалогов, инструменты, поиск по документам и сценарии обработки текста.

Репозиторий содержит исходный код, скрипты запуска, тесты и примеры конфигурации. Веса моделей, серверные бинарники, окружения Python, личные настройки, токены, журналы и сохранённые диалоги устанавливаются или создаются отдельно.

## Возможности

- Локальные модели через управляемый сервер llama.cpp с Vulkan; подключение к внешнему API и установленному vLLM на Linux.
- Каталог GGUF-моделей: загрузка выбранного квантования, проверка SHA256, переключение и удаление пакетов.
- Терминальный чат с горячими клавишами, сохранением и продолжением диалогов.
- Telegram-бот с персональными диалогами, файлами и базой знаний пользователей.
- Веб, чтение и запись файлов, ограниченный PowerShell — с подтверждением каждого вызова в чате.
- RAG: локальный поиск BM25 по текстам, исходникам, HTML, PDF и DOCX.
- Сценарии с проверкой JSON Schema: текст, stdin, файлы, отправка результата POST и входящий вебхук.
- Подготовка JSONL-датасетов и отдельный процесс LoRA/QLoRA-дообучения.

## Быстрый запуск в Windows

Нужен Python **3.10 или новее**; клиент проверяется на Python 3.12. Автоматический установщик локального сервера предназначен для Windows x64. Для Vulkan нужен совместимый видеодрайвер; объём RAM, VRAM и свободного диска зависит от выбранной модели.

Из каталога проекта запустите:

```powershell
.\Install.cmd --model qwen3-0.6b
.\Start-Chat.cmd
```

Первый скрипт создаёт `.venv`, устанавливает зависимости из `requirements.lock.txt`, **скачивает** сервер llama.cpp и выбранные веса, проверяет контрольные суммы и записывает `config.json`. Модель `qwen3-0.6b` — компактный вариант из встроенного каталога; для более сложных задач можно выбрать другую модель.

Без `--model` установщик скачивает Huihui GLM-4.7-Flash из источника по умолчанию. Для просмотра каталога после создания окружения:

```powershell
.\.venv\Scripts\python.exe -m llmopenchat models catalog
# Пример установки другой модели и квантования:
.\Install.cmd --model qwen3-4b-instruct-2507 --quant Q5_K_M
```

Повтор той же команды установки продолжает прерванное скачивание. При последующих запусках используются локальные файлы. Сервер запускается при входе в чат или запуске бота и останавливается при завершении клиента. Уже работающий совместимый API используется без его остановки.

## Подключение к внешнему API

Для внешнего сервера подготовьте только окружение Python:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
Copy-Item -LiteralPath config.example.json -Destination config.json
```

Измените в `config.json` следующие поля; остальные настройки можно оставить из примера:

```json
{
  "backend": "external",
  "base_url": "https://your-server.example/v1",
  "model": "your-model-id",
  "target_model": "your-model-id",
  "request_extra": {}
}
```

`model` должен совпадать с одним из ID в ответе сервера `/v1/models`. `target_model` используется как название источника в интерфейсе. Пустой `request_extra` убирает параметры шаблона, специфичные для локального профиля. Если API требует ключ, задайте его в окружении текущего PowerShell:

```powershell
$env:LLMOPENCHAT_API_KEY = "YOUR_API_KEY"
.\Start-Chat.cmd --config config.json chat
```

При `backend: "external"` скачивание весов и локального сервера не требуется. Пример отдельного профиля GLM-5.3 находится в `config.glm53.example.json`; адрес и ID модели в нём нужно заменить своими. Сообщения передаются выбранному серверу.

## Команды и настройки

Все примеры выполняются из каталога проекта:

```powershell
.\.venv\Scripts\python.exe -m llmopenchat                       # Главное меню
.\.venv\Scripts\python.exe -m llmopenchat chat                  # Сразу открыть чат
.\.venv\Scripts\python.exe -m llmopenchat ask "Привет"          # Один ответ и выход
.\.venv\Scripts\python.exe -m llmopenchat history               # История диалогов
.\.venv\Scripts\python.exe -m llmopenchat models                # Менеджер моделей
.\.venv\Scripts\python.exe -m llmopenchat doctor                # Диагностика установки
.\.venv\Scripts\python.exe -m llmopenchat --help                # Справка CLI
```

Глобальный `--config ПУТЬ` указывается **перед** командой. Например: `python -m llmopenchat --config config.json chat`. `Start-Chat.cmd` передаёт клиенту все аргументы. При активном `.venv` можно использовать `python` вместо полного пути к интерпретатору.

`chat` и `ask` принимают `--no-autostart`, `--max-tokens N`, `--show-reasoning` и `--rag`. `chat --session ПУТЬ` продолжает сохранённый диалог; `ask` историю сеанса не сохраняет. Подробные параметры: `python -m llmopenchat КОМАНДА --help`.

| Настройка `config.json` | Назначение |
| --- | --- |
| `backend`, `base_url`, `model` | Способ подключения, адрес API и ID модели |
| `temperature`, `max_tokens` | Вариативность и предел длины ответа |
| `system_prompt`, `show_reasoning` | Инструкция и отображение рассуждений |
| `request_timeout`, `request_extra` | Таймаут и дополнительные параметры запроса |
| `server.*` | Исполняемый файл, GGUF, размер контекста и параметры локального сервера |
| `rag.*` | Размер фрагментов и объём найденного контекста |

Поля, отсутствующие в конфигурации, дополняются значениями из `llmopenchat/config.py`. Объект `request_extra` заменяется целиком. Относительные пути сервера считаются от каталога проекта. После ручного изменения настроек перезапустите клиент.

Внутри чата `/help` показывает все команды. `F1` — справка, `F2` — главное меню, `F3` — инструменты, `Alt+Enter` — новая строка, `Ctrl+C` — отмена, `Ctrl+Q` — выход. `/save`, `/load` и `/history` управляют диалогами; `/thinking`, `/system` и `/clear` меняют состояние чата.

Веб, файлы и PowerShell включаются через `/tools` для текущего диалога. Перед каждым вызовом клиент показывает аргументы и запрашивает подтверждение. `/status` показывает состояние инструментов и RAG.

## Ответы по документам

Укажите путь к собственному документу или каталогу:

```powershell
.\.venv\Scripts\python.exe -m llmopenchat rag add "C:\Documents\notes.txt"
.\.venv\Scripts\python.exe -m llmopenchat rag list
.\.venv\Scripts\python.exe -m llmopenchat rag search "нужная тема"
.\.venv\Scripts\python.exe -m llmopenchat chat --rag
```

В чате доступны `/rag add ПУТЬ`, `/rag on`, `/rag off`, `/rag list`, `/rag search ЗАПРОС`, `/rag remove ID` и `/rag clear`. Поиск выполняется локально по словам, без отдельной embedding-модели. Сканированные PDF требуют внешнего OCR. Удаление документа из индекса сохраняет исходный файл.

## Сценарии обработки текста

Пример анализа новости находится в `examples/scenarios/news/`:

```powershell
.\.venv\Scripts\python.exe -m llmopenchat scenario validate examples/scenarios/news/scenario.json
.\.venv\Scripts\python.exe -m llmopenchat scenario run examples/scenarios/news/scenario.json --text "Текст новости"
.\.venv\Scripts\python.exe -m llmopenchat scenario init scenarios/my/scenario.json
```

Сценарий задаёт инструкцию, JSON Schema ответа, параметры генерации и способ выдачи результата. Вместо `--text` можно передать `--input-file ПУТЬ` или UTF-8 stdin. Сетевые адреса разрешаются списком origins в соседнем `hosts.txt`; пример по умолчанию запрещает сетевые действия. Разрешения инструментов сценария задаются его конфигурацией.

Для входящего вебхука установите `LLMOPENCHAT_WEBHOOK_TOKEN` и запустите `scenario serve ПУТЬ`. По умолчанию он слушает `127.0.0.1:8765` и принимает `POST /run` с JSON `{"text": "..."}` и Bearer-токеном. Дополнительные параметры доступны через `scenario serve --help`.

## Telegram-бот

Создайте в корне проекта два личных файла:

- `credits.txt` — только токен вашего бота, полученный у BotFather.
- `whitelist.txt` — разрешённые Telegram username через запятую, например `@alice, bob_user`. Символ `@` необязателен, регистр не учитывается.

Запуск: `.\Start-Telegram.cmd` или `python -m llmopenchat telegram`. Бот использует общий профиль модели. Другие пути можно передать флагами `--credits` и `--whitelist`; `--config` ставится перед `telegram`.

У каждого пользователя отдельные диалоги, рабочая папка и RAG. Отправленный боту документ добавляется в базу знаний; `/rag on` включает поиск. `/tools` управляет инструментами с подтверждением кнопкой. Модель и сервер общие: переключение через `/models` влияет на последующие ответы всем пользователям. Команды: `/help`, `/id`, `/history`, `/save`, `/load`, `/cancel`.

## Подготовка данных и дообучение

Формат датасета показан в `examples/sft.example.jsonl`. Создайте свой файл и проверьте его:

```powershell
.\.venv\Scripts\python.exe -m llmopenchat sft init .local/sft/data.jsonl
.\.venv\Scripts\python.exe -m llmopenchat sft validate .local/sft/data.jsonl
```

Исправьте примеры и отметьте проверенные ответы `status: "approved"`. `sft prepare` включает в обучение только одобренные данные. Для него требуется явный `--base-model` — HF ID или каталог исходной модели Transformers. GGUF не подходит для исходного checkpoint обучения.

```powershell
.\.venv\Scripts\python.exe -m llmopenchat sft prepare .local/sft/data.jsonl --output .local/sft/job --base-model Qwen/Qwen3-0.6B
.\.venv\Scripts\python.exe -m llmopenchat sft check --python .venv-sft/Scripts/python.exe
.\.venv\Scripts\python.exe -m llmopenchat sft train .local/sft/job/job.json --python .venv-sft/Scripts/python.exe
```

Перед `check` и `train` создайте отдельное `.venv-sft` и установите подходящий PyTorch и зависимости из `requirements-training.txt`. Обучение может скачать исходный checkpoint; режим LoRA/QLoRA выбирается при `prepare`. `sft from-session` позволяет подготовить кандидатов из собственных будущих диалогов; они также требуют ручной проверки.

## Локальные данные и проверка

| Путь | Содержимое |
| --- | --- |
| `.venv/`, `.venv-sft/` | Окружения клиента и обучения |
| `models/` | Скачанные GGUF и сведения о загрузке |
| `.local/runtime/` | Установленный сервер llama.cpp |
| `.local/logs/` | Журналы сервера |
| `.local/sessions/`, `.local/input-history` | Диалоги и история терминального ввода |
| `.local/rag/` | Индекс документов |
| `.local/telegram/users/` | Данные пользователей Telegram |
| `.local/sft/` | Собственные датасеты и задания из примеров выше |
| `config.json`, `credits.txt`, `whitelist.txt` | Личные настройки и доступы |

Окружения, веса, `.local/` и личные файлы перечислены в `.gitignore` и не входят в исходную поставку. Диагностика: `python -m llmopenchat doctor`; журнал локального сервера — `.local/logs/server.log`.

Проверка клиента и управления процессами без загрузки модели:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Лицензии

Лицензия проекта — GNU GPLv3: см. [LICENSE](LICENSE). Шаблон `llmopenchat/templates/GLM-4.7-Flash.jinja` поставляется с лицензией upstream llama.cpp: [LICENSE.llama.cpp](llmopenchat/templates/LICENSE.llama.cpp). У скачиваемых моделей собственные лицензии, указанные авторами.
