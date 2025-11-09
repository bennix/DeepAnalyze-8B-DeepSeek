import openai
import json
import os
import shutil
import re
import io
import contextlib
import traceback
from pathlib import Path
from urllib.parse import quote
import subprocess
import sys
import tempfile
import requests
import threading
import http.server
from functools import partial
import socketserver
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import httpx
import uvicorn
import os
import re
import json
from fastapi.responses import StreamingResponse
import os
import re
from copy import deepcopy
import openai
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse

import re

os.environ.setdefault("MPLBACKEND", "Agg")


def ensure_common_imports(code_str: str) -> str:
    """Inject missing common data-analysis imports into Python code.

    - Adds `import pandas as pd` if `pd.` used or no pandas import and 'pandas' operations expected.
    - Adds `import matplotlib.pyplot as plt` if `plt.` used or plotting functions detected without import.
    - Adds `import seaborn as sns` if `sns.` used or seaborn function names appear.
    - Ensures non-interactive backend via `matplotlib.use('Agg')` when plt exists.
    """
    try:
        lines = code_str.splitlines()
        src = "\n".join(lines)

        has_pd = bool(re.search(r"(^|\n)\s*import\s+pandas\s+as\s+pd\b|(^|\n)\s*from\s+pandas\b", src))
        has_plt = bool(re.search(r"(^|\n)\s*import\s+matplotlib\.pyplot\s+as\s+plt\b|(^|\n)\s*from\s+matplotlib\s+import\s+pyplot\b", src))
        has_sns = bool(re.search(r"(^|\n)\s*import\s+seaborn\s+as\s+sns\b|(^|\n)\s*from\s+seaborn\b", src))

        needs_pd = (not has_pd) and bool(re.search(r"\bpd\.|read_csv\(|read_excel\(", src))
        needs_plt = (not has_plt) and bool(re.search(r"\bplt\.|plot\(|bar\(|hist\(|scatter\(", src))
        needs_sns = (not has_sns) and bool(re.search(r"\bsns\.|seaborn|scatterplot\(|lineplot\(|barplot\(", src))

        inject = []
        # Prefer ordering: pandas, matplotlib, seaborn
        if needs_pd:
            inject.append("import pandas as pd")
        if needs_plt:
            inject.append("import matplotlib")
            inject.append("matplotlib.use('Agg')")
            inject.append("import matplotlib.pyplot as plt")
        if needs_sns:
            inject.append("import seaborn as sns")

        if not inject:
            return code_str

        # Place imports at the very top (before code), keep original content intact
        return "\n".join(inject) + "\n" + code_str
    except Exception:
        # Fallback: do not alter code if detection fails
        return code_str

def execute_code(code_str):
    import io
    import contextlib
    import traceback

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
            stderr_capture
        ):
            # 自动补全常用数据分析库导入
            code_str = ensure_common_imports(code_str)
            exec(code_str, {})
        output = stdout_capture.getvalue()
        if stderr_capture.getvalue():
            output += stderr_capture.getvalue()
        return output
    except Exception as exec_error:
        code_lines = code_str.splitlines()
        tb_lines = traceback.format_exc().splitlines()
        error_line = None
        for line in tb_lines:
            if 'File "<string>", line' in line:
                try:
                    line_num = int(line.split(", line ")[1].split(",")[0])
                    error_line = line_num
                    break
                except (IndexError, ValueError):
                    continue
        error_message = f"Traceback (most recent call last):\n"
        if error_line is not None and 1 <= error_line <= len(code_lines):
            error_message += f'  File "<string>", line {error_line}, in <module>\n'
            error_message += f"    {code_lines[error_line-1].strip()}\n"
        error_message += f"{type(exec_error).__name__}: {str(exec_error)}"
        if stderr_capture.getvalue():
            error_message += f"\n{stderr_capture.getvalue()}"
        return f"[Error]:\n{error_message.strip()}"


def execute_code_safe(
    code_str: str, workspace_dir: str = None, timeout_sec: int = 120
) -> str:
    """在独立进程中执行代码，支持超时，避免阻塞主进程。"""
    if workspace_dir is None:
        workspace_dir = WORKSPACE_BASE_DIR
    exec_cwd = os.path.abspath(workspace_dir)
    os.makedirs(exec_cwd, exist_ok=True)
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".py", dir=exec_cwd)
        os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(ensure_common_imports(code_str))
        print(
            f"[exec] Running script: {tmp_path} (timeout={timeout_sec}s) cwd={exec_cwd}"
        )
        # 在子进程中设置无界面环境变量，避免 GUI 后端
        child_env = os.environ.copy()
        child_env.setdefault("MPLBACKEND", "Agg")
        child_env.setdefault("QT_QPA_PLATFORM", "offscreen")
        child_env.pop("DISPLAY", None)

        completed = subprocess.run(
            [sys.executable, tmp_path],
            cwd=exec_cwd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=child_env,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        return output
    except subprocess.TimeoutExpired:
        return f"[Timeout]: execution exceeded {timeout_sec} seconds"
    except Exception as e:
        return f"[Error]: {str(e)}"
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def execute_code_stream(
    code_str: str, workspace_dir: str = None, timeout_sec: int = 120
):
    """以流式方式执行代码，逐步返回输出内容。

    - 在独立子进程中运行脚本，实时读取 stdout/stderr。
    - 自动注入无界面环境变量，避免 GUI 阻塞。
    - 超时后终止进程并返回超时提示。
    """
    import time
    import queue

    if workspace_dir is None:
        workspace_dir = WORKSPACE_BASE_DIR
    exec_cwd = os.path.abspath(workspace_dir)
    os.makedirs(exec_cwd, exist_ok=True)
    tmp_path = None

    # 线程安全队列，用于收集输出
    q: queue.Queue[str] = queue.Queue()

    def _reader(stream, prefix: str = ""):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                # 逐行推送，保留原始换行
                q.put((prefix + line))
        except Exception as _:
            pass

    start_ts = time.time()
    try:
        # 写入临时脚本
        fd, tmp_path = tempfile.mkstemp(suffix=".py", dir=exec_cwd)
        os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(ensure_common_imports(code_str))

        # 子进程环境
        child_env = os.environ.copy()
        child_env.setdefault("MPLBACKEND", "Agg")
        child_env.setdefault("QT_QPA_PLATFORM", "offscreen")
        child_env.pop("DISPLAY", None)

        # 使用 -u 强制无缓冲输出
        proc = subprocess.Popen(
            [sys.executable, "-u", tmp_path],
            cwd=exec_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=child_env,
            bufsize=1,
        )

        # 启动读取线程
        t_out = threading.Thread(target=_reader, args=(proc.stdout, ""), daemon=True)
        t_err = threading.Thread(target=_reader, args=(proc.stderr, ""), daemon=True)
        t_out.start()
        t_err.start()

        last_heartbeat = 0.0
        heartbeat_interval = 1.0

        # 主循环：从队列取出并 yield；间隔心跳提示
        while True:
            # 超时控制
            if time.time() - start_ts > timeout_sec:
                try:
                    proc.kill()
                except Exception:
                    pass
                yield "[Timeout]: execution exceeded %d seconds\n" % timeout_sec
                break

            try:
                line = q.get(timeout=0.2)
                yield line
            except queue.Empty:
                # 若进程仍在运行且一段时间没有输出，给心跳提示
                if proc.poll() is None:
                    now = time.time()
                    if now - last_heartbeat >= heartbeat_interval:
                        last_heartbeat = now
                        yield "[status] Executing...\n"
                else:
                    # 进程已结束，队列也空，退出循环
                    break

        # 排空剩余输出
        while True:
            try:
                line = q.get_nowait()
                yield line
            except queue.Empty:
                break

    except Exception as e:
        yield f"[Error]: {str(e)}\n"
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# API endpoint and model path (auto-detect Ollama first, fallback to mock server)
def _detect_ollama_base() -> tuple[str, str]:
    try:
        # Prefer OpenAI-compatible endpoint when available
        resp = requests.get("http://localhost:11434/api/tags", timeout=1.0)
        if resp.ok:
            data = resp.json()
            models = {m.get("name") for m in data.get("models", [])}
            if "deepseek-r1:1.5b" in models:
                return "http://localhost:11434/v1", "deepseek-r1:1.5b"
    except Exception:
        pass
    # Fallback to local mock server
    return "http://localhost:8000/v1", "RUC-DataLab/DeepAnalyze-8B"

API_BASE, MODEL_PATH = _detect_ollama_base()


# Initialize OpenAI client
client = openai.OpenAI(base_url=API_BASE, api_key="dummy")

# Workspace directory
WORKSPACE_BASE_DIR = "workspace"
HTTP_SERVER_PORT = 8100
HTTP_SERVER_BASE = (
    f"http://localhost:{HTTP_SERVER_PORT}"  # you can replace localhost to your local ip
)


def get_session_workspace(session_id: str) -> str:
    """返回指定 session 的 workspace 路径（workspace/{session_id}/）。"""
    if not session_id:
        session_id = "default"
    session_dir = os.path.join(WORKSPACE_BASE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def build_download_url(rel_path: str) -> str:
    try:
        encoded = quote(rel_path, safe="/")
    except Exception:
        encoded = rel_path
    return f"{HTTP_SERVER_BASE}/{encoded}"


# FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def start_http_server():
    """启动HTTP文件服务器（不修改全局工作目录）。"""
    os.makedirs(WORKSPACE_BASE_DIR, exist_ok=True)
    handler = partial(
        http.server.SimpleHTTPRequestHandler, directory=WORKSPACE_BASE_DIR
    )
    with socketserver.TCPServer(("", HTTP_SERVER_PORT), handler) as httpd:
        print(f"HTTP Server serving {WORKSPACE_BASE_DIR} at port {HTTP_SERVER_PORT}")
        httpd.serve_forever()


# Start HTTP server in a separate thread
threading.Thread(target=start_http_server, daemon=True).start()


def collect_file_info(directory: str) -> str:
    """收集文件信息"""
    all_file_info_str = ""
    dir_path = Path(directory)
    if not dir_path.exists():
        return ""

    files = sorted([f for f in dir_path.iterdir() if f.is_file()])
    for idx, file_path in enumerate(files, start=1):
        size_bytes = os.path.getsize(file_path)
        size_kb = size_bytes / 1024
        size_str = f"{size_kb:.1f}KB"
        file_info = {"name": file_path.name, "size": size_str}
        file_info_str = json.dumps(file_info, indent=4, ensure_ascii=False)
        all_file_info_str += f"File {idx}:\n{file_info_str}\n\n"
    return all_file_info_str


def collect_rich_data_context(directory: str, max_files: int = 5, max_rows: int = 5) -> str:
    """收集更丰富的数据上下文：文件名、大小、类型、以及CSV/XLSX的列名与样例行。

    仅对已知结构（CSV/XLSX）提供轻量解析，避免加载巨大文件；其他类型仅给基本信息。
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return ""

    items = []
    files = sorted([f for f in dir_path.iterdir() if f.is_file()])[:max_files]
    for file_path in files:
        size_bytes = os.path.getsize(file_path)
        size_kb = size_bytes / 1024
        size_str = f"{size_kb:.1f}KB"
        entry = {
            "name": file_path.name,
            "size": size_str,
            "type": file_path.suffix.lower()
        }

        # CSV 预览：列名 + 前 N 行样例
        if file_path.suffix.lower() == ".csv":
            try:
                import csv
                with open(file_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    rows = []
                    header = next(reader, [])
                    for i, r in enumerate(reader):
                        if i >= max_rows:
                            break
                        rows.append(r)
                entry["columns"] = header
                entry["sample_rows"] = rows
            except Exception:
                entry["columns"] = []
                entry["sample_rows"] = []
        # XLSX 简预览：列名（若 pandas 可用）
        elif file_path.suffix.lower() == ".xlsx":
            try:
                import pandas as pd  # 可选
                df = pd.read_excel(file_path, nrows=max_rows)
                entry["columns"] = list(df.columns)
                entry["sample_rows"] = df.head(max_rows).values.tolist()
            except Exception:
                entry["columns"] = []
                entry["sample_rows"] = []

        items.append(entry)

    # 格式化为紧凑 JSON 字符串，便于模型理解
    try:
        ctx = json.dumps({"files": items}, ensure_ascii=False, indent=2)
    except Exception:
        ctx = str(items)
    return ctx


def collect_rich_data_context_for(
    workspace_dir: str,
    selected_rel_paths: Optional[List[str]] = None,
    max_files: int = 5,
    max_rows: int = 5,
) -> str:
    """基于 workspace 目录与可选的相对路径列表，收集 JSON 格式的数据上下文。

    - 若提供 selected_rel_paths：仅包含这些文件（存在且为文件）。
    - 否则：枚举 workspace 根目录下的前 max_files 个文件。
    - 对 CSV/XLSX 提供列名与样例行，其他类型仅提供基本信息。
    """
    root = Path(workspace_dir)
    if not root.exists():
        return ""

    items = []
    targets: List[Path] = []

    if selected_rel_paths:
        for rel in selected_rel_paths[:max_files]:
            try:
                p = (root / rel).resolve()
                if p.exists() and p.is_file() and root in p.parents:
                    targets.append(p)
            except Exception:
                continue
    else:
        targets = sorted([p for p in root.iterdir() if p.is_file()])[:max_files]

    for p in targets:
        try:
            rel_name = str(p.relative_to(root))
        except Exception:
            rel_name = p.name
        size_kb = (p.stat().st_size) / 1024.0
        entry = {"name": rel_name, "size": f"{size_kb:.1f}KB", "type": p.suffix.lower()}

        if p.suffix.lower() == ".csv":
            try:
                import csv
                with open(p, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                    rows = []
                    for i, r in enumerate(reader):
                        if i >= max_rows:
                            break
                        rows.append(r)
                entry["columns"] = header
                entry["sample_rows"] = rows
            except Exception:
                entry["columns"] = []
                entry["sample_rows"] = []
        elif p.suffix.lower() == ".xlsx":
            try:
                import pandas as pd
                df = pd.read_excel(p, nrows=max_rows)
                entry["columns"] = list(df.columns)
                entry["sample_rows"] = df.head(max_rows).values.tolist()
            except Exception:
                entry["columns"] = []
                entry["sample_rows"] = []

        items.append(entry)

    try:
        return json.dumps({"files": items}, ensure_ascii=False, indent=2)
    except Exception:
        return str(items)


def _find_first_dataset(workspace_dir: str) -> Optional[Path]:
    """Find the first CSV or XLSX file in the workspace (breadth-first)."""
    root = Path(workspace_dir)
    if not root.exists():
        return None
    candidates: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"}:
            candidates.append(p)
    if not candidates:
        return None
    # Prefer files not under hidden dirs
    candidates.sort(key=lambda x: ("/." in str(x), len(str(x))))
    return candidates[0]


def _simple_data_analysis(session_id: str) -> str:
    """Generate a deterministic analysis on the first dataset and return tagged content.

    This provides meaningful output when using the local Mock API, without relying on LLM.
    """
    workspace_dir = get_session_workspace(session_id)
    data_path = _find_first_dataset(workspace_dir)
    if not data_path:
        return "<Analyze>\n未在 workspace 中找到数据文件（CSV/XLSX）。请先上传数据再重试。\n</Analyze>\n<Answer>\nNo dataset found in workspace.\n</Answer>"

    rel_path = str(Path(data_path).relative_to(Path(workspace_dir)))
    ext = data_path.suffix.lower()

    # Prefer standard library for broad compatibility
    if ext == ".csv":
        # Build executable code using csv module
        code_lines = [
            "import csv",
            f"path = r\"{rel_path}\"",
            "with open(path, 'r', newline='', encoding='utf-8') as f:",
            "    reader = csv.reader(f)",
            "    rows = list(reader)",
            "header = rows[0] if rows else []",
            "data = rows[1:] if len(rows) > 1 else []",
            "print('Rows:', len(data))",
            "print('Cols:', len(header))",
            "print('Header:', header)",
            "print('Head(5):')",
            "for r in data[:5]:",
            "    print(r)",
        ]
        code_str = "\n".join(code_lines)

        # In-process summary for Analyze
        try:
            import csv
            with open(data_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            header = rows[0] if rows else []
            data = rows[1:] if len(rows) > 1 else []
            analyze_text = (
                f"数据文件: {rel_path}\n行数: {len(data)}\n列数: {len(header)}\n列名: {header}"
            )
        except Exception as e:
            analyze_text = f"CSV 解析错误: {e}"
    else:
        # XLSX or others: provide guidance and minimal code path using pandas if available
        code_lines = [
            "# Optional: requires pandas and openpyxl",
            "import pandas as pd",
            f"path = r\"{rel_path}\"",
            "df = pd.read_excel(path)",
            "print('Shape:', df.shape)",
            "print('Columns:', list(df.columns))",
            "print(df.head(5).to_string())",
        ]
        code_str = "\n".join(code_lines)
        analyze_text = (
            f"检测到非 CSV 文件: {rel_path}。如需更详细分析，请确保已安装 pandas/openpyxl。"
        )

    # Execute code in sandboxed workspace
    exe_output = execute_code_safe(code_str, workspace_dir)

    # Write a simple markdown report under generated/
    try:
        generated_dir = Path(workspace_dir) / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        report_path = uniquify_path(generated_dir / "data_overview.md")
        report_path.write_text(
            f"# Data Overview\n\nFile: {rel_path}\n\n## Summary\n\n{analyze_text}\n\n## Sample Head\n\n{exe_output}",
            encoding="utf-8",
        )
        report_rel = str(report_path.relative_to(Path(workspace_dir)))
        file_tag = f"<File>\n{report_rel}\n</File>\n"
    except Exception:
        file_tag = ""

    # 补充 Answer 段，确保前端步骤能够正确标记完成
    final = (
        f"<Analyze>\n{analyze_text}\n</Analyze>\n"
        f"<Code>\n```python\n{code_str}\n```\n</Code>\n"
        f"<Execute>\n{exe_output}\n</Execute>\n"
        f"{file_tag}"
        f"<Answer>\n分析已完成。你可以继续提出问题或导出报告。\n</Answer>\n"
    )
    return final


def get_file_icon(extension):
    """获取文件图标"""
    ext = extension.lower()
    icons = {
        (".jpg", ".jpeg", ".png", ".gif", ".bmp"): "🖼️",
        (".pdf",): "📕",
        (".doc", ".docx"): "📘",
        (".txt",): "📄",
        (".md",): "📝",
        (".csv", ".xlsx"): "📊",
        (".json", ".sqlite"): "🗄️",
        (".mp4", ".avi", ".mov"): "🎥",
        (".mp3", ".wav"): "🎵",
        (".zip", ".rar", ".tar"): "🗜️",
    }

    for extensions, icon in icons.items():
        if ext in extensions:
            return icon
    return "📁"


def uniquify_path(target: Path) -> Path:
    """若目标已存在，生成 'name (1).ext'、'name (2).ext' 形式的新路径。"""
    if not target.exists():
        return target
    parent = target.parent
    stem = target.stem
    suffix = target.suffix
    import re as _re

    m = _re.match(r"^(.*) \((\d+)\)$", stem)
    base = stem
    start = 1
    if m:
        base = m.group(1)
        try:
            start = int(m.group(2)) + 1
        except Exception:
            start = 1
    i = start
    while True:
        candidate = parent / f"{base} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def execute_code(code_str):
    """执行Python代码"""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
            stderr_capture
        ):
            exec(code_str, {})
        output = stdout_capture.getvalue()
        if stderr_capture.getvalue():
            output += stderr_capture.getvalue()
        return output
    except Exception as exec_error:
        return f"[Error]: {str(exec_error)}"


# API Routes
@app.get("/workspace/files")
async def get_workspace_files(session_id: str = Query("default")):
    """获取工作区文件列表（支持 session 隔离）"""
    workspace_dir = get_session_workspace(session_id)
    generated_dir = Path(workspace_dir) / "generated"
    # 获取 generated 目录下的文件名集合
    generated_files = (
        set(f.name for f in generated_dir.iterdir() if f.is_file())
        if generated_dir.exists()
        else set()
    )

    files = []
    for file_path in Path(workspace_dir).iterdir():
        if file_path.is_file():
            if file_path.name in generated_files:
                continue
            stat = file_path.stat()
            rel_path = f"{session_id}/{file_path.name}"
            files.append(
                {
                    "name": file_path.name,
                    "size": stat.st_size,
                    "extension": file_path.suffix.lower(),
                    "icon": get_file_icon(file_path.suffix),
                    "download_url": build_download_url(rel_path),
                    "preview_url": (
                        build_download_url(rel_path)
                        if file_path.suffix.lower()
                        in [
                            ".jpg",
                            ".jpeg",
                            ".png",
                            ".gif",
                            ".bmp",
                            ".pdf",
                            ".txt",
                            ".doc",
                            ".docx",
                            ".csv",
                            ".xlsx",
                        ]
                        else None
                    ),
                }
            )
    return {"files": files}


# ---------- Workspace Tree & Single File Delete ----------
def _rel_path(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        return rel.as_posix()
    except Exception:
        return path.name


def build_tree(path: Path, root: Path | None = None) -> dict:
    if root is None:
        root = path
    node: dict = {
        "name": path.name or "workspace",
        "path": _rel_path(path, root),
        "is_dir": path.is_dir(),
    }
    if path.is_dir():
        children = []

        # 自定义排序：generated 文件夹放在最后，其他按目录优先、名称排序
        def sort_key(p):
            is_generated = p.name == "generated"
            is_dir = p.is_dir()
            return (is_generated, not is_dir, p.name.lower())

        for child in sorted(path.iterdir(), key=sort_key):
            if child.name.startswith("."):
                continue
            children.append(build_tree(child, root))
        node["children"] = children
    else:
        node["size"] = path.stat().st_size
        node["extension"] = path.suffix.lower()
        node["icon"] = get_file_icon(path.suffix)
        rel = _rel_path(path, root)
        node["download_url"] = build_download_url(rel)
    return node


@app.get("/workspace/tree")
async def workspace_tree(session_id: str = Query("default")):
    workspace_dir = get_session_workspace(session_id)
    root = Path(workspace_dir)
    tree_data = build_tree(root, root)

    # 在下载链接前加上 session_id 前缀
    def prefix_urls(node, sid):
        if "download_url" in node and node["download_url"]:
            # 重新构建包含 session_id 的路径
            rel = node.get("path", "")
            node["download_url"] = build_download_url(f"{sid}/{rel}")
        if "children" in node:
            for child in node["children"]:
                prefix_urls(child, sid)

    prefix_urls(tree_data, session_id)
    return tree_data


@app.delete("/workspace/file")
async def delete_workspace_file(
    path: str = Query(..., description="relative path under workspace"),
    session_id: str = Query("default"),
):
    workspace_dir = get_session_workspace(session_id)
    abs_workspace = Path(workspace_dir).resolve()
    target = (abs_workspace / path).resolve()
    if abs_workspace not in target.parents and target != abs_workspace:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    if target.is_dir():
        raise HTTPException(status_code=400, detail="Folder deletion not allowed")
    try:
        target.unlink()
        return {"message": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workspace/move")
async def move_path(
    src: str = Query(..., description="relative source path under workspace"),
    dst_dir: str = Query("", description="relative target directory under workspace"),
    session_id: str = Query("default"),
):
    """在同一 workspace 内移动（或重命名）文件/目录。
    - src: 源相对路径（必填）
    - dst_dir: 目标目录（相对路径，空表示移动到根目录）
    """
    workspace_dir = get_session_workspace(session_id)
    abs_workspace = Path(workspace_dir).resolve()

    abs_src = (abs_workspace / src).resolve()
    if abs_workspace not in abs_src.parents and abs_src != abs_workspace:
        raise HTTPException(status_code=400, detail="Invalid src path")
    if not abs_src.exists():
        raise HTTPException(status_code=404, detail="Source not found")

    abs_dst_dir = (abs_workspace / (dst_dir or "")).resolve()
    if abs_workspace not in abs_dst_dir.parents and abs_dst_dir != abs_workspace:
        raise HTTPException(status_code=400, detail="Invalid dst_dir path")
    abs_dst_dir.mkdir(parents=True, exist_ok=True)

    target = abs_dst_dir / abs_src.name
    target = uniquify_path(target)
    try:
        shutil.move(str(abs_src), str(target))
        rel_new = str(target.relative_to(abs_workspace))
        return {"message": "moved", "new_path": rel_new}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Move failed: {e}")


@app.delete("/workspace/dir")
async def delete_workspace_dir(
    path: str = Query(..., description="relative directory under workspace"),
    recursive: bool = Query(True, description="delete directory recursively"),
    session_id: str = Query("default"),
):
    """删除 workspace 下的目录。默认递归删除，禁止删除根目录。"""
    workspace_dir = get_session_workspace(session_id)
    abs_workspace = Path(workspace_dir).resolve()
    target = (abs_workspace / path).resolve()
    if abs_workspace not in target.parents and target != abs_workspace:
        raise HTTPException(status_code=400, detail="Invalid path")
    if target == abs_workspace:
        raise HTTPException(status_code=400, detail="Cannot delete workspace root")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Not a directory")
    try:
        if recursive:
            shutil.rmtree(target)
        else:
            target.rmdir()
        return {"message": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/proxy")
async def proxy(url: str):
    """Simple CORS proxy for previewing external files.
    WARNING: For production, add domain allowlist and authentication.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            r = await client.get(url)
        return Response(
            content=r.content,
            media_type=r.headers.get("content-type", "application/octet-stream"),
            headers={"Access-Control-Allow-Origin": "*"},
            status_code=r.status_code,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Proxy fetch failed: {e}")


@app.post("/workspace/upload")
async def upload_files(
    files: List[UploadFile] = File(...), session_id: str = Query("default")
):
    """上传文件到工作区（支持 session 隔离）"""
    workspace_dir = get_session_workspace(session_id)
    uploaded_files = []

    for file in files:
        # 唯一化文件名，避免覆盖
        dst = uniquify_path(Path(workspace_dir) / file.filename)
        with open(dst, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        uploaded_files.append(
            {
                "name": dst.name,
                "size": len(content),
                "path": str(dst.relative_to(Path(workspace_dir))),
            }
        )

    return {
        "message": f"Successfully uploaded {len(uploaded_files)} files",
        "files": uploaded_files,
    }


@app.delete("/workspace/clear")
async def clear_workspace(session_id: str = Query("default")):
    """清空工作区（支持 session 隔离）"""
    workspace_dir = get_session_workspace(session_id)
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)
    return {"message": "Workspace cleared successfully"}


@app.post("/workspace/upload-to")
async def upload_to_dir(
    dir: str = Query("", description="relative directory under workspace"),
    files: List[UploadFile] = File(...),
    session_id: str = Query("default"),
):
    """上传文件到 workspace 下的指定子目录（仅限工作区内）。"""
    workspace_dir = get_session_workspace(session_id)
    abs_workspace = Path(workspace_dir).resolve()
    target_dir = (abs_workspace / dir).resolve()
    if abs_workspace not in target_dir.parents and target_dir != abs_workspace:
        raise HTTPException(status_code=400, detail="Invalid dir path")
    target_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        dst = uniquify_path(target_dir / f.filename)
        try:
            with open(dst, "wb") as buffer:
                content = await f.read()
                buffer.write(content)
            saved.append(
                {
                    "name": dst.name,
                    "size": len(content),
                    "path": str(dst.relative_to(abs_workspace)),
                }
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Save failed: {e}")
    return {"message": f"uploaded {len(saved)}", "files": saved}


@app.post("/execute")
async def execute_code_api(request: dict):
    """执行 Python 代码"""
    print("🔥 Execute API called:", request)  # Debug log

    try:
        code = request.get("code", "")
        session_id = request.get("session_id", "default")
        workspace_dir = get_session_workspace(session_id)

        if not code:
            raise HTTPException(status_code=400, detail="No code provided")

        print(f"Executing code: {code[:100]}...")  # Debug log (first 100 chars)

        # 使用子进程安全执行，避免 GUI/线程问题（在指定 session workspace 中）
        result = execute_code_safe(code, workspace_dir)
        print(f"✅ Execution result: {result[:200]}...")  # Debug log

        return {
            "success": True,
            "result": result,
            "message": "Code executed successfully",
        }

    except Exception as e:
        print(f"❌ Execution error: {traceback.format_exc()}")  # Debug log
        return {
            "success": False,
            "result": f"Error: {str(e)}",
            "message": "Code execution failed",
        }


def fix_code_block(content):
    def fix_text(text):
        stack = []
        lines = text.splitlines(keepends=True)
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```python"):
                if stack and stack[-1] == "```python":
                    result.append("```\n")
                    stack.pop()
                stack.append("```python")
                result.append(line)
            elif stripped == "```":
                if stack and stack[-1] == "```python":
                    stack.pop()
                result.append(line)
            else:
                result.append(line)
        while stack:
            result.append("```\n")
            stack.pop()
        return "".join(result)

    if isinstance(content, str):
        return fix_text(content)
    elif isinstance(content, tuple):
        text_part = content[0] if content[0] else ""
        return (fix_text(text_part), content[1])
    return content


def fix_tags_and_codeblock(s: str) -> str:
    """
    修复未闭合的tags，并确保</Code>后代码块闭合。
    """
    pattern = re.compile(
        r"<(Analyze|Understand|Code|Execute|Answer)>(.*?)(?:</\1>|(?=$))", re.DOTALL
    )

    # 找所有匹配
    matches = list(pattern.finditer(s))
    if not matches:
        return s  # 没有标签，直接返回

    # 检查最后一个匹配是否闭合
    last_match = matches[-1]
    tag_name = last_match.group(1)
    matched_text = last_match.group(0)

    if not matched_text.endswith(f"</{tag_name}>"):
        # 没有闭合时谨慎补齐，仅当存在起始标签才补
        if tag_name == "Code":
            # 若文本中存在 <Code> 但缺少 </Code>，且代码块未闭合，则补齐
            if "<Code>" in s and "</Code>" not in s:
                s = fix_code_block(s) + f"\n</{tag_name}>"
        else:
            s += f"\n</{tag_name}>"

    return s


def bot_stream(messages, workspace, session_id="default", language: str = "zh"):
    original_cwd = os.getcwd()
    WORKSPACE_DIR = get_session_workspace(session_id)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    # 创建 generated 子文件夹用于存放代码生成的文件
    GENERATED_DIR = os.path.join(WORKSPACE_DIR, "generated")
    os.makedirs(GENERATED_DIR, exist_ok=True)
    # print(messages)
    if messages and messages[0]["role"] == "assistant":
        messages = messages[1:]
    if messages and messages[-1]["role"] == "user":
        user_message = messages[-1]["content"]
        # workspace 可能是相对路径列表；我们将其用于构建精细的数据上下文
        try:
            selected_paths = [str(p) for p in workspace] if isinstance(workspace, list) else []
        except Exception:
            selected_paths = []

        file_info_rich = collect_rich_data_context_for(WORKSPACE_DIR, selected_paths)
        # 若未选择文件或解析失败，退回到基础信息
        if not file_info_rich:
            file_info_rich = collect_rich_data_context(WORKSPACE_DIR)

        if file_info_rich:
            messages[-1]["content"] = (
                f"# UserQuestion\n{user_message}\n\n"
                f"# DataContext\n{file_info_rich}"
            )
        else:
            messages[-1]["content"] = f"# UserQuestion\n{user_message}"

    # 在最前注入系统提示，指导大模型生成可执行、可视化且鲁棒的 Python 代码
    # 根据调用方传入的 language 切换系统提示词
    lang = (language or "zh").lower()

    if lang.startswith("zh"):
        system_prompt = (
            "你是一位资深的数据分析助手。请同时考虑用户询问 (# UserQuestion) 与数据上下文 (# DataContext)。"
            "务必严格按以下格式输出：<Analyze>…</Analyze><Code>```python\n…\n```</Code>。"
            "要求：1) 代码仅使用标准库或 pandas/numpy/matplotlib（可选），"
            "2) 禁止使用除上述之外的第三方库（如 scikit-learn、seaborn、xgboost、tensorflow、torch 等），"
            "3) 自动处理缺失值与异常；4) 给出统计摘要与可视化建议；"
            "5) 不要使用 notebook 魔法命令（例如 %pwd）；6) 读取文件路径必须来自 # DataContext 的 files 列表（相对路径）。"
        )
    else:
        system_prompt = (
            "You are a senior data analysis assistant. Consider both # UserQuestion and # DataContext."
            " Strictly output in the format: <Analyze>…</Analyze><Code>```python\n…\n```</Code>."
            " Requirements: 1) Use only standard library or pandas/numpy/matplotlib (optional),"
            " 2) Do NOT use third-party libraries beyond these (e.g., scikit-learn, seaborn, xgboost, tensorflow, torch),"
            " 3) Handle missing/outlier values; 4) Provide statistical summaries and visualization suggestions;"
            " 5) Avoid notebook magic commands (e.g., %pwd); 6) File paths must come from files listed in # DataContext (relative paths)."
        )
    messages = ([{"role": "system", "content": system_prompt}] + messages)
    # print("111",messages)
    initial_workspace = set(workspace)
    assistant_reply = ""
    finished = False
    exe_output = None
    while not finished:
        # Decide streaming capability by API base (mock server doesn't stream)
        use_stream = "localhost:8000" not in API_BASE
        if use_stream:
            # Streaming path
            response = client.chat.completions.create(
                model=MODEL_PATH,
                messages=messages,
                temperature=0.4,
                stream=True,
                extra_body={
                    "add_generation_prompt": False,
                    "stop_token_ids": [151676, 151645],
                    "max_new_tokens": 32768,
                },
            )
            cur_res = ""
            last_finish_reason = None
            for rchunk in response:
                if rchunk.choices:
                    if getattr(rchunk.choices[0], "delta", None):
                        delta = rchunk.choices[0].delta.content
                        if delta is not None:
                            cur_res += delta
                            assistant_reply += delta
                            yield assistant_reply
                    if rchunk.choices[0].finish_reason:
                        last_finish_reason = rchunk.choices[0].finish_reason
            # 不再盲目补齐 </Code>，仅在存在 <Code> 且缺少闭合时补齐
            if (
                last_finish_reason == "stop"
                and ("<Code>" in cur_res)
                and ("</Code>" not in cur_res)
            ):
                cur_res += "</Code>"
                assistant_reply += "</Code>"
                yield assistant_reply
            finished = True
        else:
            # Non-stream path (mock server)
            response = client.chat.completions.create(
                model=MODEL_PATH,
                messages=messages,
                temperature=0.4,
                stream=False,
                extra_body={
                    "add_generation_prompt": False,
                    "stop_token_ids": [151676, 151645],
                    "max_new_tokens": 32768,
                },
            )
            try:
                msg = response.choices[0].message
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            except Exception:
                content = ""
            cur_res = content
            assistant_reply += content
            yield assistant_reply
            finished = True
            if "</Answer>" in assistant_reply:
                finished = True
                break
        # 执行代码段（即使流式已结束也需要执行）
        if "</Code>" in cur_res:
            messages.append({"role": "assistant", "content": cur_res})
            code_match = re.search(r"<Code>(.*?)</Code>", cur_res, re.DOTALL)
            if code_match:
                code_content = code_match.group(1).strip()
                md_match = re.search(r"```(?:python)?(.*?)```", code_content, re.DOTALL)
                code_str = md_match.group(1).strip() if md_match else code_content
                # 执行前快照（路径 -> (size, mtime)）
                try:
                    before_state = {
                        p.resolve(): (p.stat().st_size, p.stat().st_mtime_ns)
                        for p in Path(WORKSPACE_DIR).rglob("*")
                        if p.is_file()
                    }
                except Exception:
                    before_state = {}
                # 在子进程中以固定工作区执行（流式）
                stream_started = False
                exe_collected = []
                for chunk in execute_code_stream(code_str, WORKSPACE_DIR):
                    if not stream_started:
                        # 首次块：开始 Execute 段（使用三反引号供前端高亮）
                        assistant_reply += "\n<Execute>\n```\n"
                        stream_started = True
                    # 追加输出
                    assistant_reply += chunk
                    exe_collected.append(chunk)
                    yield assistant_reply
                # 结束 Execute 段
                if stream_started:
                    assistant_reply += "\n```\n</Execute>\n"
                exe_output = "".join(exe_collected)

                # 执行后快照
                try:
                    after_state = {
                        p.resolve(): (p.stat().st_size, p.stat().st_mtime_ns)
                        for p in Path(WORKSPACE_DIR).rglob("*")
                        if p.is_file()
                    }
                except Exception:
                    after_state = {}
                # 计算新增与修改
                added_paths = [p for p in after_state.keys() if p not in before_state]
                modified_paths = [
                    p
                    for p in after_state.keys()
                    if p in before_state and after_state[p] != before_state[p]
                ]

                # 将新增和修改的文件移动到 generated 文件夹
                artifact_paths = []
                for p in added_paths:
                    try:
                        # 如果文件不在 generated 文件夹中，移动它
                        if not str(p).startswith(GENERATED_DIR):
                            dest_path = Path(GENERATED_DIR) / p.name
                            dest_path = uniquify_path(dest_path)
                            shutil.copy2(str(p), str(dest_path))
                            artifact_paths.append(dest_path.resolve())
                        else:
                            artifact_paths.append(p)
                    except Exception as e:
                        print(f"Error moving file {p}: {e}")
                        artifact_paths.append(p)

                # 为修改的文件生成副本并移动到 generated 文件夹
                for p in modified_paths:
                    try:
                        dest_name = f"{Path(p).stem}_modified{Path(p).suffix}"
                        dest_path = Path(GENERATED_DIR) / dest_name
                        dest_path = uniquify_path(dest_path)
                        shutil.copy2(p, dest_path)
                        artifact_paths.append(dest_path.resolve())
                    except Exception as e:
                        print(f"Error copying modified file {p}: {e}")

                # 旧：Execute 内部放控制台输出；新：追加 <File> 段落给前端渲染卡片
                exe_str = ""  # 已通过流式输出到 assistant_reply，这里不重复输出
                file_block = ""
                if artifact_paths:
                    lines = ["<File>"]
                    for p in artifact_paths:
                        try:
                            rel = (
                                Path(p)
                                .relative_to(Path(WORKSPACE_DIR).resolve())
                                .as_posix()
                            )
                        except Exception:
                            rel = Path(p).name
                        # 在相对路径前加上 session_id 前缀
                        url = build_download_url(f"{session_id}/{rel}")
                        name = Path(p).name
                        lines.append(f"- [{name}]({url})")
                        if Path(p).suffix.lower() in [
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".gif",
                            ".webp",
                            ".svg",
                        ]:
                            lines.append(f"![{name}]({url})")
                    lines.append("</File>")
                    file_block = "\n" + "\n".join(lines) + "\n"
                assistant_reply += exe_str + file_block
                # 若模型未提供 <Answer>，在执行完成后补充一个简短的结论，便于前端流程结束
                if "</Answer>" not in assistant_reply:
                    assistant_reply += (
                        "\n<Answer>\n执行已完成，结果与生成文件已展示。若需继续分析或导出报告，请告知。\n</Answer>\n"
                    )
                yield assistant_reply
                messages.append({"role": "execute", "content": f"{exe_output}"})
                # 刷新工作区快照（路径集合）
                current_files = set(
                    [
                        os.path.join(WORKSPACE_DIR, f)
                        for f in os.listdir(WORKSPACE_DIR)
                        if os.path.isfile(os.path.join(WORKSPACE_DIR, f))
                    ]
                )
                new_files = list(current_files - initial_workspace)
                if new_files:
                    workspace.extend(new_files)
                    initial_workspace.update(new_files)
    os.chdir(original_cwd)


@app.post("/chat/completions")
async def chat(body: dict = Body(...)):
    messages = body.get("messages", [])
    workspace = body.get("workspace", [])
    session_id = body.get("session_id", "default")
    stream_flag = bool(body.get("stream", False))
    # Detect language from request body, default to Chinese
    lang = str(body.get("language") or "zh").lower()

    # When stream=True → return text/plain streaming of OpenAI-style JSON objects
    if stream_flag:
        def generate():
            for reply in bot_stream(messages, workspace, session_id, language=lang):
                print(reply)
                result = {
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": "deepanalyze-8b",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": fix_tags_and_codeblock(reply),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield json.dumps(result)

        return StreamingResponse(generate(), media_type="text/plain")

    # When stream=false → return a single JSON object with full content
    # Special-case: if using local mock API, fetch directly to ensure content
    final_reply = ""
    if "localhost:8000" in API_BASE:
        try:
            # Build language-specific system prompt for mock server path
            if lang.startswith("zh"):
                system_prompt = (
                    "你是一位资深的数据分析助手。请同时考虑用户询问 (# UserQuestion) 与数据上下文 (# DataContext)。"
                    "务必严格按以下格式输出：<Analyze>…</Analyze><Code>```python\n…\n```</Code>。"
                    "要求：1) 代码仅使用标准库或 pandas/numpy/matplotlib（可选），"
                    "2) 禁止使用除上述之外的第三方库（如 scikit-learn、seaborn、xgboost、tensorflow、torch 等），"
                    "3) 自动处理缺失值与异常；4) 给出统计摘要与可视化建议；"
                    "5) 不要使用 notebook 魔法命令（例如 %pwd）；6) 读取文件路径必须来自 # DataContext 的 files 列表（相对路径）。"
                )
            else:
                system_prompt = (
                    "You are a senior data analysis assistant. Consider both # UserQuestion and # DataContext."
                    " Strictly output in the format: <Analyze>…</Analyze><Code>```python\n…\n```</Code>."
                    " Requirements: 1) Use only standard library or pandas/numpy/matplotlib (optional),"
                    " 2) Do NOT use third-party libraries beyond these (e.g., scikit-learn, seaborn, xgboost, tensorflow, torch),"
                    " 3) Handle missing/outlier values; 4) Provide statistical summaries and visualization suggestions;"
                    " 5) Avoid notebook magic commands (e.g., %pwd); 6) File paths must come from files listed in # DataContext (relative paths)."
                )
            messages_with_system = ([{"role": "system", "content": system_prompt}] + messages)
            resp = client.chat.completions.create(
                model=MODEL_PATH,
                messages=messages_with_system,
                temperature=0.4,
                stream=False,
                extra_body={"add_generation_prompt": False},
            )
            msg = resp.choices[0].message
            final_reply = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        except Exception:
            final_reply = ""
        # Fallback: if still empty, use internal generator to synthesize reply
        if not final_reply or not str(final_reply).strip():
            # Prefer deterministic data analysis over LLM when using mock server
            try:
                final_reply = _simple_data_analysis(session_id)
            except Exception:
                # as ultimate fallback, attempt internal generator
                try:
                    for reply in bot_stream(messages, workspace, session_id, language=lang):
                        final_reply = reply
                except Exception:
                    final_reply = ""
        else:
            # If we have dataset and mock reply lacks Execute/File, upgrade to real analysis
            try:
                workspace_dir = get_session_workspace(session_id)
                if _find_first_dataset(workspace_dir) and (
                    ("</Execute>" not in final_reply) or ("</File>" not in final_reply)
                ):
                    final_reply = _simple_data_analysis(session_id)
            except Exception:
                pass
    else:
        for reply in bot_stream(messages, workspace, session_id, language=lang):
            final_reply = reply

    result = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "deepanalyze-8b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": fix_tags_and_codeblock(final_reply),
                },
                "finish_reason": "stop",
            }
        ],
    }
    return JSONResponse(result)


# -------- Export Report (PDF + MD) --------
from datetime import datetime


def _extract_sections_from_messages(messages: list[dict]) -> str:
    """从历史消息中抽取 <Answer>..</Answer> 作为报告主体，其余部分按原始顺序作为 Appendix 拼成 Markdown。"""
    if not isinstance(messages, list):
        return ""
    import re as _re

    parts: list[str] = []
    appendix: list[str] = []

    tag_pattern = r"<(Analyze|Understand|Code|Execute|File|Answer)>([\s\S]*?)</\1>"

    for idx, m in enumerate(messages, start=1):
        role = (m or {}).get("role")
        if role != "assistant":
            continue
        content = str((m or {}).get("content") or "")

        step = 1
        # 按照在文本中的出现顺序依次提取
        for match in _re.finditer(tag_pattern, content, _re.DOTALL):
            tag, seg = match.groups()
            seg = seg.strip()
            if tag == "Answer":
                parts.append(f"{seg}\n")

            appendix.append(f"\n### Step {step}: {tag}\n\n{seg}\n")
            step += 1

    final_text = "".join(parts).strip()
    if appendix:
        final_text += (
            "\n\n\\newpage\n\n# Appendix: Detailed Process\n"
            + "".join(appendix).strip()
        )

    # print(final_text)
    return final_text


def _save_md(md_text: str, base_name: str, workspace_dir: str) -> Path:
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    md_path = uniquify_path(Path(workspace_dir) / f"{base_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    return md_path


import pypandoc


def _save_pdf(md_text: str, base_name: str, workspace_dir: str) -> Path:
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    pdf_path = uniquify_path(Path(workspace_dir) / f"{base_name}.pdf")

    output = pypandoc.convert_text(
        md_text,
        "pdf",
        format="md",
        outputfile=str(pdf_path),
        extra_args=[
            "--standalone",
            "--pdf-engine=xelatex",
        ],
    )
    return pdf_path


from typing import Optional


def _render_md_to_html(md_text: str, title: Optional[str] = None) -> str:
    """简化为占位实现（仅供未来 PDF 渲染使用）。当前仅生成 MD。"""
    doc_title = (title or "Report").strip() or "Report"
    safe = (md_text or "").replace("<", "&lt;").replace(">", "&gt;")
    return f"<html><head><meta charset='utf-8'><title>{doc_title}</title></head><body><pre>{safe}</pre></body></html>"


def _save_pdf_from_md(html_text: str, base_name: str) -> Path:
    """TODO: 服务端 PDF 渲染未实现。"""
    raise NotImplementedError("TODO: implement server-side PDF rendering")


def _save_pdf_with_chromium(html_text: str, base_name: str) -> Path:
    """TODO: 使用 Chromium 渲染 PDF（暂不实现）。"""
    raise NotImplementedError("TODO: chromium-based PDF rendering")


def _save_pdf_from_text(text: str, base_name: str) -> Path:
    """TODO: 纯文本 PDF 渲染（暂不实现）。"""
    raise NotImplementedError("TODO: text-based PDF rendering")


@app.post("/export/report")
async def export_report(body: dict = Body(...)):
    """
    接收全部聊天历史（messages: [{role, content}...]），抽取 <Analyze>..</Analyze> ~ <Answer>..</Answer>
    仅生成 Markdown 文件并保存到 workspace；PDF 渲染留作 TODO。
    """
    try:
        messages = body.get("messages", [])
        title = (body.get("title") or "").strip()
        session_id = body.get("session_id", "default")
        workspace_dir = get_session_workspace(session_id)

        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="messages must be a list")

        md_text = _extract_sections_from_messages(messages)
        if not md_text:
            md_text = (
                "(No <Analyze>/<Understand>/<Code>/<Execute>/<Answer> sections found.)"
            )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r"[^\w\-_.]+", "_", title) if title else "Report"
        base_name = f"{safe_title}_{ts}" if title else f"Report_{ts}"

        # Save MD into generated/ folder under workspace
        export_dir = os.path.join(workspace_dir, "generated")
        os.makedirs(export_dir, exist_ok=True)

        print(md_text)
        md_path = _save_md(md_text, base_name, export_dir)

        # PDF 暂不生成（TODO）。
        pdf_path = _save_pdf(md_text, base_name, export_dir)

        result = {
            "message": "exported",
            "md": md_path.name,
            "pdf": pdf_path.name if pdf_path else None,
            "download_urls": {
                "md": build_download_url(f"{session_id}/generated/{md_path.name}"),
                "pdf": (
                    build_download_url(f"{session_id}/generated/{pdf_path.name}")
                    if pdf_path
                    else None
                ),
            },
        }
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🚀 启动后端服务...")
    print(f"   - API服务: http://localhost:8200")
    print(f"   - 文件服务: http://localhost:8100")
    uvicorn.run(app, host="0.0.0.0", port=8200)
