"""A small LangGraph ReAct agent harness backed by a local Ollama model.

Run interactively:
    python main.py

Run a single prompt:
    python main.py "What files are in this directory?"

Environment variables:
    HARNESS_MODEL   Ollama model name (default: qwen3.5:4b)
    HARNESS_TEMP    Sampling temperature (default: 0.2)
    HARNESS_THREAD  Conversation thread id for memory (default: default)
    HARNESS_THINK   Show model thinking/reasoning (default: true)
"""

from __future__ import annotations
import json
import string
import random

import os
import subprocess
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

name = input("username: ")

# --- Chat Storage ---
CHATS_DIR = Path(".chats")
CHATS_DIR.mkdir(exist_ok=True)

def save_chat(thread_id: str, messages: list) -> str:
    """Saves the current message history to a JSON file."""
    file_path = CHATS_DIR / f".{thread_id}.json"
    # Convert LangChain messages to serializable dicts
    serializable = []
    for msg in messages:
        serializable.append({
            "type": msg.type,
            "content": msg.content,
            "tool_calls": getattr(msg, "tool_calls", None)
        })
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return str(file_path)

def load_chat(thread_id: str):
    """Loads messages from a JSON file. This is complex with MemorySaver, 
    but for simple persistence we can recreate the state or use the thread_id."""
    file_path = CHATS_DIR / f"{thread_id}.json"
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_chats() -> list:
    """Lists all saved chat files."""
    return [f.stem for f in CHATS_DIR.glob("*.json")]
        
# --- Config ---
MODEL = os.getenv("HARNESS_MODEL", "llama3.2")
TEMPERATURE = float(os.getenv("HARNESS_TEMP", "0.2"))
THREAD_ID = os.getenv("HARNESS_THREAD", "default")

WORKSPACE = Path.cwd().resolve()

SYSTEM_PROMPT = """You are a helpful, careful agent running inside a local harness.
Use the available tools when they help answer the user.
Prefer small, verifiable steps. When you change files, explain what you changed.
If a tool fails, diagnose the failure and try a safer alternative.
Keep answers concise and practical."""


def _safe_path(path: str) -> Path:
    """Resolve a user-supplied path and keep it inside the workspace."""
    candidate = (WORKSPACE / path).resolve()
    if candidate != WORKSPACE and WORKSPACE not in candidate.parents:
        raise ValueError(f"Path escapes workspace: {path}")
    return candidate


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file from the workspace. `path` is relative to the workspace."""
    p = _safe_path(path)
    if not p.is_file():
        return f"ERROR: not a file: {path}"
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"ERROR: {path} is not valid UTF-8 text"


@tool
def write_file(path: str, content: str) -> str:
    """Write UTF-8 text to a file in the workspace, creating parent directories as needed."""
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}"


@tool
def list_directory(path: str = ".") -> str:
    """List files and directories in a workspace directory."""
    p = _safe_path(path)
    if not p.is_dir():
        return f"ERROR: not a directory: {path}"
    entries = []
    for child in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
        marker = "/" if child.is_dir() else ""
        entries.append(f"{child.name}{marker}")
    return "\n".join(entries) if entries else "(empty)"


@tool
def make_directory(path: str) -> str:
    """Create a directory (and parents) in the workspace."""
    p = _safe_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return f"Created directory: {path}"


@tool
def python_run(code: str, timeout: int = 30) -> str:
    """Run Python code in a subprocess and return stdout/stderr. Use for calculations or experiments."""
    if timeout < 1 or timeout > 120:
        timeout = 30
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(WORKSPACE),
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {timeout} seconds"
    output = (result.stdout or "") + (("\nSTDERR:\n" + result.stderr) if result.stderr else "")
    if result.returncode != 0:
        output = f"EXIT CODE {result.returncode}\n{output}"
    return output.strip() or "(no output)"


TOOLS = [read_file, write_file, list_directory, make_directory, python_run]


def build_agent():
    llm = ChatOllama(model=MODEL, temperature=TEMPERATURE, num_ctx=4096)
    checkpointer = MemorySaver()
    return create_agent(
        llm,
        TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def run_once(agent, user_input: str) -> None:
    config = {"configurable": {"thread_id": THREAD_ID}}
    
    total_printed_len = 0
    started_printing = False
    
    for msg, metadata in agent.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\n[tool] {tc['name']}({tc.get('args', {})})", flush=True)
        
        elif isinstance(msg, ToolMessage):
            preview = str(msg.content)
            if len(preview) > 500:
                preview = preview[:500] + "..."
            print(f"[tool result] {preview}\n", flush=True)
        
        elif isinstance(msg, AIMessage) and msg.content:
            if not started_printing:
                print("\nassistant> ", end="", flush=True)
                started_printing = True
            
            content = msg.content
            # If current content is shorter than total printed, it MUST be a delta
            if len(content) < total_printed_len:
                print(content, end="", flush=True)
                total_printed_len += len(content)
            # If it's longer, it might be accumulated. 
            # But if it's only slightly longer (e.g. by 1-5 chars), it's still likely a delta.
            # The safest way is to check if it starts with the previous total_printed_len content.
            # However, we don't have the previous full string.
            # Most Ollama streams in 'messages' mode are consistent.
            # Let's assume if it's >= total_printed_len, we check for accumulation.
            elif len(content) >= total_printed_len:
                # This is the critical part. If we just print the delta, 
                # and it was actually a delta, we lose content.
                # If we print the whole thing, we repeat.
                # Let's use a simple heuristic: if we are at the very start, just print.
                if total_printed_len == 0:
                    print(content, end="", flush=True)
                    total_printed_len = len(content)
                else:
                    # Try to detect if it's a delta or accumulated by checking common patterns.
                    # In most LangGraph/Ollama setups, it's one or the other.
                    # Let's try to detect by seeing if the 'delta' would make sense.
                    # If the 'content' is small (e.g. < 20 chars), it's almost always a delta.
                    if len(content) < 20:
                        print(content, end="", flush=True)
                        total_printed_len += len(content)
                    else:
                        # Treat as accumulated
                        delta = content[total_printed_len:]
                        if delta:
                            print(delta, end="", flush=True)
                        total_printed_len = len(content)

    print() 
    
    new_state = agent.get_state(config)
    save_chat(THREAD_ID, new_state.values.get("messages", []))

def interactive(agent) -> None:
    global THREAD_ID
    print(f"Harness ready | model={MODEL} | temp={TEMPERATURE} | thread={THREAD_ID}")
    print("Commands: /exit, /reset, /tools, /load <thread_id>, /list")
    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            print("Goodbye!")
            break
        if user_input == "/reset":
            THREAD_ID = f"{THREAD_ID}-reset"
            print(f"Thread reset. new thread={THREAD_ID}")
            continue
        if user_input == "/tools":
            print("Tools:", ", ".join(t.name for t in TOOLS))
            continue
        if user_input == "/list":
            chats = list_chats()
            print("Saved chats:", ", ".join(chats) if chats else "None")
            continue
        if user_input.startswith("/load "):
            try:
                new_id = user_input.split(maxsplit=1)[1].strip()
                if not list_chats() or new_id not in list_chats():
                    print(f"Chat {new_id} not found.")
                else:
                    THREAD_ID = new_id
                    print(f"Loaded thread: {THREAD_ID}")
            except IndexError:
                print("Usage: /load <thread_id>")
            continue
        try:
            run_once(agent, user_input)
        except Exception as exc:  # Keep the REPL alive on transient model/tool errors.
            print(f"\nERROR: {exc}")


def main() -> int:
    agent = build_agent()
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        try:
            run_once(agent, prompt)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0
    interactive(agent)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
