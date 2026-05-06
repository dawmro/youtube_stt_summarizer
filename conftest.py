"""
conftest.py — stubs all heavy dependencies before app/api modules are imported.
Run tests with: pytest -v
Every stub is installed into sys.modules inside pytest_configure, which runs
before any test module is collected. This guarantees zero GPU, Ollama, or
network requirements during unit tests.
"""
import sys
import types
from unittest.mock import MagicMock

def _stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

def pytest_configure(config):
    # Register custom markers to suppress warnings
    config.addinivalue_line(
        "markers", "integration: marks tests requiring real Ollama/Whisper models"
    )

    # ---- gradio ----------------------------------------------------------------
    gr = _stub("gradio")
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=MagicMock())
    ctx.__exit__ = MagicMock(return_value=False)
    gr.Blocks = MagicMock(return_value=ctx)
    for attr in ["State", "Row", "Column", "Textbox", "Button", "Slider",
                 "Markdown", "Tabs", "TabItem", "Label", "Chatbot", "File",
                 "Accordion", "HTML", "Dataframe", "CheckboxGroup"]:
        setattr(gr, attr, MagicMock())

    # ---- faster_whisper --------------------------------------------------------
    fw = _stub("faster_whisper")
    fw.WhisperModel = MagicMock()

    # ---- langchain 0.3.x family ------------------------------------------------
    _stub("langchain_ollama")
    sys.modules["langchain_ollama"].ChatOllama = MagicMock()
    sys.modules["langchain_ollama"].OllamaEmbeddings = MagicMock()

    _stub("langchain_core")
    _stub("langchain_core.prompts")
    sys.modules["langchain_core.prompts"].PromptTemplate = MagicMock()
    _stub("langchain_core.runnables")
    sys.modules["langchain_core.runnables"].RunnableSequence = MagicMock()
    _stub("langchain_core.messages")
    sys.modules["langchain_core.messages"].AIMessage = MagicMock()

    _stub("langchain_text_splitters")
    sys.modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = MagicMock()

    _stub("langchain_community")
    _stub("langchain_community.vectorstores")
    sys.modules["langchain_community.vectorstores"].FAISS = MagicMock()

    # ---- qdrant & bm25 ---------------------------------------------------------
    _stub("langchain_qdrant")
    sys.modules["langchain_qdrant"].Qdrant = MagicMock()
    sys.modules["langchain_qdrant"].QdrantVectorStore = MagicMock()
    _stub("qdrant_client")
    sys.modules["qdrant_client"].QdrantClient = MagicMock()
    _stub("qdrant_client.models")
    sys.modules["qdrant_client.models"].VectorParams = MagicMock()
    sys.modules["qdrant_client.models"].Distance = MagicMock()
    _stub("rank_bm25")
    sys.modules["rank_bm25"].BM25Okapi = MagicMock()

    # ---- pyannote & torch ------------------------------------------------------
    _stub("pyannote")
    _stub("pyannote.audio")
    sys.modules["pyannote.audio"].Pipeline = MagicMock()
    _stub("torch")
    sys.modules["torch"].device = MagicMock()
    sys.modules["torch"].cuda = MagicMock()
    sys.modules["torch"].cuda.is_available = MagicMock(return_value=False)

    # ---- dotenv ----------------------------------------------------------------
    _stub("dotenv")
    sys.modules["dotenv"].load_dotenv = MagicMock()

    # ---- requests: mock Ollama health check ------------------------------------
    req = _stub("requests")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "models": [
            {"name": "llama3.1:8b-instruct-q8_0"},
            {"name": "mxbai-embed-large"},
        ]
    }
    req.get = MagicMock(return_value=mock_resp)

    # ---- tiktoken --------------------------------------------------------------
    tiktoken = _stub("tiktoken")
    enc = MagicMock()
    enc.encode.side_effect = lambda text: text.split()
    tiktoken.get_encoding = MagicMock(return_value=enc)