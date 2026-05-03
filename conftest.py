"""
conftest.py — stubs all heavy dependencies before the app module is imported.
Run tests with:
    pytest -v
Place conftest.py in the same directory as app.py and test_api.py.
Every stub is installed into sys.modules inside pytest_configure, which runs
before any test module is collected or imported. This guarantees that when
test_api.py does `import app`, the heavy libraries are 
already replaced by lightweight in-memory fakes.
"""
import sys
import types
from unittest.mock import MagicMock

def _stub(name: str) -> types.ModuleType:
    """Create an empty module, register it in sys.modules, and return it."""
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

def pytest_configure(config):  # noqa: ARG001
    # Register custom markers to suppress PytestUnknownMarkWarning
    config.addinivalue_line(
        "markers", "integration: marks tests requiring real Ollama/Whisper models"
    )
    # ---- gradio ----------------------------------------------------------------
    gr = _stub("gradio")
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=MagicMock())
    ctx.__exit__ = MagicMock(return_value=False)
    gr.Blocks = MagicMock(return_value=ctx)
    for attr in ["State", "Row", "Column", "Textbox", "Button",
                 "Slider", "Markdown", "Tabs", "TabItem", "Label", "Chatbot", "File", "Accordion", "HTML"]:
        setattr(gr, attr, MagicMock())

    # ---- faster_whisper --------------------------------------------------------
    fw = _stub("faster_whisper")
    fw.WhisperModel = MagicMock()

    # ---- langchain family ------------------------------------------------------
    for mod_name in [
        "langchain", "langchain.chains", "langchain.prompts", "langchain.text_splitter",
        "langchain_community", "langchain_community.embeddings", "langchain_community.llms",
        "langchain_community.vectorstores", "langchain_qdrant",
    ]:
        _stub(mod_name)

    sys.modules["langchain.chains"].LLMChain = MagicMock()
    sys.modules["langchain.prompts"].PromptTemplate = MagicMock()
    sys.modules["langchain.text_splitter"].RecursiveCharacterTextSplitter = MagicMock()
    sys.modules["langchain_community.embeddings"].OllamaEmbeddings = MagicMock()
    sys.modules["langchain_community.llms"].Ollama = MagicMock()
    sys.modules["langchain_community.vectorstores"].FAISS = MagicMock()
    sys.modules["langchain_qdrant"].Qdrant = MagicMock()
    sys.modules["langchain_qdrant"].QdrantVectorStore = MagicMock()

    # ---- qdrant & bm25 ---------------------------------------------------------
    qc = _stub("qdrant_client")
    qc.QdrantClient = MagicMock()
    qc.models = MagicMock()
    qc.models.VectorParams = MagicMock()
    qc.models.Distance = MagicMock()
    _stub("rank_bm25")
    sys.modules["rank_bm25"].BM25Okapi = MagicMock()

    # ---- requests: make ensure_ollama_ready() pass at import time --------------
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