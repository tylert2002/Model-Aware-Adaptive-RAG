"""Serve the unchanged upstream retriever against isolated ES on port 9201."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'adaptive-rag' / 'retriever_server'))
import serve
from unified_retriever import UnifiedRetriever
import uvicorn

if __name__ == '__main__':
    serve.retriever = UnifiedRetriever(host='http://127.0.0.1/', port=9201)
    uvicorn.run(serve.app, host='127.0.0.1', port=8000)
