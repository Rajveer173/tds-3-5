# Compatibility shim: some deploys start `uvicorn app:app`, others `uvicorn main:app`.
# The real server lives in main.py.
from main import app  # noqa: F401
