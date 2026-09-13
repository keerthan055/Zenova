"""Entrypoint alias for ZENOVA FastAPI application."""
from zenova.api.app import app, main

if __name__ == "__main__":
    main()

__all__ = ["app", "main"]

