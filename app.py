# -*- coding: utf-8 -*-
import os
import uvicorn

if __name__ == "__main__":
    # Hugging Face Spaces runs on port 7860 by default
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port)
