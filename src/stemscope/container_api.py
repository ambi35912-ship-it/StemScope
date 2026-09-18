"""Container entry point; publish the container port to localhost only."""

import uvicorn

from stemscope.api import create_app
from stemscope.logging_config import configure_logging

if __name__ == "__main__":
    configure_logging()
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
