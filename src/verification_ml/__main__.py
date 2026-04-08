"""Run: PYTHONPATH=src py -m verification_ml"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "verification_ml.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
