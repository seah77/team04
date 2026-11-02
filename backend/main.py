from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 상대 경로로 수정
from .routers import ocr

app = FastAPI()

# CORS 설정 등 기존 코드...
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(ocr.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Food Safety Validation API is running!"}