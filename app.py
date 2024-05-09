from fastapi import FastAPI, Depends

from auth.jwt_bearer import JWTBearer
from config.config import initiate_database
from routes.admin import router as AdminRouter
from routes.user import router as UserRouter
from routes.portfolio import router as PortfolioRouter
from routes.analysis import router as StockAnalysisRouter
from routes.chatbot import router as ChatbotRouter
from routes.news import router as NewsFeedRouter
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


token_listener = JWTBearer()


@app.on_event("startup")
async def start_database():
    await initiate_database()


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the portfolio management app, charlotte."}


app.include_router(AdminRouter, tags=["Administrator"], prefix="/admin")
app.include_router(
    UserRouter,
    tags=["User"],
    prefix="/users",
)
app.include_router(PortfolioRouter)
app.include_router(StockAnalysisRouter)
app.include_router(ChatbotRouter)
app.include_router(NewsFeedRouter)