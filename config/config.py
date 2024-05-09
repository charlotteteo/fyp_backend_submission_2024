from typing import Optional

from langchain.embeddings.openai import OpenAIEmbeddings
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic_settings import BaseSettings
from langchain.vectorstores import DeepLake
import models as models


class Settings(BaseSettings):
    # database configurations, delete all hardcoded values later
    DATABASE_URL: Optional[str]
    ACTIVE_LOOP_API_TOKEN: Optional[str]
    OPEN_API_TOKEN: Optional[str]

    # JWT
    secret_key: str = "secret"
    algorithm: str = "HS256"

    class Config:
        env_file = ".env.dev"
        orm_mode = True


async def initiate_database():
    DATABASE_URL = 'mongodb+srv://charlotteteo:cocobelly@cluster0.bnrfiag.mongodb.net/quantfolio'
    client = AsyncIOMotorClient(DATABASE_URL,ssl=True)
    await init_beanie(
        database=client.get_default_database(), document_models=models.__all__
    )

# async def connect_to_active_loop(dataset_path: str, *args, **kwargs)-> Optional[DeepLake]:
#     try:
#         embeddings = OpenAIEmbeddings(model="text-embedding-ada-002" ,openai_api_key=Settings().OPEN_API_TOKEN)
#         db = DeepLake(dataset_path, *args, **kwargs, embedding=embeddings, token=Settings().ACTIVE_LOOP_API_TOKEN)
#         return db

#     except ConnectionError:
#         print("Failed to connect to the DeepLake database.")
#         return None

#     except ValueError as ve:
#         print(f"Value error encountered: {ve}")
#         return None

#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return None






     