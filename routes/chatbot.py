from fastapi import APIRouter, Body, Depends, HTTPException
from database.chatbot import *
from database.user import *
from models.portfolio.portfolio_rec import *
from models.user import *
import ast

router = APIRouter(tags=["chatbot"], prefix="/chatbot")


@router.get("/chatbot_reply/{id}/{prompt}")
async def get_chatbot_reply(id: str, prompt: str):
    # await initialise_chatbot(id)
    chatbot_reply = await retrieve_chatbot_reply(id, prompt)
    # await add_chatbot_reply(id,  chatbot_reply)
    if chatbot_reply:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": f"chatbot reply to {id}",
            "data": chatbot_reply,
        }

    return {
        "status_code": 404,
        "response_type": "chatbot error",
        "description": f"An error occurred",
    }

@router.get("/chatbot_user_description/{id}/")
async def get_chatbot_description(id: str):
    # await initialise_chatbot(id)
    chatbot_reply = await retrieve_user_description(id)
    if chatbot_reply:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": f"chatbot reply to {id}",
            "data": chatbot_reply,
        }

    return {
        "status_code": 404,
        "response_type": "chatbot error",
        "description": f"An error occurred",
    }


@router.get("/chatbot_user_allocation/{id}/")
async def get_chatbot_allocation(id: str):
    # await initialise_chatbot(id)
    chatbot_reply = await retrieve_user_stock_allocation(id)
    if chatbot_reply:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": f"chatbot reply to {id}",
            "data": chatbot_reply,
        }

    return {
        "status_code": 404,
        "response_type": "chatbot error",
        "description": f"An error occurred",
    }

@router.get("/chatbot_history/{id}")
async def get_chatbot_history(id: str):
    chatbot_memory = await get_chatbot_memory(id)
    if chatbot_memory:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": f"chatbot memory of {id}",
            "data": chatbot_memory,
        }

    return {
        "status_code": 404,
        "response_type": "chatbot error",
        "description": "An error occurred",
    }


# actual route not used
@router.post("/update_chatbot_history/{id}")
async def update_chatbot_history(id: PydanticObjectId, portfolio_chatbot_addition):
    chatbot_memory = await add_chatbot_reply(id, portfolio_chatbot_addition)
    if chatbot_memory:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": f"chatbot memory of {id} updated",
            "data": chatbot_memory,
        }

    return {
        "status_code": 404,
        "response_type": "chatbot error",
        "description": "An error occurred",
    }
