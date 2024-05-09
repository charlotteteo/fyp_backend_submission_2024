from fastapi import APIRouter, Body, HTTPException
from beanie.odm.documents import PydanticObjectId
from database.user import *
from models.user import *

router = APIRouter()
user_collection = User

@router.get("/", response_description="Users retrieved")
async def get_users():
    users = await retrieve_users()
    return {
        "status_code": 200,
        "response_type": "success",
        "description": "User data retrieved successfully",
        "data": users,
    }


@router.get("/{id}", response_description="User data retrieved")
async def get_user_data(id: str):
    user = await retrieve_user(id)
    if user:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": "User data retrieved successfully",
            "data": user,
        }
    return {
        "status_code": 404,
        "response_type": "error",
        "description": "User doesn't exist",
    }

@router.post("/", response_description="User data added into the database")
async def add_user_data(user: User = Body(...)):
    new_user = await add_user(user)
    return {
        "status_code": 200,
        "response_type": "success",
        "description": "User created successfully",
        "data": new_user,
    }


@router.delete("/{id}", response_description="User data deleted from the database")
async def delete_user_data(id: str):
    deleted_user = await delete_user(id)
    if deleted_user:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": f"User with ID: {id} removed",
        }
    return {
        "status_code": 404,
        "response_type": "error",
        "description": f"User with ID: {id} doesn't exist",
    }


@router.put("/{id}")
async def update_user(id: str, req: dict):
    updated_user = await update_user_data(id, req)
    if updated_user:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": f"User with ID: {id} updated",
            "data": updated_user,
        }
    return {
        "status_code": 404,
        "response_type": "error",
        "description": f"An error occurred. User with ID: {id} not found",
    }


@router.get("/login/{email}/{password}")
async def login(email: str, password: str):
    # Search for the user with the provided email in the database
    # user = next((user_data for user_data in user_collection.values() if user_data["personalInformation"]["email"] == email), None)
    user = await User.find_one(User.personalInformation.email == email and User.personalInformation.password == password )

    
    # If user is not found, raise 404 error
    if user is None:
        raise HTTPException(status_code=404, detail="Email not found")
    
    # Check if the provided password matches the password stored in the database    
    # Return success message if login credentials are correct
    # return {"message": "Login successful"}
    return user


