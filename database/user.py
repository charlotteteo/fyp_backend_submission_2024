from typing import List, Union
from beanie.odm.documents import PydanticObjectId
from config.config import Settings
from models.user import User



user_collection = User


async def retrieve_users() -> List[User]:
    users = await user_collection.all().to_list()
    return users


async def add_user(new_user: User) -> User:
    user = await new_user.create()
    return user


async def retrieve_user(id: str) -> User:
    user = await user_collection.get(id)
    if user:
        return user


async def delete_user(id: str) -> bool:
    user = await user_collection.get(id)
    if user:
        await user.delete()
        return True



# async def update_user_data(id: str, data: dict) -> Union[bool, User]:
#     des_body = {k: v for k, v in data.items() if v is not None}
#     update_query = {
#         "$set": {field: value for field, value in des_body.items()}}
#     user = await user_collection.get(id)
#     if user:
#         await user.update(update_query)
#         return user
#     return False
async def update_user_data(id: str, data: dict) -> Union[bool, User]:
    user = await retrieve_user(id)
    if user:
        # Modify only the related keys
        for key, value in data.items():
            if value is not None:
                setattr(user, key, value)
        await user.save()
        return user
    return False







    ## include retrieve_user_description - where gpt describes user

    

        
