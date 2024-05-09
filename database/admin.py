from typing import List, Union

from beanie import PydanticObjectId

from models.admin import Admin

admin_collection = Admin


async def add_admin(new_admin: Admin) -> Admin:
    admin = await new_admin.create()
    return admin

