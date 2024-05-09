from typing import Optional, List, Dict
from enum import Enum
from beanie import Document
from pydantic import BaseModel, EmailStr, Field
from models.portfolio.portfolio import Portfolio
# from models.portfolio.portfolio_analysis import PortfolioChatBot
from beanie.odm.documents import PydanticObjectId
import uuid


class User(Document):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    personalInformation: Optional[dict]
    financialSituation: Optional[dict]
    investmentObjectives: Optional[dict]
    riskWillingness: Optional[dict]
    riskCapacity: Optional[dict]
    portfolio: Optional[dict]
    chatbotHistory: Optional[dict]

    class Config:
        schema_extra = {
            "personalInformation": {
                "fullName": "John Doe",
                "dob": "1990-01-01",
                "email": "john@example.com",
                "contact": "1234567890",
                "occupation": "Software Engineer",
                "gender": "male",
                "maritalStatus": "married",
                "dependents": 2
            },
            "financialSituation": {
                "netWorth": "500000",
                "majorExpenses": ["education", "houseConstruction"],
                "earningCapacity": "70000"
            },
            "investmentObjectives": {
                "investmentTimeHorizon": [10, 20],
                "returnRequirements": "5",
                "objectivePrioritisation": "growth",
                "futureFinancialGoals": ["retirement", "house"]
            },
            "riskWillingness": {
                "investmentKnowledge": "some",
                "riskPerception": "understand",
                "responseToLoss": "reallocate",
                "regretAversion": "reluctantlyDecide"
            },
            "riskCapacity": {
                "initialInvestment": "5000-25000",
                "acceptableReturns": "-15%-25%",
                "monthlyContribution": ">=10%",
                "withdrawalStartTime": "5-9years"
            },
            "portfolio":{
                "stocks":[],
                "weights":[],
                "start_date":[],
                "end_date":[]
            },
            "chatbotHistory": {
                "date": [],
                "topic": []
            }
            }
    class Settings:
        name = 'user_data'





class UpdateUserModel(BaseModel):
    # fullname: Optional[str]
    # email: Optional[EmailStr]
    # age: Optional[int]
    # occupation: Optional[str]
    # annual_income: Optional[float]
    # net_worth: Optional[float]
    # investment_experience: Optional[int]
    # # risk_tolerance: Optional[int]
    # dependents: Optional[int]
    # married: Optional[bool]
    # gender: Optional[str]
    # monthly_expense: Optional[float]
    # open_ended_responses: Optional[Dict[str, str]]
    # chatbot_history: Optional[list]

    # class Collection:
    #     name = "users"

    # class Config:
    #     schema_extra = {
    #         "example": {
    #             "fullname": "Jane Doe",
    #             "email": "jane.doe@example.com",
    #             "age": 20,
    #             "occupation": "Data Scientist",
    #             "annual_income": 120000.0,
    #             "net_worth": 600000.0,
    #             "investment_experience": 1,
    #             "risk_tolerance": 1,
    #             "dependents": 1,
    #             "married": False,
    #             "gender": 2,
    #             "monthly_expense": 4000.0,
    #             "open_ended_responses": {"What is your primary investment goal?": "Retirement"},


    #         }
    #     }
    id: str
    personalInformation: Optional[dict]
    financialSituation: Optional[dict]
    investmentObjectives: Optional[dict]
    riskWillingness: Optional[dict]
    riskCapacity: Optional[dict]
    portfolio: Optional[dict]
    chatbot_history: Optional[dict]

    class Config:
        schema_extra = {
            "personalInformation": {
                "fullName": "John Doe",
                "dob": "1990-01-01",
                "email": "john@example.com",
                "contact": "1234567890",
                "occupation": "Software Engineer",
                "gender": "male",
                "maritalStatus": "married",
                "dependents": 2
            },
            "financialSituation": {
                "netWorth": "500000",
                "majorExpenses": ["education", "houseConstruction"],
                "earningCapacity": "70000"
            },
            "investmentObjectives": {
                "investmentTimeHorizon": [10, 20],
                "returnRequirements": "5",
                "objectivePrioritisation": "growth",
                "futureFinancialGoals": ["retirement", "house"]
            },
            "riskWillingness": {
                "investmentKnowledge": "some",
                "riskPerception": "understand",
                "responseToLoss": "reallocate",
                "regretAversion": "reluctantlyDecide"
            },
            "riskCapacity": {
                "initialInvestment": "5000-25000",
                "acceptableReturns": "-15%-25%",
                "monthlyContribution": ">=10%",
                "withdrawalStartTime": "5-9years"
            }
            }
    class Settings:
        name = 'user_data'
