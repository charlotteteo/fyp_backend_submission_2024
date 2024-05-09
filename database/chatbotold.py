from typing import List, Optional, Union
from beanie.odm.documents import PydanticObjectId
from database.portfolio import retrieve_portfolios
from models.portfolio.portfolio_analysis import PortfolioChatBot
from models.user import User
import datetime
from logic.langchain_chatbot import *

user_collection = User


agent_executor = initialise_agent()

async def retrieve_chatbot_reply(user_id: str, prompt: str) -> Optional[PortfolioChatBot]:
    # if user_id in dictionary_of_agents.keys():
    #     agent_executor = dictionary_of_agents[user_id] 
    # else:
    #     agent_executor = initialise_chatbot(user_id)
    user = await user_collection.get(user_id)
  
    if user:
        initial_reply, chat_history_initial= get_reply(agent_executor,f'here is info on user:{user}. use this to contextualise future answers')
        # print(chat_history_initial)
        chatbot_reply, chat_history = get_reply(agent_executor,prompt, chat_history_initial)
        return chatbot_reply
    return None





async def get_chatbot_memory(user_id: PydanticObjectId) -> Optional[List[PortfolioChatBot]]:
    user = await user_collection.get(user_id)
    if user:
        return user.chatbot_history
    else:
        return None


async def add_chatbot_reply(user_id: str, portfolio_chatbot_addition: PortfolioChatBot) -> Optional[List[PortfolioChatBot]]:
    user = await user_collection.get(user_id)
    if user:
    # Append the new chatbot reply to the chatbot memory
        user.chatbot_history.append(portfolio_chatbot_addition)
        update_query = {"$set": {"chatbot_history": user.chatbot_history}}
        await user.update(update_query)

        return user.chatbot_history
    else:
        return None


async def retrieve_user_description(user_id: str):
    user = await user_collection.get(user_id)
    user_info ={    "full_name": user.full_name,
      "email": user.email,
      "age": user.age,
      "occupation": user.occupation,
      "annual_income": user.annual_income,
      "net_worth": user.net_worth,
      "investment_experience": user.investment_experience,
      "dependents": user.dependents,
      "married": user.married,
      "gender": user.gender,
      "open_ended_responses": user.open_ended_responses}
    print(user_info)
    result, chat_history = get_reply(agent_executor, 
    f"""You are a powerful and versatile financial robo-advisor designed to assist users with a wide range of financial inquiries. .n\n\
        please evaluate the user below: {user_info}, describe her risk tolerance, risk profile etc."""
                   ,chat_history = [])

    return result

async def calculate_risk(user):
    if user:
       
        investment_knowledge_scores = {
            "no": 0,
            "some": 7,
            "good": 11,
            "extensive": 15
        }
        risk_willingness_score = investment_knowledge_scores.get(user.riskWillingness['investmentKnowledge'], 0)

        risk_perception_scores = {
            "worried": 0,
            "understand": 3,
            "opportunity": 7,
            "thrill": 10
        }
        risk_willingness_score += risk_perception_scores.get(user.riskWillingness['riskPerception'], 0)

        response_to_loss_scores = {
            "sellEverything": 0,
            "sellSome": 10,
            "doNothing": 5,
            "reallocate": 20,
            "buyMore": 40
        }
        risk_willingness_score += response_to_loss_scores.get(user.riskWillingness['responseToLoss'], 0)

        regret_aversion_scores = {
            "avoidDecisions": 0,
            "reluctantlyDecide": 6,
            "confidentlyDecide": 15
        }
        risk_willingness_score += regret_aversion_scores.get(user.riskWillingness['regretAversion'], 0)

        # Map React values to corresponding scores for risk capacity
        initial_investment_scores = {
            "5000-25000": 5,
            "moreThan25000": 10
        }
        risk_capacity_score = initial_investment_scores.get(user.riskCapacity['initialInvestment'], 0)

        acceptable_returns_scores = {
            "-10%-15%": 0,
            "-15%-25%": 5,
            "-25%-35%": 8,
            "-30%-45%": 10,
            "-35%-50%": 13,
            "-40%-55%": 17,
            "-45%-60%": 20
        }
        risk_capacity_score += acceptable_returns_scores.get(user.riskCapacity['acceptableReturns'], 0)

        monthly_contribution_scores = {
            "<10%": 5,
            ">=10%": 10
        }
        risk_capacity_score += monthly_contribution_scores.get(user.riskCapacity['monthlyContribution'], 0)
        withdrawal_start_time_scores = {
            "1-4years": 0,
            "5-9years": 5,
            "10-19years": 20,
            "over19years": 35
        }
        risk_capacity_score += withdrawal_start_time_scores.get(user.riskCapacity['withdrawalStartTime'], 0)

        return {"risk_willingness_score": risk_willingness_score, "risk_capacity_score": risk_capacity_score}

async def retrieve_user_stock_allocation(user_id: str):
    user = await user_collection.get(user_id)
    risk = calculate_risk(user)
    result, chat_history = get_reply(agent_executor, 
    f"""You are a powerful and versatile financial robo-advisor designed to assist users with a wide range of financial inquiries. .n\n\
        considering the user below: {user}, especially age, risk tolerance, risk profile etc.
        After a survey, the user is found to have {risk} against 100
        Recommend an actual allocation between stocks and bonds:
        Consider these information to substantiate
       
        Risk Tolerance Allocation:
        Determine the individual's risk tolerance through given questionnaire results.
        Allocate assets based on the risk tolerance level, with higher risk tolerance leading to a higher allocation to stocks and lower risk tolerance leading to a higher allocation to bonds and cash equivalents.
        Diversification Allocation:
        Allocate assets across different asset classes (stocks, bonds, real estate, etc.) to achieve diversification.
        Determine the allocation percentages for each asset class based on historical performance, correlation with other asset classes, and future outlook.
        Lifecycle Allocation:
        Divide the investment portfolio into different phases based on the individual's financial goals and time horizon.
        Allocate assets dynamically across phases, with a higher allocation to growth assets (stocks) in the early accumulation phase and a gradual shift towards more conservative assets (bonds, cash) as the individual approaches their financial goals.

        Write like you are conversationally speaking to her professionally! use 'your' and 'you'
        Reply should be in the format:
        " Dear /user name/, We recommend a 80/20 allocation of stocks to bonds. 
        Below are the reasons for our suggested allocation
        1.
        2.
        3. 
        
        Best Regards,
        QuantfolioX Advisor
        
        "

"""
                   ,chat_history = [])

    return result



