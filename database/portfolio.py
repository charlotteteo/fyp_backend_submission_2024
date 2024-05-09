from time import strftime
from typing import List, Optional, Union
from beanie.odm.documents import PydanticObjectId
from models.portfolio.portfolio import Portfolio
from models.portfolio.portfolio_analysis import PortfolioAnalysis
from models.user import User
from bson.objectid import ObjectId
# from logic.langchain_portfolio_agents import PortfolioLLMSetup
from logic.portfolio_analyzer import PortfolioAnalyzer

user_collection = User


async def retrieve_portfolios(user_id: str) -> List[Portfolio]:
    user = await user_collection.get(user_id)
    if user:
        return user.portfolio
    else:
        return None


async def add_portfolio(user_id: str, new_portfolio: Portfolio) -> Portfolio:
    user = await user_collection.get(user_id)
    portfolio = await new_portfolio.create()
    if user:
        user.portfolios.append(portfolio)
        update_query = {"$set": {"portfolio": user.portfolios}}
        await user.update(update_query)
        return new_portfolio
    return None


async def retrieve_portfolio(user_id: str, portfolio_name: str) -> Optional[Portfolio]:
    user = await user_collection.get(user_id)
    if user:
        return next((p for p in user.portfolios if p.portfolio_name == portfolio_name), None)
    return None


async def delete_portfolio(user_id: str, portfolio_name: str) -> bool:
    user = await user_collection.get(user_id)
    if user:
        portfolio = next(
            (p for p in user.portfolios if p.portfolio_name == portfolio_name), None)
        if portfolio:
            user.portfolios = [
                p for p in user.portfolios if p.id != portfolio.id]
            await portfolio.delete()
        await user.update({"$set": {"portfolios": user.portfolios}})
        return True
    return False


async def update_portfolio_data(user_id: str, portfolio_name: str, data: dict) -> Union[bool, Portfolio]:
    user = await user_collection.get(user_id)
    if user:
        portfolio = next(
            (p for p in user.portfolios if p.portfolio_name == portfolio_name), None)
        des_body = {k: v for k, v in data.items() if v is not None}
        update_query = {
            "$set": {field: value for field, value in des_body.items()}}

        if portfolio:
            await portfolio.update(update_query)
            return portfolio
    return False






# async def retrieve_portfolio_analysis(user_id: PydanticObjectId, portfolio_name: str) -> Optional[PortfolioAnalysis]:
#     user = await user_collection.get(user_id)
#     if user:
#         portfolio: Portfolio = next(
#             (p for p in user.portfolios if p.portfolio_name == portfolio_name), None)
#         if portfolio != None:
#             stocks = list(portfolio.latest_day_weights.keys())
#             weights = list(portfolio.latest_day_weights.values())
#             qualitative_summary: str = PortfolioLLMSetup().qualitative_summary_using_llm(
#                 stocks, weights, portfolio.start_date, '2023-09-30', portfolio.initial_value)
#             portfolio_evaluator: PortfolioAnalyzer = PortfolioAnalyzer(
#                 stocks, weights, portfolio.start_date, '2023-09-30', portfolio.initial_value)
#             metrics = portfolio_evaluator.calculate_metrics()
#             portfolio_analysis = PortfolioAnalysis(
#                 qualitative_summary=qualitative_summary, quantitative_metrics=metrics)
#             return portfolio_analysis
#     return None
