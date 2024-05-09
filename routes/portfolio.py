from os import strerror
from fastapi import APIRouter, Body
from beanie.odm.documents import PydanticObjectId
from database.portfolio import *
from database.user import retrieve_user
from models.portfolio.portfolio import *
from logic.portfolio_analyzer import PortfolioAnalyzer
from logic.portfolio_recommender import *
router = APIRouter(tags=["Portfolio"], prefix="/portfolio")


@router.get("/{id}/portfolios", response_description="Portfolios retrieved")
async def get_portfolios(u_id: str):
    portfolios = await retrieve_portfolios(u_id)
    if portfolios != None:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": "Portfolios retrieved successfully",
            "data": portfolios,
        }
    return {
        "status_code": 404,
        "response_type": "error",
        "description": "User not found",
    }


@router.get("/portfolios/{id}/{p_name}", response_description="Portfolio retrieved")
async def get_portfolio(u_id: str, p_name: str):
    portfolio: Portfolio = await retrieve_portfolio(u_id, p_name)

    if portfolio:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": "Portfolio retrieved successfully",
            "data": portfolio,
        }
    return {
        "status_code": 404,
        "response_type": "error",
        "description": "Portfolio not found for the user",
    }
    
@router.get("/portfolio_trades/{id}/{p_name}", response_description="Portfolio retrieved")
async def get_portfolio_trades(u_id: str, p_name: str):
    portfolio: Portfolio = await retrieve_portfolio(u_id, p_name)

    if portfolio:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": "Portfolio retrieved successfully",
            "data": portfolio.portfolio_trades,
        }
    return {
        "status_code": 404,
        "response_type": "error",
        "description": "Portfolio not found for the user",
    }


@router.post("/portfolios/{id}", response_description="Portfolio added to user")
async def add_portfolio_to_user(u_id: str, portfolio: Portfolio):
    user = await add_portfolio(u_id, portfolio)
    if user:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": "Portfolio added successfully",
            "data": user,
        }
    return {
        "status_code": 404,
        "response_type": "error",
        "description": "User not found",
    }


@router.delete("/{id}/portfolios/{p_name}", response_description="Portfolio removed from user")
async def remove_portfolio_from_user(u_id: str, p_name: str):
    is_deleted = await delete_portfolio(u_id, p_name)
    if is_deleted:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": f"Portfolio with ID: {p_name} removed",
        }
    return {
        "status_code": 404,
        "response_type": "error",
        "description": "User or portfolio not found",
    }


@router.post("/{id}/portfolios/{p_name}", response_description="Portfolio updated")
async def update_user_portfolio(u_id: PydanticObjectId, p_name: str, req: UpdatePortfolioModel = Body(...)):
    updated_portfolio = await update_portfolio_data(u_id, p_name, req.dict(exclude_unset=True))
    if updated_portfolio:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": f"Portfolio with ID: {p_name} updated",
            "data": updated_portfolio,
        }
    return {
        "status_code": 404,
        "response_type": "error",
        "description": "User or portfolio not found",
    }




@router.get("/constituent_ts/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def constituent_ts(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)
    return portfolio_evaluator.download_data()

@router.get("/constituent_table/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def constituent_table(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)
    
    timeseries_dict = {}
    timeseries_dict['value'] = portfolio_evaluator.get_correct_format_portfolio_ts()
    # timeseries_dict['allocation_drift'] = portfolio_evaluator.get_allocation_drift_ts()

    return portfolio_evaluator.constituent_table()

@router.get("/time_series/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def portfolio_timeseries(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)
    
    timeseries_dict = {}
    timeseries_dict['value'] = portfolio_evaluator.get_correct_format_portfolio_ts()
    # timeseries_dict['allocation_drift'] = portfolio_evaluator.get_allocation_drift_ts()

    return portfolio_evaluator.get_correct_format_portfolio_ts()


@router.get("/portfolio_summary/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def portfolio_summary(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)
    
    return portfolio_evaluator.get_portfolio_summary()



@router.get("/portfolio_rec/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def portfolio_rec(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)
    
    return portfolio_evaluator.get_portfolio_rec()


@router.get("/return_ts/{interval}/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def return_ts(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float, interval: str):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)

    return portfolio_evaluator.returns_ts(interval=interval,convert_data = True)

  
@router.get("/max_drawdown_ts/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def max_drawdown_ts(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)

    return portfolio_evaluator.max_drawdown_ts()  

@router.get("/allocation_drift_timeseries/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def allocation_drift_timeseries(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)

    return portfolio_evaluator.get_allocation_drift_ts()


@router.get("/metrics/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def portfolio_metrics(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)
    return portfolio_evaluator.calculate_metrics()




@router.get("/top_movers/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def top_movers(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)
    return portfolio_evaluator.top_movers()





@router.get("/sectoral_ts/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def sectoral_ts(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)
    
    timeseries_dict = {}
    timeseries_dict['value'] = portfolio_evaluator.get_correct_format_portfolio_ts()
    # timeseries_dict['allocation_drift'] = portfolio_evaluator.get_allocation_drift_ts()

    return portfolio_evaluator.sectoral_exposure(convert_data=True)


@router.get("/sectoral_returns/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def sectoral_returns(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)
    
    timeseries_dict = {}
    timeseries_dict['value'] = portfolio_evaluator.get_correct_format_portfolio_ts()
    # timeseries_dict['allocation_drift'] = portfolio_evaluator.get_allocation_drift_ts()

    sector_sum_df = portfolio_evaluator.sectoral_exposure(convert_data=False)

    return portfolio_evaluator.convert_data(sector_sum_df.pct_change().dropna(),round_digit=4)


@router.get("/sectoral_metrics/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def sectoral_metrics(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)
    
    timeseries_dict = {}
    timeseries_dict['value'] = portfolio_evaluator.get_correct_format_portfolio_ts()
    # timeseries_dict['allocation_drift'] = portfolio_evaluator.get_allocation_drift_ts()

    return portfolio_evaluator.sectoral_metrics()


@router.get("/sectoral_pie/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def sectoral_pie(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)
    
    timeseries_dict = {}
    timeseries_dict['value'] = portfolio_evaluator.get_correct_format_portfolio_ts()
    # timeseries_dict['allocation_drift'] = portfolio_evaluator.get_allocation_drift_ts()

    return portfolio_evaluator.sectoral_pie_chart_data()



@router.get("/type_pie/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def type_pie(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)
    
    timeseries_dict = {}
    timeseries_dict['value'] = portfolio_evaluator.get_correct_format_portfolio_ts()
    # timeseries_dict['allocation_drift'] = portfolio_evaluator.get_allocation_drift_ts()

    return portfolio_evaluator.type_pie_chart_data()


@router.get("/constituent_pie/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def constituent_pie(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)
    
    timeseries_dict = {}
    timeseries_dict['value'] = portfolio_evaluator.get_correct_format_portfolio_ts()
    # timeseries_dict['allocation_drift'] = portfolio_evaluator.get_allocation_drift_ts()

    return portfolio_evaluator.constituent_pie()



@router.get("/sectoral_box_chart/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def sectoral_box(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)
    
    timeseries_dict = {}
    timeseries_dict['value'] = portfolio_evaluator.get_correct_format_portfolio_ts()
    # timeseries_dict['allocation_drift'] = portfolio_evaluator.get_allocation_drift_ts()

    return portfolio_evaluator.sectoral_box_plot()


@router.get("/sectoral_corr_chart/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def sectoral_corr(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)
    
    timeseries_dict = {}
    timeseries_dict['value'] = portfolio_evaluator.get_correct_format_portfolio_ts()
    # timeseries_dict['allocation_drift'] = portfolio_evaluator.get_allocation_drift_ts()

    return portfolio_evaluator.sectoral_corr_matrix()





@router.get("/drift_sectoral_ts/{stocks}/{weights}/{start_date}/{end_date}/{initial_investment}")
async def drift_sectoral_ts(stocks: str,weights: str,start_date: str,end_date: str,initial_investment: float):
    
    stocks = stocks.split(";")
    weights = weights.split(";")
    weights = [float(x) for x in weights]
    portfolio_evaluator = PortfolioAnalyzer(
        stocks, weights, start_date, end_date, initial_investment)
    
    timeseries_dict = {}
    timeseries_dict['value'] = portfolio_evaluator.get_correct_format_portfolio_ts()
    # timeseries_dict['allocation_drift'] = portfolio_evaluator.get_allocation_drift_ts()

    return portfolio_evaluator.get_allocation_drift_sectoral_ts()



@router.get("/recommendation_allocation/{tickers}/{start_date}/{end_date}")
async def recommended_portfolio_allocation(tickers,start_date,end_date):
    tickers = tickers.split(";")

    # tickers = ['AAPL','GOOGL','TSLA','JNJ']
    # start_date = '2020-01-01'
    # end_date = '2023-12-31'
    step_size = 21

    # print(get_recommended_allocation(tickers,start_date,end_date,step_size))
    return get_recommended_allocation(tickers,start_date,end_date,step_size)