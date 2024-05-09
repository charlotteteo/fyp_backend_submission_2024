from models.stocks import StockAnalysis
from logic.stock_analyzer import StockAnalyzer


async def retrieve_stock_analysis(ticker: str) -> StockAnalysis:
    try:
        stock_object: StockAnalyzer = StockAnalyzer(ticker)
    except:
        return None
    if stock_object:
        stock_analysis = StockAnalysis(ticker=ticker, live_price=stock_object.get_cur_price(),
                                       company_name=stock_object.company_name,
                                       full_yfinance_data=stock_object.get_asset_info(),
                                       company_news=stock_object.get_top_company_news(summary=False),
                                       stock_performance=stock_object.evaluate_stock_performance())

        return stock_analysis
    return None


async def retrieve_stock_time_series(ticker: str, start_date_str: str, freq: str):
    try:
        stock_object: StockAnalyzer = StockAnalyzer(ticker)
    except:
        return None
    if stock_object:
        stock_analysis = stock_object.get_stock_data(
            start_date_str=start_date_str, freq=freq)

        return stock_analysis
    return None


async def retrieve_stock_fundamental_analysis(ticker: str):
    try:
        stock_object: StockAnalyzer = StockAnalyzer(ticker)
    except:
        return None
    if stock_object:
        return stock_object.get_yf_fundamentals_for_analysis()
    return None





async def retrieve_stock_technical_analysis(ticker: str,start_date:str):
    try:
        stock_object: StockAnalyzer = StockAnalyzer(ticker)
    except:
        return None
    if stock_object:
        return stock_object.get_technical_analysis(start_date)
    return None



    get_stock_summary


async def retrieve_stock_summary(ticker: str,start_date:str):
    try:
        stock_object: StockAnalyzer = StockAnalyzer(ticker)
    except:
        return None
    if stock_object:
        return stock_object.get_stock_summary(start_date)
    return None