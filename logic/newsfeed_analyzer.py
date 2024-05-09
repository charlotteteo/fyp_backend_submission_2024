import requests
import json 
from dotenv import load_dotenv
from newspaper import Article
import unicodedata
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import re


load_dotenv()
api_key = 'd22275525c964de4abeedbbcb8b7444b'

os.environ['OPENAI_API_KEY']='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'


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}
def get_agent_evaluator():
    return ChatOpenAI(model_name="gpt-4", temperature=0)


def get_article_summary(article_url, agent_evaluator=get_agent_evaluator()):
    session = requests.Session()
    try:
        response = session.get(article_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            article = Article(article_url)
            article.download()
            article.parse()
            
            print(f"Title: {article.title}")
            print(f"Text: {article.text}")
            
        else:
            print(f"Failed to fetch article at {article_url}")
    except Exception as e:
        print(f"Error occurred while fetching article at {article_url}: {e}")

    # Prepare template for prompt
    template = """You are a very good assistant that summarizes online articles.

    Here's the article you want to summarize.

    ==================
    Title: {article_title}
    {article_text}
    ==================

    Write a summary of the previous article.
    if article text is empty just return 'Quantfolio is unable to access the article directly. Please click on the link to read the full article.'
    """

    prompt = template.format(article_title=article.title, article_text=article.text)

    messages = [HumanMessage(content=prompt)]
    # Generate summary
    summary = agent_evaluator(messages)
    return summary.content

current_portfolio = "AAPL,AMZN,JNJ,V,NVDIA,TSLA"

def get_article_evaluation(article_url, agent_evaluator=get_agent_evaluator()):
    session = requests.Session()
    try:
        response = session.get(article_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            article = Article(article_url)
            article.download()
            article.parse()
            
            print(f"Title: {article.title}")
            print(f"Text: {article.text}")
            
        else:
            print(f"Failed to fetch article at {article_url}")
    except Exception as e:
        print(f"Error occurred while fetching article at {article_url}: {e}")

 
    
    # Prepare template for prompt
    template = """You are a very good assistant that reads and understands online articles and its effects on the stocks market.

    Here's the article you want to read.

    ==================
    Title: {article_title}
    {article_text}
    ==================

    Considering my current porttfolio of {current_portfolio}, please evaluate:
        - how this news might affect the stock price of these holdings
        - events/indicators to look out for moving forward
    """

    prompt = template.format(article_title=article.title, article_text=article.text, current_portfolio=current_portfolio)

    messages = [HumanMessage(content=prompt)]

    
    # Load the model
    
    # Generate summary
    evaluation = agent_evaluator(messages)
    return evaluation.content


def remove_special_characters(input_string):
    return re.sub(r'[^\w\s]', '', input_string)


def has_special_characters(input_string):
    for char in input_string:
        if unicodedata.category(char) in ['Pc', 'Pd', 'Pe', 'Pf', 'Pi', 'Po', 'Ps']:
            return True
    return False


def get_top_finance_news(query='', num_articles=5):

    """
    Fetches the top finance news articles from the News API and returns a list of dictionaries containing relevant information.

    Args:
        api_key (str): Your News API key.
        query (str): Optional query string to filter news articles (default is an empty string).
        num_articles (int): Number of articles to fetch (default is 5).

    Returns:
        list: A list of dictionaries containing information about each news article.
    """
    # Define parameters for the request
    params = {
        'apiKey': api_key,
        'category': 'business',  # Filter by the business category (finance news)
        'language': 'en',
        'pageSize': num_articles,
        'q': query  # Your desired search query
    }

    # Make a GET request to the News API
    response = requests.get('https://newsapi.org/v2/top-headlines', params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Initialize a list to store news articles
        news_list = []
        sid = SentimentIntensityAnalyzer()
        # Extract relevant information from each article and append it to the list
        for article in data['articles']:
            sentiment_score = sid.polarity_scores(article['title'])["compound"]
            news_dict = {
                'title':article['title'],
                # 'link': article['source']['name'],
                'date': article['publishedAt'],
                'summary': get_article_summary(article['url']),
                'sentiment':sentiment_score,
                # 'description': article['description'],
                'link': article['url']
            }
            # if len(article['description'])>0 and has_special_characters(article['description']):
            #     continue
            news_list.append(news_dict)
            # title = item.find("title").text
            #     link = item.find("link").text
            #     summary = get_article_summary(link)
            #     pub_date = item.find("pubDate").text  # Extract publication date
            #     sentiment_score = sid.polarity_scores(title)["compound"]
            #     # if self.has_special_characters(title):
            #     #     continue
            #     articles.append({"title": title, "link": link,'date':pub_date,
            #                     "sentiment": sentiment_score,"summary":summary})

        return news_list
    else:
        print("Error:", response.status_code)
        return []



# def get_related_news(tickers,number):
#     company_news = []
#     for x in tickers:
#         company_news.append(StockAnalyzer(x).get_top_company_news(number))
#     return company_news
    




# print(get_related_news(['AAPL','MSFT'],5))
