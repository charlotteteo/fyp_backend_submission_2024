from email.policy import default
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.prompts import PromptTemplate
from logic.langchain_portfolio_tools import *
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.chains import SimpleSequentialChain
from logic.prompt_templates import CustomPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
os.environ['OPENAI_API_KEY'] = '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'
os.environ['SERPAPI_API_KEY'] = '85ce8786996e0fa5568e8c4db622cde5b9e883a9667abf39e3f51e3397a3b8e7'
os.environ['ACTIVELOOP_TOKEN'] = 'eyJhbGciOiJIUzUxMiIsImlhdCI6MTY5NDA3Mzk3OSwiZXhwIjoxNzA4Njc1NTAwfQ.eyJpZCI6ImNoYXJsb3R0ZXRlb2N0In0.1b3mHSwu8l4bGj_YRZrteBLXO50E9ydbUnvvPVPMqMJrgHa_Y8aUSZ0fl-8CPgxzmlKra09WvHF6NWcFU5BGKw'


class PortfolioLLMSetup():
    search = SerpAPIWrapper()
    default_tools = [CurrentStockPriceTool(), StockPerformanceTool(), StockNewsTool(),
                     StockFundamentalsTool(), PortfolioEvaluationTool(),
                     Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events.  You should enter the stock ticker symbol recognized by the yahoo finance."
    ),]

    def __init__(self) -> None:
        self.llm = ChatOpenAI(model="gpt-4 turbo", temperature=0.1,
                              openai_api_key='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')

    def initialise_agent(self, agent_type, verbose=False, tools=default_tools):
        agent = initialize_agent(
            tools, self.llm, agent=agent_type, verbose=verbose)
        return agent

    def initialise_chatbot(self, verbose=False, tools=default_tools):
        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        }
        memory = ConversationBufferMemory(
            memory_key="memory", return_messages=True)
        agent = initialize_agent(tools=tools, llm=self.llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS,
                                 verbose=verbose, memory=memory, agent_kwargs=agent_kwargs)

        return agent

    def intialise_llmchain(self, temperature: float, template: str):
        llm = OpenAI(temperature=temperature)
        return LLMChain(llm=llm, prompt=template)

    def qualitative_summary_using_llm(self, stocks: list, weights: list, start_date: str, end_date: str, initial_investment: float):
        agent = self.initialise_agent(
            AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True)
        summary_template = PromptTemplate(
            input_variables=["portfolio_info"], template=CustomPromptTemplate.SUMMARY_PROMPT_TEMPLATE_CHAIN2)
        review_chain = self.intialise_llmchain(
            temperature=0.5, template=summary_template)
        overall_chain = SimpleSequentialChain(
            chains=[agent, review_chain],
            verbose=True)
        return overall_chain.run(CustomPromptTemplate.SUMMARY_PROMPT_TEMPLATE_CHAIN1.format(stocks, weights, start_date, end_date, initial_investment, stocks)
                                 )

    def qualitative_summary_using_llm_one_shot(self, stocks: list, weights: list, start_date: str, end_date: str, initial_investment: float):
        agent = self.initialise_agent(
            AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True)
        return agent.run(CustomPromptTemplate.SUMMARY_PROMPT_TEMPLATE_ONE_SHOT.format(stocks, weights, start_date, end_date, initial_investment, stocks)
                         )


if __name__ == "__main__":
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    weights = [0.25, 0.25, 0.25, 0.25]
    start_date = '2012-01-01'
    end_date = '2022-12-31'
    initial_investment = 1000000
    PortfolioLLMSetup().qualitative_summary_using_llm(
        stocks, weights, start_date, end_date, initial_investment)
