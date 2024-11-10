from typing import Dict, Optional
from crew import Agent, Task, CrewAI
from crewai_tools import WebsiteRAGSearch
from datetime import datetime
import json

from langchain_openai import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = "NA"
model = ChatOpenAI(model = "crewai-llama3.1:latest", base_url="http://localhost:11434")

class WeatherAgent(Agent):
    def __init__(self):
        # initializing the parent Agent class
        super().__init__(
            name = "Weather Expert",
            goal = "Provide accurate and relevant weather information for trip planning",
            backstory = """You are an experienced meteorologist who specializes in analyzing weather patterns for
            travel planning. You have extensive knowledge of how weather conditions can affect different types
            of activities.""",
            tools = [self.get_current_weather, self.get_weather_forecast, self.analyze_weather_impact],
            verbose = True,
            allow_delegation = False,
            verbose = True,
            llm = model)
        # define helper function using Crew Tool
        self.weather_search = WebsiteRAGSearch(urls = [], chunk_size=500)
    
    def get_current_weather(self, location: str) -> Dict:
        """Gets the current weather condition of the location mentioned by the user using RAG search on
        the provided website links
        Args: Location -> Name of the city or location
        Returns: A dictionary containing the current weather data"""

        # We now construct a query for this task
        query = f"current weather conditions in {location}"
        search_results = self.weather_search.search(query)

        try:
            # we try and extract the relevant information from the website
            return {
                "location": location,
                "temperature": self._extract_temperature(search_results),
                "conditions": self._extract_conditions(search_results),
                "timestamp": datetime.now().isoformat(),
                "source": "Weather.com via RAG"
            }
        except Exception as e:
            return {
                "error": f"Failed to process weather data: {str(e)}",
                "raw_results": search_results
            }
    

    def get_weather_forecast(self, location: str, days: int = 5) -> Dict:
        """Gets the weather forecast for a given location using RAG search for the next 'days' days (default set to 5)
        Args:
            location: Name of the city or location
            days: Number of days for forecast
        Returns:
            Dictionary containing forecast data
        """
        query = f"{days}-day weather forecast for {location}"
        search_results = self.weather_search.search(query)
        
        try:
            return {
                "location": location,
                "forecast_days": days,
                "forecast": self._process_forecast(search_results),
                "timestamp": datetime.now().isoformat(),
                "source": "Weather.com via RAG"
            }
        except Exception as e:
            return {
                "error": f"Failed to process forecast: {str(e)}",
                "raw_results": search_results
            }   
