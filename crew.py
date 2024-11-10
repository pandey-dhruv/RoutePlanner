from crewai import Crew, Process
from agents import NewsAgent, WeatherAgent, OptimizationAgent
from agents import ItineraryGenerationAgent, InteractionAgent
from tasks import *

crew = Crew(
    agents = [NewsAgent, WeatherAgent, OptimizationAgent,
              ItineraryGenerationAgent, InteractionAgent],
    tasks = [],
    process = Process.sequential(),
    memory = True,
    cache = True,
    max_rpm = 100,
    share_crew = True
)
