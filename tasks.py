from crewai import Task
from tools import <tool_names>
from agents import WeatherAgent, NewsAgent, OptimizationAgent
from agents import ItineraryGenerationAgent, InteractionAgent

news_task = Task(
    description = (

    )
    expected_output = 
    tools = [tools_names]
    agent = NewsAgent
)
we can also have some other tasks
task2 = task(
    
    async_execution = False,
    output_file or something
    tools = []
    agent = some agent
)