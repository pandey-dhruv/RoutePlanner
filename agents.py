from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = "NA"
model = ChatOpenAI(model = "crewai-llama3.1:latest", base_url="http://localhost:11434")

# this is the weather agent
WeatherAgent = Agent(
    role = "Weather Agent",
    goal = "To fetch the expected weather conditions of the region during the time of visit",
    backstory="You're working on planning a blog article "
              "about the topic: {topic} in 'https://medium.com/'."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "You have to prepare a detailed "
              "outline and the relevant topics and sub-topics that has to be a part of the"
              "blogpost."
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    llm=model,
    allow_delegation=True,
    verbose=True,
    tools = []
)

# this is the news agent
NewsAgent = Agent(
    role = "News Agent",
    goal = "To fetch any other activity in the area that might affect the actual plan and place	of visit",
    backstory="You're working on planning a blog article "
              "about the topic: {topic} in 'https://medium.com/'."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "You have to prepare a detailed "
              "outline and the relevant topics and sub-topics that has to be a part of the"
              "blogpost."
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    llm=model,
    allow_delegation=True,
    verbose=True,
    tools = []
)

# this is the weather agent
OptimizationAgent = Agent(
    role = "Optimization Agent",
    goal = "To optimize travel paths based on budget, preferences, and time constraints",
    backstory="You're working on planning a blog article "
              "about the topic: {topic} in 'https://medium.com/'."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "You have to prepare a detailed "
              "outline and the relevant topics and sub-topics that has to be a part of the"
              "blogpost."
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    llm=model,
    allow_delegation=True,
    verbose=True,
    tools = []
)

ItineraryGenerationAgent = Agent(
    role = "Itinerary Generation Agent",
    goal = "To generate an initial itinerary based on user preferences	and	inputs",
    backstory="You're working on planning a blog article "
              "about the topic: {topic} in 'https://medium.com/'."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "You have to prepare a detailed "
              "outline and the relevant topics and sub-topics that has to be a part of the"
              "blogpost."
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    llm=model,
    allow_delegation=True,
    verbose=True,
    tools = []
)

InteractionAgent = Agent(
    role = "User Interaction Agent",
    goal = "To gather user preferences and collect required	details",
    backstory="You're working on planning a blog article "
              "about the topic: {topic} in 'https://medium.com/'."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "You have to prepare a detailed "
              "outline and the relevant topics and sub-topics that has to be a part of the"
              "blogpost."
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    llm=model,
    allow_delegation=True,
    verbose=True,
    tools = []
)

