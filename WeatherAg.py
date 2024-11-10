from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os
from typing import Dict, List
import json

class WeatherDataExtractor:
    def __init__(self, openai_api_key: str):
        """Initialize the weather data extractor with OpenAI API key."""
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            temperature=0,
            api_key=openai_api_key,
            model="gpt-3.5-turbo-0613"
        )
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        
    def load_weather_page(self, location: str) -> List[str]:
        """Load weather data from a weather website for a given location."""
        url = f"https://weather.com/weather/tenday/l/{location}"
        loader = AsyncChromiumLoader([url])
        html = loader.load()
        
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(html)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs_transformed)
        
        return splits

    def extract_weather_data(self, text_chunks: List[str]) -> List[Dict]:
        """Extract structured weather data from text using LLM."""
        functions = [
            {
                "name": "extract_weather",
                "description": "Extract weather information from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "The date of the weather forecast"
                        },
                        "temperature_high": {
                            "type": "integer",
                            "description": "The high temperature in Fahrenheit"
                        },
                        "temperature_low": {
                            "type": "integer",
                            "description": "The low temperature in Fahrenheit"
                        },
                        "conditions": {
                            "type": "string",
                            "description": "Weather conditions description"
                        },
                        "precipitation_chance": {
                            "type": "integer",
                            "description": "Chance of precipitation as a percentage"
                        },
                        "humidity": {
                            "type": "integer",
                            "description": "Humidity percentage"
                        },
                        "wind_speed": {
                            "type": "string",
                            "description": "Wind speed description"
                        }
                    },
                    "required": ["date", "temperature_high", "temperature_low", "conditions"]
                }
            }
        ]

        all_weather_data = []
        for chunk in text_chunks:
            try:
                messages = [
                    {"role": "system", "content": "Extract weather information from the following text. Return only the structured data."},
                    {"role": "user", "content": chunk.page_content}
                ]
                
                response = self.llm.predict_messages(
                    messages,
                    functions=functions,
                    function_call={"name": "extract_weather"}
                )
                
                if response.additional_kwargs.get('function_call'):
                    extracted_data = json.loads(response.additional_kwargs['function_call']['arguments'])
                    all_weather_data.append(extracted_data)
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
        
        return all_weather_data

    def create_vector_store(self, weather_data: List[Dict]) -> FAISS:
        """Create a vector store from the weather data for RAG."""
        weather_texts = [
            f"Date: {data['date']}\n"
            f"High: {data['temperature_high']}°F\n"
            f"Low: {data['temperature_low']}°F\n"
            f"Conditions: {data['conditions']}\n"
            f"Precipitation: {data.get('precipitation_chance', 'N/A')}%\n"
            f"Humidity: {data.get('humidity', 'N/A')}%\n"
            f"Wind: {data.get('wind_speed', 'N/A')}"
            for data in weather_data
        ]
        
        vector_store = FAISS.from_texts(
            weather_texts,
            self.embeddings
        )
        
        return vector_store

    def process_location(self, location: str) -> Dict:
        """Process weather data for a given location."""
        try:
            text_chunks = self.load_weather_page(location)
            weather_data = self.extract_weather_data(text_chunks)
            
            vector_store = self.create_vector_store(weather_data)
            
            return {
                "raw_data": weather_data,
                "vector_store": vector_store
            }
        except Exception as e:
            print(f"Error processing location {location}: {str(e)}")
            return None

def main():
    # Example usage
    openai_api_key = "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"
    if not openai_api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
        
    extractor = WeatherDataExtractor(openai_api_key)
    
    location = "New-York-NY"
    
    result = extractor.process_location(location)
    if result:
        print("Weather data extracted successfully!")
        print("\nSample weather data:")
        print(json.dumps(result["raw_data"][:2], indent=2))
        
        # Example of using the vector store for RAG
        vector_store = result["vector_store"]
        query = "What's the weather like tomorrow?"
        similar_docs = vector_store.similarity_search(query, k=1)
        print("\nRAG Query Result:")
        print(similar_docs[0].page_content)

if __name__ == "__main__":
    main()
