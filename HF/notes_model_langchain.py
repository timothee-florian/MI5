from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Any
import torch

class QwenLLM(LLM):
    model_path: str = "./qwen_model"
    model: Any = None
    tokenizer: Any = None
    max_tokens: int = 512
    temperature: float = 0.7
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto"
        )
    
    @property
    def _llm_type(self) -> str:
        return "qwen"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
        
        # Handle stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in response:
                    response = response.split(stop_seq)[0]
        
        return response.strip()
    
######################################################################################
######################################################################################

    ## Langchain agent tool
######################################################################################
######################################################################################

from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# Define your custom functions
def calculate_sum(numbers: str) -> str:
    """Calculate the sum of comma-separated numbers."""
    try:
        nums = [float(x.strip()) for x in numbers.split(',')]
        result = sum(nums)
        return f"The sum is {result}"
    except:
        return "Error: Please provide comma-separated numbers"

def get_weather(location: str) -> str:
    """Get weather information for a location."""
    # This is a dummy function - replace with real API call
    return f"The weather in {location} is sunny, 25°C"

def search_database(query: str) -> str:
    """Search the database for information."""
    # Replace with your actual database search
    return f"Found 3 results for '{query}'"

# Create LangChain tools
tools = [
    Tool(
        name="Calculator",
        func=calculate_sum,
        description="Useful for calculating the sum of numbers. Input should be comma-separated numbers like '1,2,3'"
    ),
    Tool(
        name="Weather",
        func=get_weather,
        description="Useful for getting weather information. Input should be a city name."
    ),
    Tool(
        name="DatabaseSearch",
        func=search_database,
        description="Useful for searching the database. Input should be a search query."
    )
]


######################################################################################
######################################################################################

    ## Langchain Agent init and test
######################################################################################
######################################################################################
# Initialize the Qwen LLM
llm = QwenLLM(model_path="./qwen_model", temperature=0.7, max_tokens=256)

# Create the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate"
)

# Use the agent
response = agent.run("What is the sum of 10, 20, and 30?")
print(response)

response = agent.run("What's the weather in Paris?")
print(response)

response = agent.run("Search the database for python tutorials")
print(response)