import os
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain



def get_llm():
        
    model_kwargs = { #AI21
        "maxTokens": 1024, 
        "temperature": 0, 
        "topP": 0.5, 
        "stopSequences": ["Human:"], 
        "countPenalty": {"scale": 0 }, 
        "presencePenalty": {"scale": 0 }, 
        "frequencyPenalty": {"scale": 0 } 
    }
    
    llm = Bedrock(
        credentials_profile_name=os.environ.get("Braeden"), #sets the profile name to use for AWS credentials (if not the default)
        region_name=os.environ.get("us-west-2"), #sets the region name (if not the default)
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
        model_id="ai21.j2-ultra-v1", #set the foundation model
        model_kwargs=model_kwargs) #configure the properties for Claude
    
    return llm

# AWS_DEFAULT_REGION="us-west-2"
# AWS_ACCESS_KEY_ID="ASIA6PBIVHVPP4CR2G52"
# AWS_SECRET_ACCESS_KEY="P3EC2QCejqMNjrV/lHIcXOwiUoW8PbzJ61RLW4WK"
# AWS_SESSION_TOKEN="IQoJb3JpZ2luX2VjEFMaCXVzLWVhc3QtMSJHMEUCICU8g0hjJnD8jzYUJRZFmv+bwKrpQKgkeIAuNo79GolqAiEAo9wiyNINpMfxWNrXI8IJvExOV/3ZBDCQJlB1HOK97dYqmQIITBABGgw5OTQzNzAyMDcwNzAiDDCXPujbreH6AkFKmSr2AVx8JdC33MHfnS6wAY2Dr8TtiFbqveMo7o1r08uHm6g+XKW5ruhA12prlgo4GTHeWROqQWwAfKIRash8PegAeJA/eC82fa/0NEl0UDW8k2vJln/wh9hHRHIe1Rrh0+dsAfa9QBvcDuT3uSDhsyezdi8Bne4TEvPT4RdwEm53gmXl1Ile0tbuxgOPMCyzhAvkz1+zrmVZjL0fl7+otJp/nngN/9ZpieuretTko6rMaIL/0Dn57TZ092j2aK/MIHwQ7rX6I4WurMtq0pwLnN+W1EfKPsRE+Nv2Eof6J+yQpSjFjtwA1H9QBICShr17u4Zp6H1PlGBUlDD8542vBjqdAQjk3rbFFsxqn2roaSsErYUTPy6XJcOvQl/3JZNokwZbeCczQygUpJUnqzQTsVHRwKqEjGqTDF/shSUVW/pqz4DsA/4qBNNTDt8XBV/yd57hCSzfJdeQeXdmT/amSEFG6lO1b6pZwsDsN7Pf6r1Hr7K8H+5oGhwvRkSxXA1/fu3J9pyMwy8J+97YVn7Ef7MorTLj2T5Q0qWmkH6/Ots="


def get_memory(): #create memory for this chat session
    
    #ConversationSummaryBufferMemory requires an LLM for summarizing older messages
    #this allows us to maintain the "big picture" of a long-running conversation
    llm = get_llm()
    
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1024) #Maintains a summary of previous messages
    
    return memory


def get_chat_response(input_text, memory): #chat client function
    
    llm = get_llm()
    
    conversation_with_summary = ConversationChain( #create a chat client
        llm = llm, #using the Bedrock LLM
        memory = memory, #with the summarization memory
        verbose = True #print out some of the internal states of the chain while running
    )
    
    chat_response = conversation_with_summary.predict(input=input_text) #pass the user message and summary to the model
    
    return chat_response
