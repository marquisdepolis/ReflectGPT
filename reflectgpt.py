import os
import json
import openai
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
from utils.retry import retry_except
# from tenacity import retry, stop_after_attempt, wait_fixed

openai.api_key = os.getenv("OPENAI_API_KEY")
with open('info.json', 'r') as file:
    data = json.load(file)

GPT3 = data.get('GPT_3.5')
GPT4 = data.get('GPT_4')
CLAUDE = data.get('CLAUDE')
global interrupt_token_count
interrupt_token_count = 50  # adjust this as needed

def clean_word(word):
    # Remove extra punctuation and spaces
    cleaned_word = word.strip(",.' ")  
    return cleaned_word

@retry_except(exceptions_to_catch=(IndexError, ZeroDivisionError), tries=3, delay=2)
def llm_call(input, GPT):
    client = OpenAI()
    client.api_key = os.getenv('OPENAI_API_KEY')

    response = client.chat.completions.create(
        model=GPT,
        messages=[
            {"role": "system", "content": """You are a genius AI. You are brilliant and clever."""},
            {"role": "user", "content": f"{input}"}
        ],
        stream=True
    )
    collected_chunks = []
    collected_messages = []
    # iterate through the stream of events
    for chunk in response:
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk.choices[0].delta.content  # extract the message
        collected_messages.append(chunk_message)  # save the message
        processed_list = [clean_word(word) for word in collected_messages]
        sentence = " ".join(processed_list) 
        print(f"Answer so far: {sentence}")
        if len(collected_messages)>= interrupt_token_count:
            question = f"Considering the reply so far to the prompt, {chunk}, see if its a truly great answer. It needs to be perfect. Once you have thought this, give no actual feedback back to the user. Answer only with a single world saying STOP if it's good and we can stop, CONTINUE if we need to keep generating the answer, or RESTART if it's incorrect and you need to restart the answer from the beginning."
            # responsejson = llm_call_json(question, GPT3)
            responsejson = llm_call_groq(question)
            decision = extract_decision(responsejson).strip().upper()
            print(f"\nDecision is: {decision}")
            if "CONTINUE" in decision:
                continue
            elif "STOP" in decision:
                break
            elif "RESTART" in decision:   
                response.response.close()
                generate_answer()
                break

    final_answer = " ".join(collected_messages)  # Compile the final answer 
    print(f"Final answer: {final_answer}")
    try:
        response.response.close()
    except Exception as e:
        print(f"Error closing the stream: {e}")

    return response.choices[0].message.content

@retry_except(exceptions_to_catch=(IndexError, ZeroDivisionError), tries=3, delay=2)
def llm_call_groq(input):
    client = Groq()
    completion = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {
            "role": "user",
            "content": f"Considering the reply so far to the prompt, {input}, see if its a truly great answer. It needs to be perfect. Once you have thought this, give no actual feedback back to the user. Answer only with a single world saying STOP if it's good and we can stop, CONTINUE if we need to keep generating the answer, or RESTART if it's incorrect and you need to restart the answer from the beginning. Reply with a well formatted JSON."
        }
    ],
    temperature=1,
    max_tokens=1024,
    top_p=1,
    stream=False,
    response_format={"type": "json_object"},
    stop=None,
    )
    return (completion.choices[0].message.content)

@retry_except(exceptions_to_catch=(IndexError, ZeroDivisionError), tries=3, delay=2)
def llm_call_json(input, GPT):
    client = OpenAI()
    client.api_key = os.getenv('OPENAI_API_KEY')
    response = client.chat.completions.create(
        model=GPT,
        messages=[
            {"role": "system", "content": """You are a genius AI. You are brilliant and clever."""},
            {"role": "user", "content": f"Respond in JSON. {input}"}
        ],
        response_format={ "type": "json_object" }
    )
    return response.choices[0].message.content

@retry_except(exceptions_to_catch=(IndexError, ZeroDivisionError, ValueError), tries=3, delay=2)
def extract_decision(response):
    """
    Parses the response to determine the decision based on keys and their values.
    """
    try:
        parsed_response = json.loads(response)
        # print(f"\nParsed response is: {parsed_response}\n")
        if isinstance(parsed_response, dict):
            # Iterate through each key-value pair in the dictionary
            for key, value in parsed_response.items():
                # Normalize the key and value to uppercase for comparison
                normalized_key = key.upper().strip()
                normalized_value = value.upper().strip() if isinstance(value, str) else value
                
                # Check if the key is one of the decision keywords
                if normalized_key in ['CONTINUE', 'STOP', 'RESTART']:
                    print(f"Extracted decision based on key: {normalized_key}")
                    return normalized_key.lower()
                
                # Check if the value is one of the decision keywords
                if isinstance(normalized_value, str) and normalized_value in ['CONTINUE', 'STOP', 'RESTART']:
                    # print(f"Extracted decision based on value: {normalized_value}")
                    return normalized_value.lower()
            
            # If no matching key or value is found, raise an exception
            raise ValueError("No decision keywords found in the response.")
        else:
            raise ValueError("The JSON response did not contain a dictionary as expected.")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from response: {e}")
    except ValueError as e:
        print(f"ValueError: {e}")

    return ''

def generate_with_reflection(initial_prompt, interrupt_token_count):
    current_prompt = initial_prompt
    answer = llm_call(current_prompt, GPT4)
    print(f"\n\n Full answer is: {answer} \n\n")
    return answer

def generate_answer():
    try:
        request_data = "Explain the importance of Giant redwoods - the world's largest trees - in the UK and why they outnumber those found in their native range in California." # input("Input the question: ")
        initial_prompt = request_data
        if not initial_prompt:
            return ({"error": "Prompt is required."}), 400
        generated_answer = generate_with_reflection(initial_prompt, interrupt_token_count)
        print(f"Answer: {generated_answer}")
        return (f"Answer: {generated_answer}")
    except Exception as e:
        return ({"error": str(e)}), 500

if __name__ == '__main__':
    generate_answer()