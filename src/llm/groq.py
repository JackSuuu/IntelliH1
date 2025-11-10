import json
from groq import Groq
from httpx import HTTPStatusError
from config import API_KEY

client = Groq(api_key=API_KEY)

def interpret_navigation_command(prompt: str) -> dict:
    """
    Interprets a natural language navigation command to determine the target location.

    Args:
        prompt (str): The natural language command (e.g., "go to the kitchen").

    Returns:
        dict: A dictionary containing 'destination' (str) and 'confidence' (float).
              Returns None if no valid destination is found.
    """
    system_message = """
    You are a navigation command interpreter for a robot. Your task is to extract the destination 
    location from natural language commands.
    
    Available locations:
    - kitchen
    - bedroom
    - living room
    
    Return ONLY a JSON object with:
    - "destination": the location name (kitchen, bedroom, or living room) in lowercase
    - "confidence": a value between 0.0 and 1.0 indicating your confidence
    - "understood_command": a brief description of what the user wants
    
    If no valid destination is found, return {"destination": null, "confidence": 0.0, "understood_command": "unclear command"}
    
    Examples:
    User: "go to the kitchen"
    Assistant: {"destination": "kitchen", "confidence": 1.0, "understood_command": "navigate to kitchen"}
    
    User: "take me to bedroom"
    Assistant: {"destination": "bedroom", "confidence": 1.0, "understood_command": "navigate to bedroom"}
    
    User: "I want to go to the living room"
    Assistant: {"destination": "living room", "confidence": 1.0, "understood_command": "navigate to living room"}
    
    User: "stop"
    Assistant: {"destination": null, "confidence": 1.0, "understood_command": "stop movement"}
    """
    
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        response_content = completion.choices[0].message.content
        if response_content is None:
            raise ValueError("LLM returned empty response.")
            
        result = json.loads(response_content)
        return result

    except HTTPStatusError as e:
        print(f"Error calling Groq API: {e}")
        return {"destination": None, "confidence": 0.0, "understood_command": "API error"}
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error parsing LLM response: {e}")
        return {"destination": None, "confidence": 0.0, "understood_command": "parsing error"}


def get_llm_guidance(prompt: str) -> dict:
    """
    Gets navigation guidance from the Groq LLM.

    Args:
        prompt (str): The natural language command.

    Returns:
        dict: A dictionary containing the target linear and angular velocities.
    """
    system_message = """
    You are an expert in robotics motion control. Your task is to translate a natural language command 
    into a JSON object with target linear and angular velocities for a differential drive robot.
    The JSON object must be the only thing in your response.
    
    - `linear_velocity`: A value between -1.0 (full reverse) and 1.0 (full forward).
    - `angular_velocity`: A value between -1.0 (full right turn) and 1.0 (full left turn).

    Example:
    User: "Go forward at half speed and turn slightly right."
    Assistant: {"linear_velocity": 0.5, "angular_velocity": -0.3}
    """
    
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        response_content = completion.choices[0].message.content
        if response_content is None:
            raise ValueError("LLM returned empty response.")
            
        guidance = json.loads(response_content)

        # Validate the response
        if 'linear_velocity' in guidance and 'angular_velocity' in guidance:
            return guidance
        else:
            print("Error: Invalid JSON structure from LLM.")
            return {"linear_velocity": 0.0, "angular_velocity": 0.0}

    except HTTPStatusError as e:
        print(f"Error calling Groq API: {e}")
        return {"linear_velocity": 0.0, "angular_velocity": 0.0}
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error parsing LLM response: {e}")
        return {"linear_velocity": 0.0, "angular_velocity": 0.0}

if __name__ == '__main__':
    # Example usage
    command = "Drive forward and turn left."
    guidance = get_llm_guidance(command)
    print(f"Command: '{command}' -> Guidance: {guidance}")
