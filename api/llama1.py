import requests
import json

def get_response():
    url = "http://localhost:8000/generate" 
    # headers = {"Content-Type": "application/json"}

    # Send the request to the Llama API
    response = requests.post(url, json={"prompt": "your prompt here"})
    
    if response.status_code == 200:
        # Print the raw response to check its structure
        print("Raw response:", response.text)

        try:
            # Attempt to parse the response and extract the expected data
            response_data = response.text
            if '"response": "' in response_data:
                # Only proceed if the expected string is in the response
                full_text = response_data.split('"response": "')[1]
                full_text = full_text.split('"}')[0]  # Assuming the response ends with a closing quote and brace
                return full_text
            else:
                print("Unexpected response format:", response_data)
                return "Error: Response format not as expected"
        except Exception as e:
            print("Error parsing response:", str(e))
            return "Error: Could not parse response"
    else:
        print(f"Error: API request failed with status code {response.status_code}")
        return "Error: API request failed"

# Test the function
result = get_response()
print("Result:", result)