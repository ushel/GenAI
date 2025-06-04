import requests
import json


# Define the API endpoint and model
ollama_url = "http://localhost:11434/api/generate"
model_name = "llama3.2"  # Make sure this is the exact model name in your Ollama installation

# # Define the prompt you want to send to the model
# prompt_text = "Write a short poem about the beauty of a starlit sky."

# # Create the request payload
# payload = {
#     "model": model_name,
#     "prompt": prompt_text
# }

# # Send the POST request to Ollama API
# response = requests.post(ollama_url, json=payload)

# Check if the request was successful
# if response.status_code == 200:
#     # Print the raw response text to inspect it
#     print("Raw Response Text:\n", response.text)

#     # Try to parse the response as JSON if it seems like JSON
#     try:
#         response_data = response.json()
#         print("\nGenerated Text:\n", response_data.get('response', 'No response found.'))
#     except ValueError as e:
#         print("\nError parsing JSON:", e)
# else:
#     print(f"Error: {response.status_code}")
#     print(response.text)
    

# # url = "YOUR_API_URL"
# headers = {"Authorization": "Bearer YOUR_API_KEY"}

# response = requests.get(ollama_url , stream=True)

# # Read and process each JSON object in the stream
# for line in response.iter_lines():
#     if line:
#         try:
#             json_data = json.loads(line.decode("utf-8"))
#             print(json_data)
#         except json.JSONDecodeError as e:
#             print(f"Error decoding JSON: {e}")

def get_response():
    url = "http://localhost:8000/generate"  # Replace with your API endpoint
    payload = {
        "prompt": "Write me a poem about the stars.",  # Example prompt
        "model": "llama3.2",
    }
    response = requests.post(url, json=payload, stream=True)

    full_text = ""
    for line in response.iter_lines():
        if line:
            response_data = line.decode('utf-8')
            if '"done": true' in response_data:  # Check for the completion flag
                full_text += response_data.split('"response": "')[1].split('"done": true')[0]
                break
            else:
                full_text += response_data.split('"response": "')[1]
                
    return full_text

result = get_response()
print(result)