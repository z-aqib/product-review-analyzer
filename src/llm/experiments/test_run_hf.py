# src/llm/experiments/test_run_hf.py
from gradio_client import Client

client = Client("MuhammadHaaris/mlops")
print("Client created")
# result = client.predict(user_input="Hello!!", api_name="/predict")
# print("Result:", result)
