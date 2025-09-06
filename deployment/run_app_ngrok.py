import subprocess
import time
import os

STREAMLIT_APP = r"D:\sprints\Heart_Disease_Project\deployment\app.py"

NGROK_PATH = r"D:\path\to\ngrok\ngrok.exe"  

print("Starting Streamlit app...")
streamlit_process = subprocess.Popen(["streamlit", "run", STREAMLIT_APP])

time.sleep(5)

print("Starting Ngrok tunnel...")
ngrok_process = subprocess.Popen([NGROK_PATH, "http", "8501"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("Streamlit is running locally at http://localhost:8501")
print("Ngrok public URL will appear in the Ngrok terminal output")

try:
    streamlit_process.wait()
    ngrok_process.wait()
except KeyboardInterrupt:
    print("Stopping Streamlit and Ngrok...")
    streamlit_process.terminate()
    ngrok_process.terminate()
