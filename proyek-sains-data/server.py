from pyngrok import ngrok
import threading, os

def run():
    os.system("streamlit run app.py --server.port 8501")

thread = threading.Thread(target=run)
thread.start()

public_url = ngrok.connect(8501)
print("🔗 Akses app kamu di:", public_url)
