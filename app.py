from flask import Flask, render_template, request
import json
import os
from pathlib import Path
import fitz  # PyMuPDF
from gpt4all import GPT4All

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Model setup
model_name = "mistral-7b-instruct-v0.1.Q4_0.gguf"
model_path = Path("models").resolve()
gpt = GPT4All(model_name, model_path=str(model_path), allow_download=False)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize memory
if not os.path.exists('memory.json'):
    with open('memory.json', 'w') as f:
        json.dump([], f)

def load_memory():
    with open('memory.json') as f:
        return json.load(f)

def save_memory(memory):
    with open('memory.json', 'w') as f:
        json.dump(memory, f, indent=4)

def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def generate_response(user_input, memory, pdf_text=""):
    prompt = (
        f"You are BST GPT, a highly personalized AI assistant for Tharun B S.\n\n"
        f"Your previous conversations:\n{json.dumps(memory[-5:], indent=2)}\n\n"
        f"User context:\n"
        f"- Tharun is an EEE student\n"
        f"- Interested in EVs, ML, Embedded, Web Dev\n"
        f"- Wants career guidance and project help\n\n"
        f"PDF Context (if any):\n{pdf_text[:1000]}\n\n"
        f"User: {user_input}\n"
        f"BST GPT:"
    )
    with gpt.chat_session():
        output = gpt.generate(prompt, max_tokens=3000)
    return output.strip()
    with gpt.chat_session():
        output = gpt.generate(prompt, max_tokens=3000)
    return output.strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    memory = load_memory()
    pdf_text = ""
    reply = ""
    if request.method == 'POST':
        query = request.form['query']
        pdf = request.files.get('pdf')

        if pdf:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], pdf.filename)
            pdf.save(filepath)
            pdf_text = extract_text_from_pdf(filepath)

        reply = generate_response(query, memory, pdf_text)
        memory.append({'user': query, 'bot': reply})
        save_memory(memory)

    return render_template('index.html', memory=memory, reply=reply)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
