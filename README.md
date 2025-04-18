SmartNote AI – Personal Knowledge Q&A System

SmartNote AI is an intelligent, personalized knowledge assistant designed to help students, researchers, and learners efficiently understand and query their own documents.

🚀 Features
	•	📄 Upload PDFs, TXT, or Markdown documents
	•	❓ Ask questions about your documents — get answers instantly
	•	🗣️ Voice input (via WebRTC or HTML-based recorder)
	•	🌐 Knowledge graph preview (for document insights)
	•	🔍 Supports GPT-4 or custom models
	•	🧠 Works with vector database (e.g. FAISS) for semantic search
	•	🧪 Built with Streamlit for easy deployment

📦 Tech Stack
	•	Python 3.12
	•	OpenAI API (Whisper + GPT)
	•	Streamlit
	•	FAISS (or Chroma-compatible backends)
	•	HTML5 / JavaScript (for local audio recording)
	•	Requests, NumPy, SoundFile (optional)

🛠️ Getting Started
	1.	Clone the repo:

git clone https://github.com/q1q1-spefic/smartnote-ai.git
cd smartnote-ai

	2.	Set up your environment:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

	3.	Add your OpenAI API key:

You can set it as an environment variable or input it via the app UI.
	4.	Run the app:

streamlit run app.py



⸻

📚 Example Use Cases
	•	🧑‍🎓 Study assistant for summarizing textbook PDFs
	•	👩‍🔬 Research helper for analyzing papers and extracting answers
	•	🧑‍🏫 Teachers can upload course material and generate Q&A
	•	🌍 Multi-lingual support (if content is multilingual)

⸻

🔒 Privacy & Security

All document processing runs locally unless hosted via Streamlit Cloud. No data is uploaded to third-party servers except OpenAI’s API (for answering questions).

⸻

💬 Contact

Developed by Lucian Liu
Open to collaboration or feedback: lucian3@uci.edu or liu.lucian6@gmail.com

