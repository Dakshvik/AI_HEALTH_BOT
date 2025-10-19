🤖 AI Health Bot

A smart, voice-enabled health assistant built with Streamlit and Hugging Face. This bot provides factual information about diseases, symptoms, and prevention using a safe, retrieval-augmented AI model to ensure accuracy and avoid medical misinformation.

✨ Features

Clean Chat Interface: A user-friendly and responsive chat window built with Streamlit.

Voice-Enabled Communication:

Speech-to-Text: Ask questions using your voice 🎙️ (powered by speech_recognition).

Text-to-Speech: Hear the assistant's replies spoken aloud for hands-free interaction (powered by gTTS).

Fact-Grounded AI Model:

NLU: Uses a Hugging Face LLM (meta-llama/Llama-4-Scout) to understand user intent (e.g., "symptoms") and entities (e.g., "influenza").

Factual Retrieval: Pulls answers from a secure, local JSON knowledge base to prevent the AI from "hallucinating" or making up medical facts.

Conversational Generation: Uses the LLM a second time to transform the factual data into a natural, supportive, and easy-to-understand response.

Robust Entity Resolution: Intelligently understands aliases and common names (e.g., "flu" -> "influenza", "covid" -> "covid-19").

🛠️ Tech Stack

Framework: Streamlit

AI/LLM: Hugging Face InferenceClient

Speech-to-Text: speech_recognition

Text-to-Speech: gTTS (Google Text-to-Speech)

Language: Python 3.9+

🚀 How It Works

This project uses a Retrieval-Augmented Generation (RAG) workflow to ensure that its answers are both accurate and natural.

Input (Voice/Text): A user asks, "How can I prevent the flu?"

NLU (Understand): The query is sent to a Hugging Face LLM, which identifies the structured request: {"entity": "influenza", "intent": "prevention"}.

Retrieval (Find): The app looks inside its local dataset.json file for the influenza entry and pulls the data from the prevention key. This step is fast and grounded in facts you provide.

Generation (Beautify): The raw data (e.g., a Python list ["Wash hands", "get vaccine"]) is sent back to the LLM to be rewritten as a friendly, full-sentence answer.

Output (Voice/Text): The bot displays and speaks the final, helpful response.

⚙️ Getting Started

To run this project locally, follow these steps:

1. Prerequisites

Python 3.9 or higher

A Hugging Face API Token with access to the meta-llama/Llama-4-Scout model.

2. Installation & Setup

Clone the repository:

git clone [https://github.com/Dakshvik/AI_HEALTH_BOT.git](https://github.com/Dakshvik/AI_HEALTH_BOT.git)
cd AI_HEALTH_BOT



Install the required dependencies:

pip install -r requirements.txt



(Note: You will need to create a requirements.txt file containing streamlit, streamlit-chat, speechrecognition, gtts, etc.)

Create your Knowledge Base:

Make sure you have a dataset.json or diseases.json file in the main folder. It should be structured like this:

{
  "influenza": {
    "symptoms": ["fever", "cough", "sore throat", "body aches"],
    "prevention": ["Get a flu vaccine each year", "Wash your hands often"],
    "myths": ["The flu vaccine can give you the flu."]
  },
  "covid-19": {
    "symptoms": ["fever", "cough", "loss of taste or smell"],
    "prevention": ["Get vaccinated", "Wear a mask in crowded areas"]
  }
}



Set up your Streamlit Secrets:

In your project folder, create a new directory: .streamlit

Inside that folder, create a new file: secrets.toml

Add your Hugging Face API token to the secrets file:

HF_API_TOKEN = "hf_YourHuggingFaceApiTokenHere"



3. Run the Application

Once everything is set up, run the following command in your terminal:

streamlit run app.py



🗺️ Future Roadmap

This project has a strong foundation with significant potential for growth. Here are the key features planned for the future:

[ ] Implement Conversational Memory: Enable the chatbot to remember the context of the conversation, so users don't have to repeat the disease or topic in follow-up questions.

[ ] Build an Agentic AI Framework:

[ ] Hospital & Clinic Locator: Integrate with mapping services to find nearby healthcare facilities based on the user's location.

[ ] Appointment Booking: Develop an AI agent that can interact with hospital booking systems (via APIs) to schedule appointments for patients directly through the chat.

[ ] Optimize AI Pipeline: Consolidate the two LLM calls into a single, more efficient RAG prompt to improve response speed and reduce API costs.

[ ] Expand Knowledge Base: Transition from a static JSON file to a scalable vector database (e.g., ChromaDB, FAISS) to allow for semantic search across a much larger dataset of medical information.

🤝 Contributing

Contributions are welcome! If you have an idea for a new feature or an improvement, please feel free to fork the repo and create a pull request.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📧 Contact

Dakshvik - Project Link
