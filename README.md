# Arogya Vidya™ Chatbot

A Flask-based RAG (Retrieval-Augmented Generation) chatbot for Dr. Vidya Kollu's medical consultation service, powered by Google's Gemini 2.5 Flash model.

## Features

- **RAG Engine**: Uses FAISS vector store for efficient document retrieval
- **Gemini 2.5 Flash**: Powered by Google's latest language model
- **Medical Knowledge Base**: Contains information about Dr. Kollu's services
- **Flask API**: RESTful API endpoint for chat interactions
- **CORS Enabled**: Ready for frontend integration

## Setup

1. Clone the repository:
```bash
git clone https://github.com/JohnJoel4/ArogyaVidyaChatbot.git
cd ArogyaVidyaChatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
GOOGLE_API_KEY=your_google_api_key_here
```

4. Run the application:
```bash
python app.py
```

## API Usage

Send POST requests to `/api/chat`:

```json
{
  "message": "What services does Dr. Kollu offer?"
}
```

Response:
```json
{
  "reply": "Dr. Vidya Kollu offers various medical consultation services..."
}
```

## Deployment

This application is ready for deployment on Render:

- **Runtime**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`
- **Environment**: Add your `GOOGLE_API_KEY` as a secret

## Technology Stack

- **Backend**: Flask, Flask-CORS
- **AI/ML**: Google Generative AI (Gemini 2.5 Flash)
- **Vector Store**: FAISS
- **Deployment**: Gunicorn (production server)