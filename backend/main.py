from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.schemas import ChatMessage, ChatResponse
from services.gemini_service import GeminiService
import uvicorn
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Physical AI Chatbot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini only (NO DATABASE)
gemini_service = GeminiService()


@app.on_event("startup")
async def startup_event():
    print("🚀 Server starting...")
    print("✅ Gemini chatbot ready (Database disabled)")


@app.get("/")
async def root():
    return {"message": "Physical AI Chatbot API is running (No Database)"}


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        print(f"📩 Message received: {message.message[:50]}...")

        # Context handling
        if message.selected_text:
            context = message.selected_text
            sources = [
                {"text": message.selected_text, "source": "Selected Text"}
            ]
        else:
            context = (
                "This is a textbook about Physical AI and Humanoid Robotics. "
                "Topics include ROS 2, robot simulation, NVIDIA Isaac, "
                "computer vision, machine learning, and vision-language-action systems."
            )
            sources = [
                {
                    "text": "General Physical AI knowledge",
                    "source": "Textbook",
                    "score": 0.0
                }
            ]

        # Generate response
        try:
            response = gemini_service.generate_response(
                message.message,
                context
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Gemini error: {str(e)}"
            )

        return ChatResponse(
            response=response,
            sources=sources
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


@app.post("/personalize")
async def personalize_content(request: dict):
    try:
        content = request.get("content", "")
        user_background = request.get("userBackground", {})

        experience_level = user_background.get("experienceLevel", "beginner")
        software_bg = user_background.get("softwareBackground", "")
        hardware_bg = user_background.get("hardwareBackground", "")

        prompt = f"""
You are an educational content personalizer.

Experience Level: {experience_level}
Software Background: {software_bg}
Hardware Background: {hardware_bg}

Content:
{content}

Rules:
- Beginner: simple language, analogies
- Intermediate: balanced explanation
- Advanced: concise, advanced insights
"""

        response = gemini_service.model.generate_content(prompt)
        return {"personalizedContent": response.text}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate")
async def translate_content(request: dict):
    try:
        content = request.get("content", "")
        target_language = request.get("targetLanguage", "urdu")

        if target_language == "urdu":
            prompt = f"""
Translate to Urdu.
Keep technical terms in English.
Maintain formatting.

{content}
"""
        else:
            prompt = f"""
Translate to English.
Maintain formatting.

{content}
"""

        response = gemini_service.model.generate_content(prompt)
        return {"translatedContent": response.text}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# For Vercel
handler = app

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
