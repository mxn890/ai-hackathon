from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.schemas import ChatMessage, ChatResponse
from services.gemini_service import GeminiService
import uvicorn
import traceback
from dotenv import load_dotenv

# Load .env
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

# Initialize Gemini service
gemini_service = GeminiService()

@app.get("/")
async def root():
    return {"message": "Physical AI Chatbot API is running!"}

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Handle chat requests"""
    try:
        print(f"📩 Received message: {message.message[:50]}...")
        
        # Use selected text as context if provided
        if message.selected_text:
            context = message.selected_text
            sources = [{"text": message.selected_text, "source": "Selected Text"}]
        else:
            context = (
                "This is a textbook about Physical AI and Humanoid Robotics. "
                "Topics include ROS 2, robot simulation, NVIDIA Isaac, "
                "computer vision, machine learning, and vision-language-action systems."
            )
            sources = [{"text": "General Physical AI knowledge", "source": "Textbook", "score": 0.0}]
        
        # Generate response using Gemini
        try:
            response = gemini_service.generate_response(message.message, context)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"AI response error: {str(e)}")
        
        return ChatResponse(response=response, sources=sources)
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/personalize")
async def personalize_content(request: dict):
    """Personalize content based on user background"""
    try:
        content = request.get("content", "")
        user_background = request.get("userBackground", {})
        
        experience_level = user_background.get("experienceLevel", "beginner")
        software_bg = user_background.get("softwareBackground", "")
        hardware_bg = user_background.get("hardwareBackground", "")
        
        prompt = f"""You are an educational content personalizer. Adjust the following technical content for a student with this background:

Experience Level: {experience_level}
Software Background: {software_bg}
Hardware Background: {hardware_bg}

Original Content:
{content}

Instructions:
- If beginner: Simplify technical terms, add more explanations, use analogies
- If intermediate: Balance theory and practice, assume basic programming knowledge
- If advanced: Add advanced topics, reduce basic explanations, include optimization tips
- Adjust code examples complexity based on software background
- Reference hardware they know when explaining concepts

Provide the personalized version of the content maintaining the same structure and format."""
        
        response = gemini_service.model.generate_content(prompt)
        return {"personalizedContent": response.text}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
async def translate_content(request: dict):
    """Translate content to Urdu or back to English"""
    try:
        content = request.get("content", "")
        target_language = request.get("targetLanguage", "urdu")
        
        if target_language == "urdu":
            prompt = f"""Translate the following technical content to Urdu (اردو). 
Keep technical terms in English but explain them in Urdu.
Maintain markdown formatting.

Content to translate:
{content}

Provide the Urdu translation:"""
        else:
            prompt = f"""Translate the following Urdu technical content back to English.
Maintain markdown formatting.

Content to translate:
{content}

Provide the English translation:"""
        
        response = gemini_service.model.generate_content(prompt)
        return {"translatedContent": response.text}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Export for Vercel
handler = app

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
