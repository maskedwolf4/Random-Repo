import os
import uuid
import re
import requests
import boto3
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from langsmith import traceable, Client
from langsmith.run_helpers import get_current_run_tree
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextvars

# --- CONFIGURATION ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# Use default or env provided project
os.environ["LANGCHAIN_PROJECT"] = os.getenv('LANGSMITH_PROJECT', 'photographer_agent')
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")

# AWS S3
S3_CLIENT = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

PRESET_AVATARS = [
    #s3 URLs
]

# Session State
# { session_id: { 
#     "stage": "AVATAR_SELECTION", # AVATAR_SELECTION, APPAREL_UPLOAD, SCENE_SELECTION, GENERATION
#     "avatars": [], 
#     "apparel": [], 
#     "generated": [], 
#     "history": [] 
#   } 
# }
SESSIONS: Dict[str, Dict] = {}

# --- HELPER FUNCTIONS ---

def upload_to_s3(file_content: bytes, folder_type: str, extension: str = "png") -> str:
    """
    folder_type: 'user_uploaded' or 'ai_generated'
    """
    file_name = f"{uuid.uuid4().hex}.{extension}"
    # Structure: photographer_agent/{folder}/{filename}
    key = f"photographer_agent/{folder_type}/{file_name}"
    
    S3_CLIENT.put_object(
        Bucket=BUCKET_NAME, 
        Key=key, 
        Body=file_content, 
        ContentType=f"image/{extension}"
    )
    return f"https://{BUCKET_NAME}.s3.amazonaws.com/{key}"

@traceable(run_type="tool", name="AIML_Image_Generation_Call")
def call_image_api(prompt: str, image_urls: List[str], angle: str = ""):
    api_key = os.environ.get("AIML_API_KEY")
    url = "https://api.aimlapi.com/v1/images/generations"
    
    # LangSmith Metadata
    rt = get_current_run_tree()
    if rt:
        rt.add_metadata({ "model": "google/nano-banana-pro-edit", "angle": angle })

    identity_guard = "Maintain exact facial features of the avatars provided. Keep the apparels identical to the reference photos."
    scene_guard = "Photorealistic, high-end fashion photography, 85mm lens, cinematic lighting from different angles -Front -Back -Side Profile -In Motion Walking Shot."
    structure_guard = "Single image, no collage, no grid, no split screen, full frame."
    
    # Inject angle into prompt
    final_prompt = f"{angle} shot. {structure_guard} {identity_guard} {scene_guard} {prompt}"
    
    payload = {
        "model": "google/nano-banana-pro-edit",
        "prompt": final_prompt,
        "image_urls": image_urls,
        "num_images": 1
    }
    
    response = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=payload)
    if response.status_code != 200:
        raise Exception(f"Image API Error: {response.text}")
        
    return response.json()

class ImageGenerationTool(BaseTool):
    name: str = "image_generation_tool"
    description: str = "Generates high-fashion images. Input: prompt (str), image_urls (list)."

    def _run(self, prompt: str, image_urls: List[str]) -> str:
        angles = ["Front View", "Back View", "Side Profile", "In Motion Walking Shot","Close Up"]
        all_urls = []
        errors = []

        # Parallelize generation
        with ThreadPoolExecutor(max_workers=5) as executor:
            ctx = contextvars.copy_context()
            
            # Helper to run in context
            def run_in_context(p, u, a):
                return ctx.run(call_image_api, p, u, a)

            future_to_angle = {
                executor.submit(run_in_context, prompt, image_urls, angle): angle 
                for angle in angles
            }
            
            for future in as_completed(future_to_angle):
                angle = future_to_angle[future]
                try:
                    result = future.result()
                    urls = [item.get('url') for item in result.get('data', [])]
                    all_urls.extend(urls)
                except Exception as e:
                    errors.append(f"{angle}: {str(e)}")
        
        if not all_urls and errors:
             return f"Errors: {'; '.join(errors)}"

        print(f"DEBUG: Generated {len(all_urls)} images: {all_urls}")
        return f"GENERATED_URLS: {all_urls}"

# --- APP SETUP ---

@traceable(run_type="tool", name="Get_Image_Description")
def analyze_image(image_url: str) -> str:
    """Uses GPT-4o Vision to get a description of the image content (apparel/vibes)."""
    api_key = os.environ.get("AIML_API_KEY")
    url = "https://api.aimlapi.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the apparel, style, and vibe of this image in 1 sentence."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        "max_tokens": 60
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return "Visual analysis unavailable."
    except:
        return "Visual analysis error."

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": "Validation Error"},
    )

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    if os.path.exists("index.html"):
        with open("index.html", "r") as f: return f.read()
    return "Index file not found."

# --- ENDPOINTS ---

@app.post("/select_avatar")
async def select_avatar(
    session_id: str = Form(...),
    avatar_url: str = Form(...)
):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"stage": "AVATAR_SELECTION", "avatars": [], "apparel": [], "generated": [], "history": []}
    
    session = SESSIONS[session_id]
    session["avatars"].append(avatar_url)
    
    # Move to next stage if we have an avatar
    if session["stage"] == "AVATAR_SELECTION":
        session["stage"] = "APPAREL_UPLOAD"
        
    return {
        "status": "success", 
        "message": "Avatar selected. Please upload apparel photos.",
        "stage": session["stage"]
    }

@app.post("/upload")
async def upload_file(
    session_id: str = Form(...),
    type: str = Form(...), # 'avatar' or 'apparel'
    files: List[UploadFile] = File(...)
):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"stage": "AVATAR_SELECTION", "avatars": [], "apparel": [], "generated": [], "history": []}
    
    session = SESSIONS[session_id]
    uploaded_urls = []
    
    for file in files:
        content = await file.read()
        # Upload to user_uploaded folder
        url = upload_to_s3(content, "user_uploaded", extension=file.filename.split('.')[-1])
        uploaded_urls.append(url)
        
        if type == "avatar":
            session["avatars"].append(url)
            if session["stage"] == "AVATAR_SELECTION":
                 session["stage"] = "APPAREL_UPLOAD"
        elif type == "apparel":
            session["apparel"].append(url)
            # If we have avatars and apparel, we are ready for scene selection
            if session["avatars"]:
                session["stage"] = "SCENE_SELECTION"

    message = "Upload successful."
    if session["stage"] == "SCENE_SELECTION":
        message = "Assets received. I can now suggest some scenes. Say 'Suggest scenes' or describe what you want."

    return {
        "status": "success",
        "urls": uploaded_urls,
        "stage": session["stage"],
        "message": message
    }

@app.post("/feedback")
async def submit_feedback(
    session_id: str = Form(...),
    image_url: str = Form(...),
    score: int = Form(...), # 1 for Like, -1 for Dislike
    run_id: str = Form(None) # Optional run_id to attach feedback to
):
    """Logs user feedback to LangSmith."""
    try:
        client = Client()
        
        kwargs = {
            "key": "user_score",
            "score": score,
            "comment": f"Feedback for image: {image_url}",
        }
        if run_id:
            kwargs["run_id"] = run_id
        else:
            kwargs["project_name"] = os.environ["LANGCHAIN_PROJECT"]
            
        client.create_feedback(**kwargs)
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        print(f"Feedback Error: {e}")
        return {"status": "error", "message": str(e)}

@traceable
@app.post("/chat")
async def chat_endpoint(
    session_id: str = Form(...),
    message: str = Form(""),
):
    # Get current run ID for feedback
    rt = get_current_run_tree()
    run_id = str(rt.id) if rt else None
    if session_id not in SESSIONS:
        # Initialize if starting via chat (edge case)
        SESSIONS[session_id] = {"stage": "AVATAR_SELECTION", "avatars": [], "apparel": [], "generated": [], "history": []}
        return {
            "status": "success", 
            "output": f"Welcome! Please select an avatar first.\n\nPreset Avatars:\n" + "\n".join(PRESET_AVATARS)
        }
    
    session = SESSIONS[session_id]
    
    # AGENT SETUP
    aiml_llm = LLM(model="openai/gpt-4o", base_url="https://api.aimlapi.com/v1", api_key=os.getenv("AIML_API_KEY"))
    
    visual_creator = Agent(
        role='Creative Director',
        goal='Guide the user through a fashion shoot workflow.',
        backstory=f"""You are an AI Creative Director.
        Current Stage: {session['stage']}
        
        Assets Available:
        - Avatars: {len(session['avatars'])}
        - Apparel: {len(session['apparel'])}
        - Generated Images: {len(session['generated'])}
        
        WORKFLOW:
        1. AVATAR_SELECTION: Ask user to select or upload an avatar.
        2. APPAREL_UPLOAD: Ask user to upload apparel.
        3. SCENE_SELECTION: Suggest 3 distinct, descriptive creative scenes upto 10 words based on vibe of apparel.
        4. GENERATION: Use 'image_generation_tool'. Use ALL Avatars and Apparel as input. 
           NOTE: The tool will automatically generate Front, Back, Side, and Motion shots for you.
           Just provide the SCENE description in the prompt.
        5. EDITING: If user wants changes, use the 'image_generation_tool' again and for reference take the images generated in the previous chat from URLs, don't make changes unless said. 
           CRITICAL: Verify you include the LAST generated image URL in the tool input for consistency if editing.
        """,
        tools=[ImageGenerationTool()],
        llm=aiml_llm
    )
    
    # Context construction
    # We always include originals + last generated image for continuity
    assets = session["avatars"] + session["apparel"]
    if session["generated"]:
        assets.append(session["generated"][-1])
        
    # VISION ANALYSIS: If we have apparel, analyze the last uploaded item to give context
    visual_context = ""
    if session["apparel"]:
         # Analyze the most recent apparel upload
         desc = analyze_image(session["apparel"][-1])
         visual_context = f"VISUAL ANALYSIS OF APPAREL: {desc}"

    # HISTORY MANAGEMENT
    # Fetch recent history to provide context (so Agent knows what "Option 3" refers to)
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in session["history"][-4:]])
    
    task_desc = f"""
    HISTORY:
    {history_text}
    
    CURRENT USER MESSAGE: {message}
    
    Current Stage: {session['stage']}
    Assets: {assets}
    {visual_context}
    
    INSTRUCTIONS:
    - If suggesting scenes, provide 3 DISTINCT, HIGHLY DETAILED, and VIVID scene descriptions. Paint a picture (lighting, mood, texture).
    - If the user selects an option (e.g., "Option 1" or "The second one"), refer to the HISTORY to find the matching scene description.
    - RESPOND NATURALLY but...
    - CRITICAL: You MUST include ALL image URLs returned by the 'image_generation_tool' in your final response. 
    - FORMATTING REQUIREMENT: At the end of your response, list the URLs as raw text, one per line, like this:
      GENERATED LINKS:
      https://...
      https://...
    - Do not use markdown resizing or hiding. Show the full links.
    """
    
    task = Task(
        description=task_desc,
        expected_output="Helpful response or generated image URLs.",
        agent=visual_creator
    )
    
    crew = Crew(agents=[visual_creator], tasks=[task])
    result = crew.kickoff()
    output_text = result.raw
    
    # Process Output for URLs (AI Generated)
    # Fix: Regex should not capture trailing ')' or ']' or '.' normally found in markdown or sentences
    urls = re.findall(r'https?://[^\s\'"<>\]\)]+', output_text)
    
    new_gen_urls = []
    
    for url in urls:
        # If the URL is from aimlapi (temp), we should ideally re-upload to S3 'ai_generated'.
        # For this prototype, we'll verify if it's new.
        if url not in assets: 
            # It is a new generation. 
            # In a full prod app, we fetch this URL and upload to S3 'ai_generated'
            # Simulating saving to 'ai_generated' logic by just tracking it for now, 
            # as AIML API urls expire.
            
            # TODO: Fetch and Upload to S3 'ai_generated'
            try:
                # Attempt to save to S3 so we have a permanent link
                if "aimlapi" in url or "d144" in url: # basic check for external gen
                    img_data = requests.get(url).content
                    s3_permanent_url = upload_to_s3(img_data, "ai_generated", "png")
                    new_gen_urls.append(s3_permanent_url)
                    # Replace temp url in text with permanent S3 url
                    output_text = output_text.replace(url, s3_permanent_url)
            except:
                print(f"Failed to upload gen image to S3: {url}")
                new_gen_urls.append(url) # Fallback

    if new_gen_urls:
         session["generated"].extend(new_gen_urls)
         session["stage"] = "GENERATION" # or EDITING
    
    # Update History
    session["history"].append({"role": "User", "content": message})
    session["history"].append({"role": "Agent", "content": output_text})

    return {
        "status": "success", 
        "output": output_text,
        "run_id": run_id # Return run_id to client
    }

