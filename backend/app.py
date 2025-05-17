from fastapi import FastAPI, Depends, HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
import joblib
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime


from utils.facebook_api import FacebookAPI
from utils.preprocessing import TextPreprocessor
from utils.auth import(
    authenticate_user,
    create_access_token,
    get_current_active_user,
    User,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

app = FastAPI()
limiter =Limiter(key_func=get_remote_address)
app.state.limiter=limiter
app.add_exception_handler(RateLimitedExceeded,_rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#monitoring
Instrumentator().instrument(app).expose(app)


#initializing  components
try:
    model=joblib.load("models/sentiment_model.pkl")
    preprocessor=TextPreprocessor()
    fb_api=FacebookAPI()
except Exception as e:
    raise RuntimeError(f"Failed to load model or components: {str(e)}")




#auth endpoints

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user=authenticate_user(form_data.username,form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    
    access_token_expires=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token=create_access_token(
        data={"sub":user.username},
        expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


#main endpoints
@app.post("/analyze")
@limiter.limit("5/minute")
async def analyze_comments(
    request:Request,
    page_id:str,
    current_user: User = Depends(get_current_active_user)
)->Dict[str,Any]:
    try:
        comments=fb_api.get_all_comments(page_id)
        df=pd.DataFrame(comments)
        df['cleaned_text']=df['message'].apply(preprocessor.clean_text)
        df['cleaned_text']=df['message'].apply(preprocessor.cleaned_text)

        return {
            "status": "success",
            "results":df[['message','sentiment']].to_dict('records'),
            "stats":df['sentiment'].value_counts().to_dict()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )
@app.get("health")
async def health_check():
    return {"status": "healthy"}