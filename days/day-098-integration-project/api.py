"""
Day 98: Integration Project - FastAPI Application
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import tempfile
from analyzer import ContentAnalyzer


# Pydantic models
class TextRequest(BaseModel):
    """Request model for text analysis."""
    text: str
    options: Optional[Dict] = {}


class TextResponse(BaseModel):
    """Response model for text analysis."""
    sentiment: Dict
    entities: Dict
    topics: Dict
    summary: Dict
    metadata: Dict


class ImageResponse(BaseModel):
    """Response model for image analysis."""
    classification: Dict
    objects: Dict
    features: Dict
    metadata: Dict


class CombinedResponse(BaseModel):
    """Response model for combined analysis."""
    text_analysis: Optional[Dict]
    image_analysis: Optional[Dict]
    insights: List[str]
    overall_sentiment: str
    confidence: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    analyzers: Dict


class StatsResponse(BaseModel):
    """Response model for statistics."""
    total_analyses: int
    text_analyzer_ready: bool
    image_analyzer_ready: bool


# Initialize FastAPI app
app = FastAPI(
    title="AI Content Analyzer API",
    description="Analyze text and images with AI",
    version="1.0.0"
)

# Initialize analyzer
analyzer = ContentAnalyzer()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Content Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "text": "/analyze/text",
            "image": "/analyze/image",
            "combined": "/analyze/combined",
            "health": "/health",
            "stats": "/stats"
        }
    }


@app.post("/analyze/text", response_model=TextResponse, tags=["Analysis"])
async def analyze_text_endpoint(request: TextRequest):
    """
    Analyze text content.
    
    Args:
        request: TextRequest with text and options
        
    Returns:
        TextResponse with analysis results
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        result = analyzer.analyze_text(request.text)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/image", response_model=ImageResponse, tags=["Analysis"])
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    Analyze uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        ImageResponse with analysis results
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {ext}"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Analyze image
            result = analyzer.analyze_image(tmp_path)
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            
            return result
            
        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/combined", response_model=CombinedResponse, tags=["Analysis"])
async def analyze_combined_endpoint(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Analyze text and image together.
    
    Args:
        text: Optional text content
        image: Optional image file
        
    Returns:
        CombinedResponse with combined analysis
    """
    try:
        if not text and not image:
            raise HTTPException(
                status_code=400, 
                detail="At least one of text or image must be provided"
            )
        
        image_path = None
        
        # Handle image if provided
        if image and image.filename:
            ext = os.path.splitext(image.filename)[1].lower()
            if ext not in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {ext}"
                )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                content = await image.read()
                tmp.write(content)
                image_path = tmp.name
        
        try:
            # Analyze content
            result = analyzer.analyze_content(
                text=text if text else None,
                image_path=image_path
            )
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            
            return result
            
        finally:
            # Cleanup temporary file
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with system status
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "analyzers": {
            "text": analyzer.text_analyzer is not None,
            "image": analyzer.image_analyzer is not None
        }
    }


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """
    Get system statistics.
    
    Returns:
        StatsResponse with statistics
    """
    stats = analyzer.get_stats()
    return stats


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    print("Starting AI Content Analyzer API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative Docs: http://localhost:8000/redoc")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
