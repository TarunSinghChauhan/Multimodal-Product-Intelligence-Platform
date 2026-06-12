from typing import List, Optional
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    query_text: str
    category: Optional[str] = None
    price_range: Optional[List[float]] = None # [min, max]
    min_rating: Optional[float] = 0.0
    k: int = 10

class ImageSearchQuery(BaseModel):
    image_base64: str
    k: int = 10

class MultimodalSearchQuery(BaseModel):
    query_text: Optional[str] = None
    image_base64: Optional[str] = None
    k: int = 10

class QAQuery(BaseModel):
    question: str
    session_id: Optional[str] = None

class ProductResponse(BaseModel):
    product_id: str
    title: str
    category: str
    price: float
    avg_rating: float
    similarity_score: float
    description_snippet: str
    attributes: dict = Field(default_factory=dict)

class QAResponse(BaseModel):
    answer: str
    source_products: List[str]

class AnalyticsResponse(BaseModel):
    query_volume_by_hour: dict
    zero_result_rate: float
    avg_similarity_score: float
    top_queries: List[str]
