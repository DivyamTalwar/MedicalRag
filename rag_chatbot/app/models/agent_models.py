from pydantic import BaseModel, Field, validator
from typing import List

class SubQueryGeneration(BaseModel):
    query1: str = Field(
        description="First subquery derived from the main question",
        min_length=5,
        max_length=200,
        example="What are the key features of CIVIE's scheduling system?"
    )
    query2: str = Field(
        description="Second subquery derived from the main question", 
        min_length=5,
        max_length=200,
        example="How does CIVIE handle patient appointment management?"
    )
    
    @validator('query1', 'query2')
    def validate_queries(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class SubQueryResponse(BaseModel):
    subquery_response: str = Field(
        description="Direct, comprehensive answer to the subquery",
        min_length=10,
        example="CIVIE's scheduling system includes automated appointment booking, real-time availability checking, and integration with multiple calendar systems."
    )
    summary: str = Field(
        description="Summary capturing ALL numerical values, percentages, names, dates, and key facts exactly as they appear",
        min_length=5,
        example="Key metrics: 75% automated booking rate, 150+ integration points, supports 24/7 operations."
    )
