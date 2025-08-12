from pydantic import BaseModel, Field, validator
from typing import List

class SubQueryGeneration(BaseModel):
    queries: List[str] = Field(
        description="A list of 2-5 subqueries derived from the main question",
        min_items=2,
        max_items=5,
        example=[
            "What are the key features of CIVIE's scheduling system?",
            "How does CIVIE handle patient appointment management?"
        ]
    )

    @validator('queries')
    def validate_queries(cls, v):
        if not all(q.strip() for q in v):
            raise ValueError("All queries must be non-empty")
        return [q.strip() for q in v]

class SubQueryResponse(BaseModel):
    subquery_response: str = Field(
        description="Direct, comprehensive answer to the subquery",
        example="CIVIE's scheduling system includes automated appointment booking, real-time availability checking, and integration with multiple calendar systems."
    )
    summary: str = Field(
        description="Summary capturing ALL numerical values, percentages, names, dates, and key facts exactly as they appear",
        example="Key metrics: 75% automated booking rate, 150+ integration points, supports 24/7 operations."
    )

class FinalAnswer(BaseModel):
    final_answer: str = Field(
        description="The final, comprehensive, and synthesized answer to the user's query.",
        min_length=20
    )
