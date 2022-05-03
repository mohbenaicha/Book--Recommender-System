from typing import Any, List, Optional, Set, Tuple

from pydantic import BaseModel
from recommender_model.utilities.validation import \
    RecommendationDataInputSchema


class RecommendationResults(BaseModel):
    recommendations: Optional[List[Tuple[str, Set[str]]]]
    version: str
    errors: Optional[Any]


class MultipleRecommendationDataInputs(BaseModel):
    inputs: List[RecommendationDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {"user_id": "100023"},
                    {"user_id": "100024"}
                ]
            }
        }
