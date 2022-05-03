from typing import List

from recommender_model import __version__ as _version
from recommender_model.pipeline import Pipeline
from recommender_model.utilities.validation import validate_inputs

pipe = Pipeline(
    data=None, trained_book_slu=None, trained_user_slu=None, trained_le=None
)


def make_recommendation(*, input_data: List[str], test: bool = False) -> dict:
    """Make a prediction using a saved model pipeline."""

    validated_data, errors = validate_inputs(input_data=input_data)
    validated_data = [element.get("user_id") for element in validated_data]

    results = {"recommendations": None, "version": _version, "errors": errors}

    if not errors:
        recs = pipe.recommend(query=validated_data, test=test)
        results = {
            "recommendations": recs,
            "version": _version,
            "errors": errors,
        }

    return results
