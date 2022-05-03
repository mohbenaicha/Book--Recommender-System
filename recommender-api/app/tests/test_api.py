import math
from typing import List

from fastapi.testclient import TestClient


def test_make_recommendation(client: TestClient, test_data: List[List[str]]) -> None:
    # Given
    payload = {
        "inputs": [{"user_id": i} for i in test_data]
    }

    # When
    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )

    # Then
    assert response.status_code == 200
    result = response.json()

    expected_no_predictions = len(test_data)

    recommendations = result.get("recommendations")
    assert isinstance(recommendations, list)
    # json decoding returns a list of lists rather than a list of typles
    assert isinstance(recommendations[-1], list)
    assert result.get("errors") is None
    assert len(recommendations) == expected_no_predictions
    # make sure the translation isn't unreasonably long
    assert math.isclose(len(recommendations[-1][1]), 6, abs_tol=2)
