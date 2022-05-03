import math

from recommender_model.recommend import make_recommendation


def test_make_recommendation(test_recommender_input):
    # Given
    expected_no_predictions = len(test_recommender_input)

    # Test
    result = make_recommendation(input_data=test_recommender_input, test=True)

    # Assert
    recommendations = result.get("recommendations")
    assert isinstance(recommendations, list)
    assert isinstance(recommendations[-1], tuple)
    assert result.get("errors") is None
    assert len(recommendations) == expected_no_predictions
    # make sure the translation isn't unreasonably long
    assert math.isclose(len(recommendations[-1][1]), 6, abs_tol=2)
