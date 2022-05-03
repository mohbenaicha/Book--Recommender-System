import pytest

from recommender_model.config.base import config
from recommender_model.utilities.data_manager import load_dataset, zip_unzip_model


@pytest.fixture()
def test_input_data():
    return load_dataset(
        file_name=config.app_config.test_data_file,
        features_to_drop=config.model_config.features_to_drop_te,
    )


# Normally, the user ID's would be received in thier raw format,
# to ensure the fitted LabelEncoder works as it is supposed.
# However, since this code repo is being made public,
# the encoded ID's are used to maintain confidentiality.
@pytest.fixture()
def test_recommender_input():
    return ["100056", "100049", "100047", "100023", "100012"]


@pytest.fixture()
def test_zip_unzip():
    return (zip_unzip_model(
        zip=True, test=True), zip_unzip_model(zip=False, test=True))
