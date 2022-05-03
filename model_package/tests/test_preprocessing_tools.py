import numpy as np
import tensorflow as tf

from recommender_model.config.base import TRAINED_SLU_DIR, config
from recommender_model.utilities.data_manager import load_le, load_slu

# some of the tests are commented out for privacy reasons
# noted in conftest.py


def test_label_encoder(test_input_data):

    # Test
    le_name = config.app_config.label_encoder_name
    le = load_le(file_name=f"{le_name}").get(config.model_config.book_column)
    titles = np.array(
        [
            "Forgotten Fire",
            "Kiss Hollywood Goodbye",
            "In Constant Fear The Detainee Trilogy 3",
        ]
    )
    encoded = np.array([1337, 1001, 1])
    test_object_1 = le.transform(titles)
    test_object_2 = le.inverse_transform(encoded)

    # Assert
    assert all(test_object_1 == np.array([15941, 22607, 20557]))
    assert all(
        test_object_2
        == np.array(
            [
                "A Little Russian Cook Book International little cookbooks",
                "A General Introduction to Psychoanalysis",
                " E venne chiamata due cuori",
            ]
        )
    )

    # Confidential
    # le = load_le(config.app_config.label_encoder_name).get(
    #     'user_le')
    # users, encoded = np.empty([3,1]), np.empty([3,1])
    # test_object_3 == le.transform(users)
    # test_object_4 == le.inverse_transform(encoded)
    # assert test_object_3 == np.empty([3,1])
    # assert test_object_4 == np.empty([3,1])


def test_stringlookups(test_input_data):
    # Test:
    slu_path = TRAINED_SLU_DIR / config.app_config.books_string_lookup_name
    book_col = config.model_config.book_column
    slu_books = load_slu(file_name=slu_path)
    test_object_1 = slu_books.get_vocabulary()
    test_object_2 = slu_books.call(
        tf.constant(
            load_le().get(book_col).transform(test_input_data[book_col]).astype(str)
        )
    )

    # Assert:
    assert test_object_1[:10] == [
        "[UNK]",
        "52974",
        "48945",
        "40371",
        "57083",
        "49257",
        "42587",
        "30094",
        "25989",
        "1697",
    ]
    assert len(test_object_1) == 57198 + 1
    assert len(test_object_2) == 10
    assert all(
        test_object_2
        == np.array([47705, 9212, 9106, 10005, 1401, 3669, 49909, 3666, 3668, 47638])
    )

    # Confidential
    # slu_path = TRAINED_SLU_DIR / config.app_config.users_string_lookup_name
    # test_object_3 = slu_users.get_vocabulary()[:10]
    # test_object_4 = slu_users.call(tf.constant(
    #     test_input_data[0].values[:, 0].astype(str)))
    # assert test_object_3 == 'CONFIDENTIAL'
    # assert test_object_4 == 'CONFIDENTIAL'


def test_saved_model_packaging(test_zip_unzip):
    # Test: test_zip_unzip

    # Assert
    assert test_zip_unzip == (True , True)
