import tensorflow as tf

from recommender_model.config.base import config
from recommender_model.pipeline import Pipeline
from recommender_model.utilities.data_manager import (
    load_dataset,
    save_le,
    save_model,
    save_slu,
)


def train() -> None:
    loaded_data = load_dataset(
        file_name=(f"{config.app_config.train_data_file}"),
        features_to_drop=(f"{config.model_config.features_to_drop_tr}"),
    )

    # Loading fitted preprocessors
    # loaded_slu_books = load_slu(file_name=
    #     config.app_config.books_string_lookup_name)
    # loaded_slu_users = load_slu(file_name=
    #     config.app_config.users_string_lookup_name)
    # loaded_le = load_le(file_name=
    #     config.app_config.label_encoder_name)

    # train le, slu and persist
    pipeline = Pipeline(
        data=loaded_data,
        trained_book_slu=None,
        trained_user_slu=None,
        trained_le=dict(),
    )

    pipeline.preprocess(data=None, train_slu=True, train_le=True)

    for p, boolean in zip(
        [pipeline.trained_book_slu, pipeline.trained_user_slu],  # iter 1
        [True, False],  # iter 2
    ):
        save_slu(slu_to_persist=p, book=boolean)

    save_le(le_to_persist=(pipeline.trained_le))

    # train recommender and persist
    pipeline.train_recommender()

    # persist trained_model
    # Note: Keras' _UserObjects somehow aren't called properly unles
    # the model/layer that is initially saved is called once, so it
    # is called on a random user contained in th string lookup as a
    # workaround until a permanent fix is discovered

    _, _ = pipeline.trained_model(
        tf.constant([pipeline.trained_user_slu.get_vocabulary()[2]])
    )

    save_model(model_to_persist=(pipeline.trained_model))


if __name__ == "__main__":
    train()
