import os
import warnings

from recommender_model import __version__ as _version
from recommender_model.config.base import TRAINED_MODEL_DIR, config


def func():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    func()
    import tensorflow as tf
    import tensorflow_recommenders as tfrs


class TwoTowerModel(tfrs.Model):
    def __init__(
        self,
        user_model: tf.keras.Model,
        book_model: tf.keras.Model,
        task: tfrs.tasks.Retrieval,
    ):
        super().__init__()

        # embedding representations
        self.user_model = user_model
        self.book_model = book_model
        # this will receive the computed losses by using the retrieval class to
        # calculate the FactorizedTopK
        self.task = task

    def compute_loss(self, features, training=False) -> tf.Tensor:
        # Define how the loss is computed.

        user_embeddings = self.user_model(features[1])
        book_embeddings = self.book_model(features[0])

        return self.task(user_embeddings, book_embeddings, compute_metrics=False)


def load_model(
    *, save_model_name: str = config.app_config.model_name, test: bool = False
):

    save_file_name = f"{save_model_name}{_version}"
    if test:
        save_path = os.path.join(TRAINED_MODEL_DIR, save_file_name, save_file_name)
    else:
        save_path = os.path.join(TRAINED_MODEL_DIR, save_file_name)
    loaded_model = tf.saved_model.load(save_path)
    return loaded_model
