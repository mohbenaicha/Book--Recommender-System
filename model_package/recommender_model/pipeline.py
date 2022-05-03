import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import recommender_model.utilities.model_tools as mt
from recommender_model.config.base import config
from recommender_model.utilities.data_manager import load_le


def func():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    func()
    import tensorflow as tf
    import tensorflow_recommenders as tfrs


class Pipeline:
    def __init__(
        self,
        data: pd.DataFrame,
        trained_book_slu: tf.keras.layers.StringLookup,
        trained_user_slu: tf.keras.layers.StringLookup,
        trained_le=None, # no type hints, used an if to assure typing is correct
        trained_model: Optional[tfrs.layers.factorized_top_k.BruteForce] = None,
    ):

        self.data = data
        self.trained_le = trained_le

        if trained_model:
            assert trained_book_slu and trained_user_slu
            self.trained_model = trained_model

        self.trained_book_slu = trained_book_slu
        self.trained_user_slu = trained_user_slu

    def preprocess(
        self,
        data: pd.DataFrame,
        feature_1: str = config.model_config.book_column,
        feature_2: str = config.model_config.user_column,
        features_to_labelencode: List[str] = (config.model_config.label_encode_columns),
        train_slu: bool = True,
        train_le: bool = True,
    ) -> None:

        if not data and self.data.empty:
            print("Error: no data provided.")
        elif data is None:
            assert not self.data.empty
            print("self.data.empty found")
        else:
            assert isinstance(data, pd.DataFrame)
            self.data = data

        self.features_to_labelencode = features_to_labelencode
        self.feature_1 = feature_1
        self.feature_2 = feature_2

        for feat in self.features_to_labelencode:
            if train_le:  # or fitting label encoders
                le = LabelEncoder()
                le.fit(self.data[feat])
                transformed = le.transform(self.data[feat])
                self.data[feat] = transformed
                self.trained_le[feat] = le

            else:  # using fitted label encoders
                if self.trained_le.get(feat) is not None and isinstance(
                    self.trained_le.get(feat), LabelEncoder
                ):
                    self.data[feat] = self.trained_le.get(feat).transform(
                        self.data[feat]
                    )

        print(self.data[self.feature_1][:5])
        print(self.data[[self.feature_1, self.feature_2]][:5])

        self.books = tf.data.Dataset.from_tensor_slices(
            self.data[self.feature_1].astype("str").values
        ).shuffle(128, seed=config.model_config.tf_seed)

        self.ratings = tf.data.Dataset.from_tensor_slices(
            self.data[[self.feature_1, self.feature_2]].astype("str").values
        ).shuffle(128, seed=config.model_config.tf_seed)

        if train_slu:
            print("Adapting book StringLookup. This will take a while...")
            self.trained_book_slu = tf.keras.layers.StringLookup(mask_token=None)
            self.trained_book_slu.adapt(self.books)

            print("Adapting user StringLookup. This will take a while...")
            self.trained_user_slu = tf.keras.layers.StringLookup(mask_token=None)
            self.trained_user_slu.adapt(self.ratings.map(lambda x: x[1]))

    def train_recommender(
        self,
        second_embedding_dim: int = config.model_config.second_embed_dim,
        lr: float = config.model_config.learning_rate,
        n_epochs: int = config.model_config.epochs,
        batch_size: int = config.model_config.batch_size,
    ) -> mt.TwoTowerModel:

        if not (self.trained_book_slu and self.trained_user_slu):
            print("Error, attempting to train without string lookups")
        elif self.data.empty:
            print("No data to train on.")

        # setup layer and loss computer
        user_model = tf.keras.Sequential(
            [
                self.trained_user_slu,
                tf.keras.layers.Embedding(
                    self.trained_user_slu.vocabulary_size() + 1, second_embedding_dim
                ),
            ]
        )

        book_model = tf.keras.Sequential(
            [
                self.trained_book_slu,
                tf.keras.layers.Embedding(
                    self.trained_book_slu.vocabulary_size() + 1, second_embedding_dim
                ),
            ]
        )

        task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(self.books.batch(128).map(book_model))
        )

        # load model skeleton and build
        tt_model = mt.TwoTowerModel(user_model, book_model, task)
        tt_model.compile(optimizer=tf.keras.optimizers.Adam(lr))

        # train model and build bruteforce indexer

        print("Fitting recommender. This may take a while...")
        tt_model.fit(self.ratings.batch(batch_size), epochs=n_epochs, verbose=False)
        indexer = tfrs.layers.factorized_top_k.BruteForce(
            tt_model.user_model, k=config.model_config.k_recommendations
        )
        self.trained_model = indexer.index_from_dataset(
            self.books.batch(128).map(lambda title: (title, tt_model.book_model(title)))
        )

    def recommend(self, query: List[str], test: bool = False) -> List[Tuple[str, set]]:

        loaded_model = mt.load_model(test=test)
        recommendations = list()
        for element in query:
            _, recommendation = loaded_model(tf.constant([element]))
            recommendations.append(recommendation)

        le = load_le().get(config.model_config.book_column)
        results = list()
        for idx, user in zip(query, recommendations):
            results.append(
                (
                    f"User {idx}",
                    set(
                        [
                            le.inverse_transform(
                                np.array([int(user.numpy()[0][i].decode())])
                            )[0]
                            for i in range(len(user.numpy()[0]))
                        ]
                    ),
                )
            )

        return results
