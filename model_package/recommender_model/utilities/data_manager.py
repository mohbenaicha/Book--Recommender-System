import os
import shutil
import warnings
import zipfile
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from recommender_model import __version__ as _version
from recommender_model.config.base import (
    DATASET_DIR,
    TRAINED_LE_DIR,
    TRAINED_MODEL_DIR,
    TRAINED_SLU_DIR,
    config,
)


def func():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    func()
    import tensorflow as tf
    import tensorflow_recommenders as tfrs
    from tensorflow.keras.layers import StringLookup


def load_dataset(
    *, file_name: str = None, features_to_drop: List[str] = None
) -> pd.DataFrame:
    #
    print("Attempting to load data...")
    df = pd.read_csv(
        (f"{DATASET_DIR}/{file_name}" + f"{config.app_config.data_version}.csv")
    )

    # 1. drop features for training
    # 2. dropping our columns of choice here and not
    #    in load_data() gives flexibility as to what
    #    columns we can keep from load_data() call to
    #    utilize during testing
    try:
        df.drop(columns=features_to_drop, inplace=True)
    except Exception:
        pass

    # The decision to drop nan is not arbitrary since nan values are almost
    # useless  when it comes to recommender systems. Further exploration
    # could be done as to the nature of the nan values once they're collected
    df.dropna(axis=0, inplace=True)

    return df


def remove_old_slu(*, files_to_keep: List[str]) -> None:
    """
    Iterates through every file in the target directory and removes all
    but the new pipeline file and the __init__.py file.
    """
    do_not_delete = files_to_keep
    for file in TRAINED_SLU_DIR.iterdir():
        if file.name not in do_not_delete:
            file.unlink()


def save_slu(*, slu_to_persist: StringLookup, book: bool) -> None:

    if book:
        save_file_name = (
            f"{config.app_config.books_string_lookup_name}" + f"{_version}.npy"
        )

    else:
        save_file_name = (
            f"{config.app_config.users_string_lookup_name}" + f"{_version}.npy"
        )

    save_path = TRAINED_SLU_DIR / save_file_name
    remove_old_slu(
        files_to_keep=[
            (f"{config.app_config.users_string_lookup_name}" + f"{_version}.npy"),
            (f"{config.app_config.books_string_lookup_name}" + f"{_version}.npy"),
        ]
    )
    np.save(save_path, slu_to_persist.get_weights())


def load_slu(*, file_name: str) -> StringLookup:

    file_path = TRAINED_SLU_DIR / file_name
    slu = StringLookup(mask_token=None)
    slu.set_weights(np.load(f"{file_path}{_version}.npy", allow_pickle=True))
    return slu


def remove_old_le(*, files_to_keep: List[str]) -> None:
    """
    Iterates through every file in the target directory and removes all
    but the new pipeline file and the __init__.py file.
    """
    do_not_delete = files_to_keep
    for file in TRAINED_LE_DIR.iterdir():
        if file.name not in do_not_delete:
            file.unlink()


def save_le(*, le_to_persist: LabelEncoder) -> None:

    save_file_name = f"{config.app_config.label_encoder_name}" + f"{_version}.pkl"

    save_path = TRAINED_LE_DIR / save_file_name
    remove_old_le(files_to_keep=[save_file_name])
    joblib.dump(le_to_persist, save_path)


def load_le(*, file_name: str = config.app_config.label_encoder_name) -> LabelEncoder:

    file_path = TRAINED_LE_DIR / file_name
    le = joblib.load(f"{file_path}{_version}.pkl")
    return le


def save_model(*, model_to_persist: tfrs.layers.factorized_top_k.BruteForce) -> None:
    # define name pipeline of newely trained model
    save_folder_name = f"{config.app_config.model_name}{_version}"
    save_path = TRAINED_MODEL_DIR / save_folder_name

    # a mix of Path objects and string formatting is used here since
    # the Keras save method doesn't work well with Path objects

    if Path(save_path).exists():
        shutil.rmtree(save_path)
    tf.saved_model.save(model_to_persist, os.path.join(save_path, save_folder_name))

    # A temporary workaround for maintaining Keras assets dir
    # is used where a placeholder is created in the model/assets
    # dir, given that this custom model doesn't have it's assets
    # saved although it clearly retains the stringlookup
    # information

    placeholder_dir = os.path.join(save_path, save_folder_name, "assets", "placeholder")
    open(placeholder_dir, "a").close()


def load_model(
    *, save_model_name: str = config.app_config.model_name, test: bool = False
):

    save_file_name = f"{save_model_name}{_version}"
    if test:
        save_path = TRAINED_MODEL_DIR / save_file_name / save_file_name
    else:
        save_path = TRAINED_MODEL_DIR / save_file_name

    loaded_model = tf.saved_model.load(save_path)
    return loaded_model


def zip_unzip_model(
    folder_path: str = os.path.join(
        TRAINED_MODEL_DIR, f"{config.app_config.model_name}{_version}"
    ),
    zip_path: str = os.path.join(
        TRAINED_MODEL_DIR, f"{config.app_config.zipped_model_name}{_version}.zip"
    ),
    zip: bool = True,
    test: bool = False,
):
    if zip:
        with zipfile.ZipFile(
            zip_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zipf:
            len_dir_path = len(folder_path)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, file_path[len_dir_path:])
            zipf.close()
            shutil.rmtree(folder_path)
            if test:
                result = os.path.isdir(folder_path) and os.path.isfile(
                    zip_path
                )  # false and true
                return result is False and True

    else:
        if test:
            os.mkdir(folder_path)
        else:
            folder_path = TRAINED_MODEL_DIR
        with zipfile.ZipFile(file=zip_path, mode="r") as f:
            f.extractall(folder_path)
            f.close()
            os.remove(zip_path)
            if test:
                result = os.path.isfile(zip_path) and os.path.isdir(folder_path)
                return result is False and True
