from enum import Enum
from typing import List

import pandas as pd


class ModelInterface:
    def fit(self, *args, **kwargs):
        raise NotImplementedError


class PropertyGetter:
    def get(self, property_name: str):
        if hasattr(self, property_name):
            return getattr(self, property_name)
        elif hasattr(self, f"get_{property_name}"):
            get_method = getattr(self, f"get_{property_name}")
            if callable(get_method):
                return get_method()
        else:
            raise NotImplementedError(f"Cannot retrieve property {property_name} of object {self}")


class MLProject(PropertyGetter):
    columns: List[str] = None
    x_train: pd.DataFrame = None
    columns_to_dummify: List[str] = []
    bool_columns_to_integer: List[str] = []
    model: ModelInterface

    def load_x_train_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def load_y_train(self) -> pd.DataFrame:
        raise NotImplementedError

    def prepare_x_data(self):
        self.x_train = self.load_x_train_data()[[*self.get("columns")]]
        self.x_train = pd.get_dummies(self.x_train, columns=self.columns_to_dummify)
        for column_name in self.bool_columns_to_integer:
            self.x_train[column_name] = self.x_train[column_name].astype(int)

    def train(self):
        model = self.build_model()
        model.fit()

    def build_model(self) -> ModelInterface:
        raise NotImplementedError


class CSVLoaderMixin(PropertyGetter):
    x_train_file_path: str
    x_test_file_path: str
    y_train_file_path: str

    class FileType(Enum):
        X_TRAIN = "x_train"
        X_TEST = "x_test"
        Y_TRAIN = "y_train"

    def load_csv(self, file_type: FileType):
        return pd.read_csv(self.get(f"{file_type.value}_file_path"))

    def load_x_train_data(self) -> pd.DataFrame:
        return self.load_csv(self.FileType.X_TRAIN)

    def load_y_train_data(self) -> pd.DataFrame:
        return self.load_csv(self.FileType.X_TEST)
