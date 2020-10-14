from enum import Enum
from typing import List, Optional, Dict, Any

import pandas as pd


class DataType(Enum):
    X_TRAIN = "x_train"
    X_TEST = "x_test"
    Y_TRAIN = "y_train"


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
    columns_to_dummify: List[str] = []
    bool_columns_to_integer: List[str] = []
    model: ModelInterface
    fit_kwargs: Dict[str, Any]

    def __init__(self):
        self._x_train: Optional[pd.DataFrame] = None

    @property
    def x_train(self) -> pd.DataFrame:
        if self._x_train is None:
            self._x_train = self.prepare_data(DataType.X_TRAIN)
        return self._x_train

    def load_x_train(self) -> pd.DataFrame:
        raise NotImplementedError

    def load_y_train(self) -> pd.DataFrame:
        raise NotImplementedError

    def prepare_data(self, data_type: DataType) -> pd.DataFrame:
        x_data = (self.load_x_train if data_type == DataType.X_TRAIN else self.load_y_train)()[[*self.get("columns")]]
        x_data = pd.get_dummies(x_data, columns=self.get("columns_to_dummify"))
        for column_name in self.get("bool_columns_to_integer"):
            x_data[column_name] = x_data[column_name].astype(int)
        return x_data

    def train(self):
        self.model = self.get("model")
        self.model.fit(x=self.x_train, **self.get("fit_kwargs"))


class CSVLoaderMixin(PropertyGetter):
    x_train_file_path: str
    x_test_file_path: str
    y_train_file_path: str

    def load_csv(self, data_type: DataType):
        return pd.read_csv(self.get(f"{data_type.value}_file_path"))

    def load_x_train(self) -> pd.DataFrame:
        # todo : use get("x_train")
        return self.load_csv(DataType.X_TRAIN)

    def load_y_train(self) -> pd.DataFrame:
        return self.load_csv(DataType.X_TEST)
