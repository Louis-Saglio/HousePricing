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
    x_columns: List[str] = None
    x_columns_to_dummify: List[str] = []
    x_bool_columns_to_integer: List[str] = []
    model: ModelInterface
    fit_kwargs: Dict[str, Any]

    def __init__(self):
        self._x_train: Optional[pd.DataFrame] = None
        self._y_train: Optional[pd.DataFrame] = None

    @property
    def x_train(self) -> pd.DataFrame:
        if self._x_train is None:
            self._x_train = self._prepare_data(DataType.X_TRAIN)
        return self._x_train

    @property
    def y_train(self) -> pd.DataFrame:
        if self._y_train is None:
            self._y_train = self._prepare_data(DataType.Y_TRAIN)
        return self._y_train

    def load_x_train(self) -> pd.DataFrame:
        raise NotImplementedError

    def load_y_train(self) -> pd.DataFrame:
        raise NotImplementedError

    def _prepare_data(self, data_type: DataType) -> pd.DataFrame:
        if data_type == DataType.X_TRAIN:
            load_method = self.load_x_train
            prefix = "x"
        elif data_type == DataType.Y_TRAIN:
            load_method = self.load_y_train
            prefix = "y"
        elif data_type == DataType.X_TEST:
            load_method = self.load_x_test
            prefix = "x"
        else:
            raise RuntimeError(f"Data type not recognized : {data_type}")
        data = load_method()[[*self.get(f"{prefix}_columns")]]
        data = pd.get_dummies(data, columns=self.get(f"{prefix}_columns_to_dummify"))
        for column_name in self.get(f"{prefix}_bool_columns_to_integer"):
            data[column_name] = data[column_name].astype(int)
        return data

    def train(self):
        self.model = self.get("model")
        self.model.fit(x=self.x_train, y=self.y_train, **self.get("fit_kwargs"))


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
