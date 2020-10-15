from enum import Enum
from typing import List, Optional, Dict, Any

import pandas as pd


class DataCategory(Enum):
    TRAIN = "train"
    TEST = "test"


class DataInformation(Enum):
    X = "x"
    Y = "y"


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
    y_columns: List[str] = None
    x_columns_to_dummify: List[str] = []
    y_columns_to_dummify: List[str] = []
    x_bool_columns_to_integer: List[str] = []
    y_bool_columns_to_integer: List[str] = []
    model: ModelInterface
    fit_kwargs: Dict[str, Any]

    def __init__(self):
        self._x_train: Optional[pd.DataFrame] = None
        self._y_train: Optional[pd.DataFrame] = None

    @property
    def x_train(self) -> pd.DataFrame:
        if self._x_train is None:
            self._x_train = self._prepare_data(DataInformation.X, DataCategory.TRAIN)
        return self._x_train

    @property
    def y_train(self) -> pd.DataFrame:
        if self._y_train is None:
            self._y_train = self._prepare_data(DataInformation.Y, DataCategory.TRAIN)
        return self._y_train

    def load_train_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def load_test_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def _prepare_data(self, data_information: DataInformation, data_category: DataCategory) -> pd.DataFrame:
        load_method = getattr(self, f"load_{data_category.value}_data")
        data = load_method()[[*self.get(f"{data_information.value}_columns")]]
        data = pd.get_dummies(data, columns=self.get(f"{data_information.value}_columns_to_dummify"))
        for column_name in self.get(f"{data_information.value}_bool_columns_to_integer"):
            data[column_name] = data[column_name].astype(int)
        return data

    def train(self):
        self.model = self.get("model")
        self.model.fit(x=self.x_train, y=self.y_train, **self.get("fit_kwargs"))


class CSVLoaderMixin(PropertyGetter):
    train_data_file_path: str
    test_data_file_path: str

    def load_csv(self, data_category: DataCategory):
        return pd.read_csv(self.get(f"{data_category.value}_file_path"))

    def load_train_data(self) -> pd.DataFrame:
        # todo : use get("x_train")
        return self.load_csv(DataCategory.TRAIN)

    def load_test_data(self) -> pd.DataFrame:
        return self.load_csv(DataCategory.TEST)
