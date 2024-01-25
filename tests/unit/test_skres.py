import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TestUintSkres:
    """Test class for skres package"""

    def test_df(self):
        """Test the creation of the dataframe"""

        base = "https://gist.githubusercontent.com/AlexandreGazagnes/"
        url = base + "9018022652ba0933dd39c9df8a600292/raw/"
        url += "0845ef4c2df4806bb05c8c7423dc75d93e37400f/titanic_train_raw_csv"

        df = pd.read_csv(url)

        assert isinstance(df, pd.DataFrame)

    def test_pipeline(self):
        """Test the creation a pipeline"""

        pipe = Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("model", DummyClassifier()),
            ]
        )

        assert isinstance(pipe, Pipeline)

    def test_grid(self):
        """Test the creation of a grid"""

        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("model", DummyClassifier()),
            ]
        )

        param_grid = {
            "imputer__strategy": ["mean", "median", "most_frequent"],
            "scaler__with_mean": [True, False],
            "estimator__strategy": ["most_frequent", "stratified", "uniform"],
        }

        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            refit=True,
        )

        assert isinstance(grid, GridSearchCV)
