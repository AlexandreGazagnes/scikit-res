__version__ = "0.0.16"


import pandas as pd
from sklearn.model_selection import GridSearchCV


class SkRes(pd.DataFrame):
    """Main class of the package"""

    def __init__(self, grid: GridSearchCV) -> pd.DataFrame:
        """initialisation of the class"""

        if not isinstance(grid, GridSearchCV):
            raise TypeError("grid must be a GridSearchCV object")

        # get the results
        res = pd.DataFrame(grid.cv_results_)

        # sort the results
        res.sort_values("mean_test_score", ascending=False, inplace=True)

        # remove the split columns
        cols = [i for i in res.columns if "split" not in i]
        res = res.loc[:, cols]

        # round the results
        res = res.round(4)

        super().__init__(res)
