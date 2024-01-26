import pandas as pd
from mapie.regression import MapieRegressor
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    x, y = load_diabetes(return_X_y=True, as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    model = RandomForestRegressor()
    model.fit(x, y)

    predictions_df = pd.concat([
        y_test.reset_index(drop=True),
        pd.DataFrame(model.predict(x_test), columns=['prediction'])
    ], axis=1)

    mapie_regressor = MapieRegressor(model)
    mapie_regressor.fit(x, y)
    alpha = [0.05]
    _, y_pis = mapie_regressor.predict(x_test, alpha=alpha)
    lower_bound = y_pis[:, 0]
    upper_bound = y_pis[:, 1]
    predictions_df['lower'] = lower_bound
    predictions_df['upper'] = upper_bound

    print(predictions_df.head())


