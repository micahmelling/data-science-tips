import pandas as pd
from auto_shap.auto_shap import generate_shap_values
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    x, y = load_breast_cancer(return_X_y=True, as_frame=True)
    # model = ExtraTreesClassifier()
    # model.fit(x, y)
    # shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x)
    #
    # print(shap_values_df.head())
    # print()
    # print(shap_expected_value)
    # print()
    # print(global_shap_df.head())
    # print()
    #
    # import sys
    # sys.exit()

    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler().set_output(transform="pandas")),
        ('model', ExtraTreesClassifier())
    ])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    pipeline.fit(x_train, y_train)

    scaler = pipeline.named_steps['scaler']
    model = pipeline.named_steps['model']
    print(x_test.head())
    x_test = scaler.transform(x_test)
    print(x_test.head())
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x_test)
    print(shap_values_df.head())
    print()
    print(shap_expected_value)
    print()
    print(global_shap_df.head())
    print()
