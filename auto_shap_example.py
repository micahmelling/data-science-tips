from auto_shap.auto_shap import generate_shap_values
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier

if __name__ == "__main__":
    x, y = load_breast_cancer(return_X_y=True, as_frame=True)
    model = ExtraTreesClassifier()
    model.fit(x, y)
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model, x)

    print(shap_values_df.head())
    print()
    print(shap_expected_value)
    print()
    print(global_shap_df.head())
    print()
