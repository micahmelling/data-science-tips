import pandas as pd


if __name__ == "__main__":
    df = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06'],
        'group': ['A', 'A', 'B', 'B', 'A', 'B'],
        'value': [10, 20, 15, 25, 30, 40]
    })

    df['expanding_group_mean'] = df.groupby('group')['value'].expanding().mean().reset_index(level=0, drop=True)
    print(df)
    print()

    ###################################################################################################################

    df = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06',
                 '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10'],
        'group': ['A', 'A', 'B', 'B', 'A', 'B', 'B', 'B', 'B', 'B'],
        'value': [10, 20, 15, 25, 30, 40, 45, 60, 5, 10]
    })

    def clipped_mean(series):
        lower_bound = series.quantile(0.05)
        upper_bound = series.quantile(0.95)
        print(lower_bound)
        print(upper_bound)
        print()
        clipped_series = series.clip(lower=lower_bound, upper=upper_bound)
        mean = clipped_series.mean()
        return mean

    df['expanding_clipped_mean'] = df['value'].expanding().apply(clipped_mean)
    df['expanding_mean'] = df['value'].expanding().mean()
    print(df)
