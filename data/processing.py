import pandas as pd
import numpy as np

def load_and_prepare_data(params_file, output_file):
    params_df = pd.read_csv(params_file)
    df = pd.read_csv(output_file)

    # filter data based on conditions
    df = df[(df['X'] <= 0.01) & (df['X'] >= -0.01) & (df['Y'] <= 0.01) & (df['Y'] >= -0.01)]
    df['X'] = df['X'] * 1000  # convert to mm
    df['Y'] = df['Y'] * 1000
    df['Vz'] = np.log1p(df['Vz'])  # log transformation using log1p to handle small values

    df = df.merge(params_df, on='simulation')

    # identify and remove rows with NaNs and infinite values
    columns = ['X', 'Y', 'Vx', 'Vy', 'Vz']
    conditioning_columns = ['cooling_beam_detuning', 'cooling_beam_radius', 'cooling_beam_power_mw',
                            'push_beam_detuning', 'push_beam_radius', 'push_beam_power',
                            'push_beam_offset', 'quadrupole_gradient', 'vertical_bias_field']

    initial_row_count = len(df)

    for col in columns + conditioning_columns:
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"Column '{col}': {nan_count} NaNs, {inf_count} infinite values found removed.")
            df = df.dropna(subset=[col])
            df = df[~np.isinf(df[col])]

    removed_row_count = initial_row_count - len(df)
    print(f"Total rows removed: {removed_row_count}")

    normalisation_params = {}

    # normalise data columns
    for col in columns:
        mean, std = df[col].mean(), df[col].std()
        df[col] = (df[col] - mean) / std
        normalisation_params[col] = (mean, std)

    # normalise conditioning columns
    for col in conditioning_columns:
        mean, std = df[col].mean(), df[col].std()
        df[col] = (df[col] - mean) / std
        normalisation_params[col] = (mean, std)

    data_to_learn = df[columns].values
    conditioning_data = df[conditioning_columns].values

    print("Imported experimental data.")

    return data_to_learn, conditioning_data, normalisation_params