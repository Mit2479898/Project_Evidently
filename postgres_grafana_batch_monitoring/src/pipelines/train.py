import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model

# Define the model architecture
def create_lstm_autoencoder(input_dim, timesteps, latent_dim):
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(latent_dim)(inputs)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)
    autoencoder = Model(inputs, decoded)
    return autoencoder

def train() -> None:
    """Train a linear regression model on the given dataset."""

    DATA_DIR = "data/features"

    # Define the target variable, numerical features, and categorical features
    target = "BB823_101_0_Z_I_f_nomRW"
    num_features = ['BB823_101_0_Z_f_a_nomRW', 'BB823_101_0_Z_h_count2RW', 'BB823_101_0_S_f_actRW', 'BB823_101_0_S_T_motRW', 'BB823_101_0_S_h_count2RW']
    cat_features = [ 'BB823_101_0_Z_I_f_actRW', 'BB823_101_0_Z_vmot_actRW']

    print("Load train data")
    data = pd.read_parquet(f"{DATA_DIR}/brawo_preproc_onprem.parquet")

    # Filter out outliers
    # data = data[(data.duration_min >= 1) & (data.duration_min <= 60)]
    # data = data[(data.passenger_count > 0) & (data.passenger_count <= 6)]

    # Set the parameters for the model
    input_dim = 3  # Replace with the number of features in your data
    timesteps = 10  # Replace with the sequence length
    latent_dim = 64  # Number of units in the LSTM layer (latent dimension)

    # Split data into training and validation sets
    train_data = data.iloc[:30000, :]
    val_data = data.iloc[30000:60000, :]
    print(train_data.columns)
    print("Train model")
    model = create_lstm_autoencoder(input_dim, timesteps, latent_dim)
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    model.fit(
        X=train_data[num_features + cat_features],
        y=train_data[target],
    )

    print("Get predictions for validation")
    train_preds = model.predict(train_data[num_features + cat_features])
    val_preds = model.predict(val_data[num_features + cat_features])

    print("Calculate validation metrics: MAE")
    # Scoring
    print(mean_absolute_error(train_data[target], train_preds))
    print(mean_absolute_error(val_data[target], val_preds))

    print("Calculate validation metrics: MAPE")
    print(mean_absolute_percentage_error(train_data[target], train_preds))
    print(mean_absolute_percentage_error(val_data[target], val_preds))

    print("Save the model")
    joblib.dump(model, "models/model.joblib")


if __name__ == "__main__":

    train()
