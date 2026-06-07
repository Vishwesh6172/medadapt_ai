from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


def build_model(input_dim):
    print("\n[Model] Building DNN...")

    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model