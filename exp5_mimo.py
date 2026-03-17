import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Step 1: Generate Dataset
# -----------------------------

np.random.seed(42)

X = np.random.rand(500,4)

y1 = X[:,0] + X[:,1]
y2 = X[:,2] * X[:,3]

Y = np.column_stack((y1,y2))


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


learning_rate = 0.01
epochs = 50
neurons = 32

input_layer = Input(shape=(4,))

hidden1 = Dense(neurons,activation='relu')(input_layer)
hidden2 = Dense(neurons,activation='relu')(hidden1)

output = Dense(2)(hidden2)

model = Model(inputs=input_layer,outputs=output)

optimizer = Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)
model.summary()
history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test,Y_test),
    epochs=epochs,
    verbose=1
)

# -----------------------------
# Step 7: Evaluation
# -----------------------------

loss,mae = model.evaluate(X_test,Y_test)

print("Test Loss:",loss)
print("Test MAE:",mae)

# -----------------------------
# Step 8: Table Output
# -----------------------------

results = pd.DataFrame({
    "Epoch":range(1,epochs+1),
    "Training Loss":history.history['loss'],
    "Validation Loss":history.history['val_loss']
})

print("\nTraining Results Table\n")
print(results)

# -----------------------------
# Step 9: Graph Output
# -----------------------------

plt.figure()

plt.plot(results["Epoch"],results["Training Loss"],label="Training Loss")
plt.plot(results["Epoch"],results["Validation Loss"],label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MIMO Model Training Performance")

plt.legend()

plt.show()
