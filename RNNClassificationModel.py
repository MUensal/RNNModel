import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dropout



print("tensorflow version: " + tf.__version__ + "\n")

# Reading csv_file: skips header, separation with delimiter, encoding needs to be set here
df = pd.read_csv(r'Data_prepared.csv',
                 encoding='ISO-8859-1',
                         engine='python',
                         delimiter=';', header=0)
# print(df.shape)

print(df.head())
# print(df.info)

# example text and label
print('\ntext: ' + df.iloc[0, 0] + '\nlabel: ' + df.iloc[0, 1]+ '\n')

# renaming the column hof_OR_none to labels, inplace updates original object
df.rename(columns={'hof_OR_none': 'labels'}, inplace=True)

# converting labels into numerical values
df['labels'] = df['labels'].replace(['HOF', 'NOT'], [1, 0])

print("--- Labels encoded and column renamed. ---")
print(df.sample(5))

# print(hs_dataset)
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

# print(x.dtype)
# print(type(x))

#  Train test Split
x_trainingset, x_testset, y_traininglabels, y_testlabels = train_test_split(x, y, test_size=0.15)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print()
print(type(x_trainingset))

# ------preprocessing----

# creating a vocabulary
VOCAB_SIZE = 1000

# using encoder on dataset
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(x)

# function to get tokens in vocabulary
def _get_vocabulary():
    keys, values = encoder._index_lookup_layer._table_handler.data()
    return [x.decode('ISO-8859-1', errors='ignore') for _, x in sorted(zip(values, keys))]

# print first 30 tokens
print('\nFirst 30 tokens in vocabulary:\n')
vocab = np.array(_get_vocabulary())
print(vocab[:30])
print()

# print first 3 examples that will be encoded
example = x[2:3]
print(example)

# encoded examples
encoded_example = encoder(example)
print(encoded_example)

# --- implementing the model ---

# text to token
model = tf.keras.Sequential(encoder)

# to vectors
model.add(tf.keras.layers.Embedding(input_dim=len(_get_vocabulary()), output_dim=64, mask_zero=True))

# Wrapper for bidirectional LSTM layers
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))

# vector to final output form for classification result
model.add(tf.keras.layers.Dense(64, activation='relu'))

#
model.add(Dropout(0.2))

model.add(tf.keras.layers.Dense(1))

# Masking for missing inputs is done after Embedding layer
print([layer.supports_masking for layer in model.layers])

# Compiling model - loss function= BinaryCrossentropy, Optimizer = Adam
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])

# Train model
history = model.fit(x_trainingset,
                    y_traininglabels,
                    epochs=40, batch_size=16,
                    validation_data=(x_testset, y_testlabels),
                    validation_steps=30)


print(model.summary())
print()

# Python3 code to print all encodings available

# --- Testing ---

# example prediction
comment = ('Zurück nach Kabul mit den rapefugees.')

predictions = model.predict(np.array([comment]))
print(predictions[0])

if predictions >= 0.0:
    print("Hate-speech")
else:
    print("Non-Hate-speech")

# model evaluation
test_loss, test_acc = model.evaluate(x_testset, y_testlabels)

print("\nTEST PERFORMANCE ")
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


