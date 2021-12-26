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
print('text: ' + df.iloc[0, 0] + '\nlabel: ' + df.iloc[0, 1])

# renaming the column hof_OR_none to labels, inplace updates original object
df.rename(columns={'hof_OR_none': 'labels'}, inplace=True)

# converting labels into numerical values
df['labels'] = df['labels'].replace(['HOF', 'NOT'], [1, 0])

print("Labels encoded and column renamed.")
print(df.sample(5))

# print(hs_dataset)
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

# print(x.dtype)
# print(type(x))

#  Train test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print()
print(type(x_train))



VOCAB_SIZE = 900
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(x)

def _get_vocabulary():
    keys, values = encoder._index_lookup_layer._table_handler.data()
    return [x.decode('ISO-8859-1', errors='ignore') for _, x in sorted(zip(values, keys))]

vocab = np.array(_get_vocabulary())
print(vocab[:30])
print()

example = x[:3]
print(example)

encoded_example = encoder(example)
print(encoded_example)

# impl the model

model = tf.keras.Sequential(encoder)

model.add(tf.keras.layers.Embedding(input_dim=len(_get_vocabulary()), output_dim=64, mask_zero=True))
model.add(tf.keras.layers.LSTM(64))
model.add(Dropout(0.2))
model.add(tf.keras.layers.Dense(64, activation='softmax'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=40, batch_size=16)

print(model.summary())
print()

# Python3 code to print
# all encodings available

# predictions = model.predict(np.array([comment]))
# print(predictions[0])

comment = ('Zurück nach Kabul mit den rapefugees.')
predictions = model.predict(np.array([comment]))
print(predictions[0])


if predictions < 0.5:
    print("Non-Hate-speech")
else:
    print("Hate-speech")


test_loss, test_acc = model.evaluate(x_test, y_test)
print("\nTEST PERFORMANCE ")
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


