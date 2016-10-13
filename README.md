Keras Tools.
=========

A collection of tools and helper functions I've made for the Keras machine learning framework.
___
## keras_io
A nice, simple to use 2-function helper that you can import and use to fully save your keras model layout and weights, and then import them back in again. When specifying a filename, do not specify an extension as the extensions for the layout (.json) and weights (.h5) are appended in the function.

```python
import keras_io

# ...Model Code...

model.fit(x_train, y_train, nb_epoch=epochs)
keras_io.save_model(model, 'filename')

# ...Later...

model = keras_io.load_model('filename')
```
___
## keras_callback
A neater callback function to display the ETA to model training completion on one line, with a total model training ETA rahter than a per-epoch ETA.

```python
from keras_callback import CustomCallback

# ...Model Code...

callback = CustomCallback(epochs, 100)
model.fit(x_train, y_train, nb_epoch=epochs, verbose=0, callbacks=[callback])
```