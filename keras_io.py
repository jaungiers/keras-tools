from keras.models import model_from_json

def save_model(model, filename):
	# serialize model to JSON
	model_json = model.to_json()
	with open(filename + '.json', 'w') as json_file:
		json_file.write(model_json)

	# serialize weights to HDF5
	model.save_weights(filename + '.h5', overwrite=True)
	print '### Model Saved as', filename + '.h5/.json ###'

def load_model(filename):
	# load json and create model
	json_file = open(filename + '.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# load weights into new model
	loaded_model.load_weights(filename + '.h5')
	print '### Model Loaded from disk ###'
	return loaded_model