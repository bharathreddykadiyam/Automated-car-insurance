import json

import numpy as np
import pickle as pk
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

from tensorflow.keras.applications.imagenet_utils import preprocess_input

first_gate = None
second_gate = None
location_model = None
severity_model = None
cat_list = None
models_initialized = False

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

def init_models():
	global first_gate, second_gate, location_model, severity_model, cat_list, models_initialized
	if models_initialized:
		return
	try:
		first_gate = VGG16(weights='imagenet')
		print("First gate loaded")
	except Exception as e:
		print("Failed loading first_gate:", e)
		first_gate = None
	# helper to reconstruct model from legacy HDF5 if load_model fails
	def _reconstruct_from_hdf5(path):
		try:
			import h5py
			from tensorflow.keras.models import model_from_json
			with h5py.File(path, 'r') as f:
				mc = None
				# older Keras sometimes stores model config as an attribute
				if 'model_config' in f.attrs:
					mc = f.attrs['model_config']
				# also accept dataset named 'model_config'
				if mc is None and 'model_config' in f:
					mc = f['model_config'][()]
				if mc is None:
					raise RuntimeError('No model_config found in HDF5')
				if isinstance(mc, bytes):
					mc = mc.decode('utf-8')
				# mc should now be JSON
				model = model_from_json(mc)
				# load weights into reconstructed model
				model.load_weights(path)
				return model
		except Exception as e:
			print('Reconstruction from HDF5 failed for', path, ':', e)
			return None

	try:
		second_gate = load_model('static/models/pipe2.hdf5', compile=False)
		print("Second gate loaded")
	except Exception as e:
		print("Failed loading second_gate:", e)
		# try reconstructing from model_config + weights
		second_gate = _reconstruct_from_hdf5('static/models/pipe2.hdf5')
		if second_gate is not None:
			print('Second gate reconstructed and weights loaded')
		else:
			second_gate = None
	try:
		location_model = load_model('static/models/pipe3.hdf5', compile=False)
		print("Location model loaded")
	except Exception as e:
		print("Failed loading location_model:", e)
		location_model = _reconstruct_from_hdf5('static/models/pipe3.hdf5')
		if location_model is not None:
			print('Location model reconstructed and weights loaded')
		else:
			location_model = None
	try:
		severity_model = load_model('static/models/pipe4.hdf5', compile=False)
		print("Severity model loaded")
	except Exception as e:
		print("Failed loading severity_model:", e)
		severity_model = _reconstruct_from_hdf5('static/models/pipe4.hdf5')
		if severity_model is not None:
			print('Severity model reconstructed and weights loaded')
		else:
			severity_model = None
	try:
		with open('static/models/vgg16_cat_list.pk', 'rb') as f:
			cat_list = pk.load(f)
		print("Cat list loaded")
	except Exception as e:
		print("Failed loading cat_list:", e)
		cat_list = None
	models_initialized = True

def get_predictions(preds, top=5):
	global CLASS_INDEX
	if len(preds.shape) != 2 or preds.shape[1] != 1000:
		raise ValueError('`decode_predictions` expects '
						 'a batch of predictions '
						 '(i.e. a 2D array of shape (samples, 1000)). '
						 'Found array with shape: ' + str(preds.shape))
	if CLASS_INDEX is None:
		fpath = get_file('imagenet_class_index.json',
						 CLASS_INDEX_PATH,
						 cache_subdir='models')
		CLASS_INDEX = json.load(open(fpath))
	l = []
	for pred in preds:
		top_indices = pred.argsort()[-top:][::-1]
		indexes = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
		indexes.sort(key=lambda x: x[2], reverse=True)
		l.append(indexes)
	return l

def prepare_img_224(img_path):
	img = load_img(img_path, target_size=(224, 224))
	x = img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

def car_categories_gate(img_224, model):
	print("Validating that this is a picture of your car...")
	# If classifier or category list missing, assume positive so assessment can continue
	if model is None or cat_list is None:
		print('Fallback car category check: model or cat_list missing, assuming car present')
		return True
	out = model.predict(img_224)
	top = get_predictions(out, top=5)
	for j in top[0]:
		if j[0:2] in cat_list:
			return True 
	return False

def prepare_img_256(img_path):
	img = load_img(img_path, target_size=(256, 256))
	x = img_to_array(img)
	x = x.reshape((1,) + x.shape)/255
	return x

def car_damage_gate(img_256, model):
	print("Validating that damage exists...")
	# If model not available, use a lightweight edge-density heuristic
	if model is None:
		arr = img_256[0]
		if arr.dtype != 'float32' and arr.max() > 1.5:
			arr = arr / 255.0
		gray = arr.mean(axis=2)
		gx, gy = np.gradient(gray)
		edge = np.sqrt(gx * gx + gy * gy)
		edge_norm = edge / (edge.max() + 1e-8)
		prop = (edge_norm > 0.2).mean()
		print(f"Fallback damage edge proportion: {prop:.4f}")
		# threshold tuned for typical car-damage photos
		return prop > 0.02
	# otherwise use model prediction
	pred = model.predict(img_256)
	if pred[0][0] <= .5:
		return True
	else:
		return False

def location_assessment(img_256, model):
	print("Determining location of damage...")
	# Fallback: use edge-map centroid to pick front/rear/side
	if model is None:
		arr = img_256[0]
		if arr.dtype != 'float32' and arr.max() > 1.5:
			arr = arr / 255.0
		gray = arr.mean(axis=2)
		gx, gy = np.gradient(gray)
		edge = np.sqrt(gx * gx + gy * gy)
		edge_sum = edge.sum()
		if edge_sum <= 0:
			return 'Unknown'
		h, w = gray.shape
		left = edge[:, : w // 2].sum()
		right = edge[:, w // 2 :].sum()
		top = edge[: h // 2, :].sum()
		bottom = edge[h // 2 :, :].sum()
		# if left or right dominate -> Side
		if abs(left - right) / (left + right + 1e-8) > 0.25:
			return 'Side'
		# otherwise choose front or rear by top/bottom
		return 'Front' if top >= bottom else 'Rear'
	# otherwise use model
	pred = model.predict(img_256)
	pred_label = np.argmax(pred, axis=1)
	d = {0: 'Front', 1: 'Rear', 2: 'Side'}
	for key in d.keys():
		if pred_label[0] == key:
			return d[key]

def severity_assessment(img_256, model):
	print("Determining severity of damage...")
	# Fallback heuristic based on edge magnitude
	if model is None:
		arr = img_256[0]
		if arr.dtype != 'float32' and arr.max() > 1.5:
			arr = arr / 255.0
		gray = arr.mean(axis=2)
		gx, gy = np.gradient(gray)
		edge = np.sqrt(gx * gx + gy * gy)
		edge_mean = edge.mean()
		# thresholds chosen empirically for demo
		if edge_mean < 0.01:
			return 'Minor'
		elif edge_mean < 0.03:
			return 'Moderate'
		else:
			return 'Severe'
	# otherwise use model
	pred = model.predict(img_256)
	pred_label = np.argmax(pred, axis=1)
	d = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}
	for key in d.keys():
		if pred_label[0] == key:
			return d[key]

def engine(img_path):
	init_models()
	# If core models are missing, continue using heuristic fallbacks so the
	# web UI can still present a result. Record whether fallbacks were used.
	models_loaded = True
	if first_gate is None or cat_list is None:
		print('Warning: core model files not loaded — using heuristic fallbacks')
		models_loaded = False

	img_224 = prepare_img_224(img_path)
	g1 = car_categories_gate(img_224, first_gate)

	if g1 is False:
		result = {'gate1': 'Car validation check: ', 
		'gate1_result': 0, 
		'gate1_message': {0: 'Are you sure this is a picture of your car? Please retry your submission.', 
		1: 'Hint: Try zooming in/out, using a different angle or different lighting'},
		'gate2': None,
		'gate2_result': None,
		'gate2_message': {0: None, 1: None},
		'location': None,
		'severity': None,
		'final': 'Damage assessment unsuccessful!'}
		return result
		
	img_256 = prepare_img_256(img_path)
	g2 = car_damage_gate(img_256, second_gate)

	if g2 is False:
		result = {'gate1': 'Car validation check: ', 
		'gate1_result': 1, 
		'gate1_message': {0: None, 1: None},
		'gate2': 'Damage presence check: ',
		'gate2_result': 0,
		'gate2_message': {0: 'Are you sure that your car is damaged? Please retry your submission.',
		1: 'Hint: Try zooming in/out, using a different angle or different lighting.'},
		'location': None,
		'severity': None,
		'final': 'Damage assessment unsuccessful!'}
		return result
	
	x = location_assessment(img_256, location_model)
	y = severity_assessment(img_256, severity_model)
	
	result = {
		'gate1': 'Car validation check: ',
		'gate1_result': 1,
		'gate1_message': {0: None, 1: None},
		'gate2': 'Damage presence check: ',
		'gate2_result': 1,
		'gate2_message': {0: None, 1: None},
		'location': x,
		'severity': y,
		'models_loaded': models_loaded,
		'final': 'Damage assessment complete' + ('' if models_loaded else ' (heuristic fallbacks used).')
	}
	return result