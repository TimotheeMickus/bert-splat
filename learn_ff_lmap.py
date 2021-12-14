import torch, pickle
import numpy as np
import sklearn.metrics
import sklearn.linear_model
import sklearn.preprocessing
import tqdm

def yield_items(layer=12):
	layer = layer -1
	def yield_source_target_pairs():
		try:
			with open('ff_sole.pkl', 'rb') as istr:
				while True:
					src, tgt = pickle.load(istr), pickle.load(istr)
					yield src, tgt
		except EOFError:
			pass
	examples_stream = yield_source_target_pairs()
	examples_stream = enumerate(examples_stream)
	examples_stream = filter(lambda p: (p[0] % 12) == layer, examples_stream)
	examples_stream = map(lambda p: (p[1][0].squeeze(0).numpy(), p[1][1].squeeze(0).numpy()), examples_stream)
	yield from examples_stream
	
for layer in range(1, 13):
	all_src, all_tgt = [], []
	for a, b in tqdm.tqdm(yield_items(layer=layer), desc=f'read (layer={layer})', total=10_000):
		all_src.append(a)
		all_tgt.append(b)

	all_src = np.vstack(all_src)
	all_tgt = np.vstack(all_tgt)
	
	all_src = sklearn.preprocessing.StandardScaler().fit_transform(all_src)
	all_tgt = sklearn.preprocessing.StandardScaler().fit_transform(all_tgt)
	
	reg = sklearn.linear_model.LinearRegression(n_jobs=-1)
	reg.fit(all_src, all_tgt)
		
	print(layer, reg.score(all_src, all_tgt))
