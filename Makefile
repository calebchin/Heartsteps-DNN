default:

black_format:
	@black src/dnn.py
	@black src/nnimputer.py
	@black src/wasserstein_nn.py
	@black src/kernel_nn.py