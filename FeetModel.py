from torch import nn

class KeypointPredictor(nn.Module):
	def __init__(self):
		super(KeypointPredictor, self).__init__()
		self.seq = nn.Sequential(
			nn.Linear(8, 64),
			nn.ReLU(),
			nn.Linear(64, 128),
			nn.ReLU(),
			nn.Linear(128, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 4)
		)

	def forward(self, x):
		return self.seq(x)