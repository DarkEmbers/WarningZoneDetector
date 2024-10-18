from torch import nn

class KeypointPredictor(nn.Module):
	def __init__(self):
		super(KeypointPredictor, self).__init__()
		self.seq = nn.Sequential(
			# nn.Linear(8, 8),
			# nn.ReLU(),
			nn.Linear(8, 4)
		)

	def forward(self, x):
		return self.seq(x)