import skeltorch
from .data import MNISTClassifierData
from .runner import MNISTClassifierRunner

skeltorch.Skeltorch(MNISTClassifierData(), MNISTClassifierRunner()).run()
