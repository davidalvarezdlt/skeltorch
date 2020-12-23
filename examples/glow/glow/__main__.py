import skeltorch
from .data import GlowData
from .runner import GlowRunner

skeltorch.Skeltorch(GlowData(), GlowRunner()).run()
