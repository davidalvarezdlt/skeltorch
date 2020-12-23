import skeltorch
from .data import SiameseData
from .runner import SiameseRunner

skeltorch.Skeltorch(SiameseData(), SiameseRunner()).run()
