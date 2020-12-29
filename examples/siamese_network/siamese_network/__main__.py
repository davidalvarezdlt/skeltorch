import skeltorch
from .data import SiameseNetworkData
from .runner import SiameseNetworkRunner

skeltorch.Skeltorch(SiameseNetworkData(), SiameseNetworkRunner()).run()
