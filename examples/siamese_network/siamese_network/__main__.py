import skeltorch
from .data import SiameseData
from .runner import SiameseRunner

# Create and run Skeltorch project
skeltorch.Skeltorch(SiameseData(), SiameseRunner()).run()
