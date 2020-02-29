import skeltorch
from .data import GlowData
from .runner import GlowRunner

# Create Skeltorch object
skel = skeltorch.Skeltorch(
    GlowData(),
    GlowRunner()
)

# Run Skeltorch project
skel.run()