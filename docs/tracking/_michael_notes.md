### Some useful conventions when reading this repo
- `__init__.py` just defines the interface and enables future users to interact with this package via `radar_forge.tracking`. `__all__` defines which features are available (i.e. when the user types `from radar_forge.tracking import *`)
- `_numerics.py` gives us some useful utilities which are used in other parts of the file
- `association.py` currently supports NN, GNN, and ChiSquared (Mahalanobis-distance?)
- `derived.py` enables us to compute some derived quantities not outputted by the filter, which supports the ability to track any arbitrary state space
- `sensors.py` contains definitions of sensors and also a "Measurements" class which refers to a measurement / detection from that sensor. `measurements.py` refer to the entire class of `MeasurementModel`s, which is why a detection (measurement) lives in sensors.py, not measurements.py.
filter = estimator

ctrl + click to view any function