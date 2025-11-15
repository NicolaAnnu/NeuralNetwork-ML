# Neural Network

Simple neural network implemented in Python for master degree machine learning
course.

## Installation and Development

In order to use the library and the notebooks it's suggested to work in a
virtual environment:

```bash
python -m venv .env # to create it
source .env/bin/activate # on Linux
.env/Scripts/activate.bat # on Windows
```

then it's necessary to install the requirements in `requirements.txt` via `pip`

```bash
pip install -r requirements.txt
```

lastly you can install the network library with this command

```bash
pip install -e .
```

that install what's inside the `network` folder once and automatically detect
changes, so that is not necessary to reinstall it every time.

Now the library is visible from any location inside the project folder and so is
possible to use also inside notebooks like:

```py
from network.neuron import Neuron

model = Neuron()
model.fit(X, y)
predictions = model.predict(X)
```
