# Neural Network

Simple neural network implemented in Python for master degree machine learning
course.

## Installation and Development

In order to use the library and the notebooks it's suggested to work in a
virtual environment:

```bash
python -m venv .env # to create it
source .env/bin/activate # to enable it on Linux
.env/Scripts/activate.bat # to enable it on Windows
```

Then it's necessary to install some dependency in `requirements.txt` via `pip`

```bash
pip install -r requirements.txt
```

Lastly is possible to install the `neural` library with this command

```bash
pip install -e .
```

that installs what's inside the `neural` folder once and automatically detect
changes, so that is not necessary to reinstall it every time while developing.

Now the library is visible from any location inside the project folder and so is
possible to use also inside notebooks like:

```py
from neural.network import Classifier

model = Classifier()
model.fit(X, y)
predictions = model.predict(X)
```

**NOTE**: Some packages like tensorflow and pytorch will not be downloaded by
default due to their size and possibly high download time. The choice to install
them is left to the user.
