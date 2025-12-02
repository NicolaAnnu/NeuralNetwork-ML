# Neural Network

Simple neural network implemented in Python for master degree machine learning
course.

## Installation and Development

In order to use the library and the notebooks it's suggested to work in a
virtual environment. On Linux is possible to run the following script

```bash
source ./scripts/setup.sh
```

while on Windows is possible to run

```bash
./scripts/setup.bat
```

Both will create the environment, download packages, install the `neural`
package containing the network implementation and activate the virtual
environment.

For a more manual way is possible to follow these equivalent instructions

```bash
python -m venv .env # to create it
source .env/bin/activate # to enable it on Linux
.env/Scripts/activate.bat # to enable it on Windows
```

then it's necessary to install the requirements in `requirements.txt` via `pip`

```bash
pip install -r requirements.txt
```

lastly you can install the network library with this command

```bash
pip install -e .
```

that install what's inside the `neural` folder once and automatically detect
changes, so that is not necessary to reinstall it every time.

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
