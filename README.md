# NeuralNetwork-ML

Implementazione didattica di una rete neurale feed forward con backpropagation, regolarizzazione L2, momentum e mini-batch. Include esempi per classificazione, regressione e notebook di supporto.

## Contenuto
- `neural/`: strati fully connected, attivazioni (linear, logistic, tanh, relu) e classi `Network`, `Classifier`, `Regressor`.
- `metrics.py`: confusion matrix e metriche base (accuracy, precision, recall, f1) con storico dell'accuracy.
- `datasets/`: CSV per i dataset Monk's e ML-CUP gia inclusi.
- `test/`: script demo `breast_cancer.py` e `monk.py`.
- `notebooks/`: `neuron.ipynb`, `monk.ipynb` e utility di grafico in `plotting.py`.
- `requirements.txt`, `pyproject.toml`: dipendenze e metadata del pacchetto `neural`.

## Installazione
Consigliato Python 3.10 o superiore.

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Uso rapido
Esempio di classificazione binaria:

```py
import numpy as np
from neural.network import Classifier

X = np.random.rand(200, 2)
y = (X.sum(axis=1) > 1).astype(int)

net = Classifier(
    hidden_layer_sizes=(8,),
    activation="tanh",
    learning_rate=0.01,
    batch_size=20,
    max_iter=300,
)

net.fit(X, y)
pred = net.predict(X)
```

Valutazione con le metriche incluse:

```py
from metrics import Metrics

mt = Metrics(metrics=["accuracy"])
mt.compute_results(y, pred)
print("accuracy:", mt.accuracy_score())
```

Regression:

```py
from neural.network import Regressor
reg = Regressor(hidden_layer_sizes=(8,), activation="tanh", learning_rate=0.01, max_iter=300)
reg.fit(X_train, y_train)
y_hat = reg.predict(X_test)
```

## Eseguire gli esempi
Lanciare gli script dalla radice del repository:

```bash
python test/breast_cancer.py
python test/monk.py 1   # dataset Monk's 1 (usa 1, 2 oppure 3)
```

I grafici di loss e accuratezza vengono mostrati a video.

## Notebook
Per esplorare gli esperimenti interattivi:

```bash
jupyter lab notebooks/neuron.ipynb
# oppure notebooks/monk.ipynb
```

Usare il venv attivato prima di avviarli.
