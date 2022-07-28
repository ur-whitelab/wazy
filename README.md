# ALP-Design

Peptide Design / Active Learning

## installing

```bash
pip install alpdesign@git+https://github.com/ur-whitelab/ALP-Design
```

## Quickstart
You can use an ask/tell style interface to design a peptide.

We can tell a few examples of sequences we know and their scalar labels.

```py
import alpdesign
import jax
key = jax.random.PRNGKey(0)
boa = alpdesign.BOAlgorithm()

boa.tell(key, "GGGG", -2.2)
boa.tell(key, "GAGG", -1.8)
boa.tell(key, "GAAG", 3.1)
```

Now we can predict on new values. This will return both a predicted label and its uncertainty.

```py
boa.predict(key, "GAAG", 3.1)
```

Then we can use Bayesian optimization to predict which sequence to try next.

```py
boa.ask(key)
```

and optionally we can specify desired sequence length
```py
boa.ask(key, 3)
```

## Files

* `alpdesign` source code
* `tests` contains unit tests. Run them with `pytest -s` after installing `pytest`
* `paper` contains jupyter notebooks which show figures from paper. Dependencies for those are in `paper/requriements.txt`
