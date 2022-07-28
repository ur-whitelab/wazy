# ALP-Design

Peptide Design / Active Learning

## installing

```bash
pip install alpdesign@git+https://github.com/ur-whitelab/ALP-Design
```

## Quickstart
You can use an ask/tell style interface to design a peptide.

We can tell a few examples of sequences we know and their scalar labels. Let's try a simple example where the label is the number of alanines. We'll start by importing and building a `BOAlgorithm` class.

```py
import alpdesign
import jax
key = jax.random.PRNGKey(0)
boa = alpdesign.BOAlgorithm()
```

Now we can tell it a few examples.
```py
boa.tell(key, "GGGG", 0)
boa.tell(key, "GAHK", 1)
boa.tell(key, "DAAE", 2)
boa.tell(key, "DAAA", 3)
boa.tell(key, "DRRK", 0)
```

We can predict on new values. This will return both a predicted label and its uncertainty and its epistemic uncertainty.

```py
boa.predict(key, "LPAH")
# Output:
(0.33622107, 12.499498, 2.0541081)
```
The accuracy is poor - $0.33\pm 12.5$. Let's now use Bayesian optimization to choose which sequence to try next:

```py
boa.ask(key)
# Output
'FAVL', -0.01533653
```
The first value is the sequence to try next. The second is an indicator in how valuable (value of acquisition function) it finds that sequence. Now we can tell it the value and see if it improves the model predictions:

```py
boa.tell(key, "FAVL", 1)
boa.predict(key, "LPAH")
# Output
0.89115465, 0.36383766, 0.04996746
```
Which is much closer to the true answer of 1!

## Files

* `alpdesign` source code
* `tests` contains unit tests. Run them with `pytest -s` after installing `pytest`
* `paper` contains jupyter notebooks which show figures from paper. Dependencies for those are in `paper/requriements.txt`
