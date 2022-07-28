# wazy

Pretrained Bayesian Optimization of Amino Acid Sequences

## installing

```bash
pip install wazy@git+https://github.com/ur-whitelab/wazy
```

## Quickstart
You can use an ask/tell style interface to design a peptide.

We can tell a few examples of sequences we know and their scalar labels. Let's try a simple example where the label is the number of alanines. We'll start by importing and building a `BOAlgorithm` class.

```py
import wazy
import jax
key = jax.random.PRNGKey(0)
boa = wazy.BOAlgorithm()
```

Now we can tell it a few examples.
```py
boa.tell(key, "GGGG", 0)
boa.tell(key, "GAHK", 1)
boa.tell(key, "DAAE", 2)
boa.tell(key, "DAAA", 3)
```

We can predict on new values. This will return both a predicted label and its uncertainty and its epistemic uncertainty.

```py
boa.predict(key, "LPAH")
# Output:
(2.892956, 17.946255, 0.02351546)
```
The accuracy is poor - $1.82\pm 12.5$. Let's now use Bayesian optimization to choose which sequence to try next:

```py
boa.ask(key)
# Output
'DKKA', -2.1462271
```
The first value is the sequence to try next. The second is an indicator in how valuable (value of acquisition function) it finds that sequence. Now we can tell it the value:

```py
boa.tell(key, "DKKA", 1)
```

We can also choose the sequence length:
```py
boa.ask(key, length=6)
# Output
'DVVSAW', -2.0747144
```

We can try our new prediction to see if it improved.

```py
boa.tell(key, "DVVSAW", 1)
boa.predict(key, "LPAH")
# Output
(1.5619717, 5.745885, 0.20795089)
```
Which is indeed closer to the true answer of 1. Finally, we can ask for the best sequence:

```py
boa.ask(key, "max")
# Output
('GRADRV', -0.9387406)
```

## Files

* `wazy` source code
* `tests` contains unit tests. Run them with `pytest -s` after installing `pytest`
* `paper` contains jupyter notebooks which show figures from paper. Dependencies for those are in `paper/requriements.txt`
