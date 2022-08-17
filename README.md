# wazy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ur-whitelab/wazy/blob/master/colab/Wazy.ipynb)

Pretrained Bayesian Optimization of Amino Acid Sequences

## installing

```bash
pip install wazy
```

## Quickstart

You can use an ask/tell style interface to design a peptide.

We can tell a few examples of sequences we know and their scalar labels. Let's try a simple example where the label is the number of alanines. You'll also want your labels to vary from about -5 to 5. We'll start by importing and building a `BOAlgorithm` class. *In this example, I re-use the same key for simplicity.*

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
(5.823452, 69.99278, 24.500998)
```

The accuracy is poor - $5.8\pm 70$. Let's now use Bayesian optimization to choose which sequence to try next:

```py
boa.ask(key)
# Output
('DAAV', 6.901945)
```

The first value is the sequence to try next. The second is an indicator in how valuable (value of acquisition function) it finds that sequence. Now we can tell it the value:

```py
boa.tell(key, "DAAV", 2)
```

We can also choose the sequence length:

```py
boa.ask(key, length=6)
# Output
('DAAATA', 5.676821)
```

We can try our new prediction to see if it improved.

```py
boa.tell(key, "DAAATA", 4)
boa.predict(key, "LPAH")
# Output
(2.0458677, 13.694655, 1.0933837)
```

Which is indeed closer to the true answer of 1. Finally, we can ask for the best sequence:

```py
boa.ask(key, "max", length=5)
# Output
('DAAAA', 3.8262398)
```

### Batching

You can increase the number of returned sequences by using the `batch_ask`, which uses an ad-hoc regret minimization strategy to spread out the proposed sequences:

```py
boa.batch_ask(key, N=3)
# returns 3 seqs
```

and you can add a multiplier to batch sequences (no overhead), but they may be similar

```py
boa.batch_ask(key, N=3, return_seqs = 10)
# returns 30 seqs
```

## Citation

Please cite [Yang et. al.](https://www.biorxiv.org/content/10.1101/2022.08.05.502972v1.abstract)

```bibtex
@article{yang2022now,
  title={Now What Sequence? Pre-trained Ensembles for Bayesian Optimization of Protein Sequences},
  author={Yang, Ziyue and Milas, Katarina A and White, Andrew D},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}

```
