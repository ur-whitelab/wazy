from functools import partial # for use with vmap
import jax
import jax.numpy as jnp
import haiku as hk
import jax.scipy.stats.norm as norm
import optax
from jax_unirep.layers import AAEmbedding, mLSTM, mLSTMAvgHidden
from jax_unirep.utils import load_params, load_embedding, seq_to_oh
from jax_unirep.utils import *
from jax_unirep import get_reps
import matplotlib.pyplot as plt

# create a random seed
key = jax.random.PRNGKey(0)

def forward(x):
    mlp = hk.nets.MLP([1900, 256, 32, 2])
    return mlp(x)

forward  = hk.transform(forward)

class MLP:

    def __init__(self, key, forward):
        self.key = key
        self.forward = forward

    def deep_ensemble_loss(self, params, ins, labels):
        outs = self.forward.apply(params, self.key, ins)
        means = outs[0]
        stds = outs[1]
        n_log_likelihoods = 0.5*jnp.log(jnp.abs(stds)) + 0.5*(labels-means)**2/jnp.abs(stds)

        return n_log_likelihoods[0]

    def adv_loss_func(self, params, seqs, labels, loss_func):
        epsilon = 1e-3
        grad_inputs = jax.grad(loss_func, 1)(params, seqs, labels)
        seqs_ = seqs + epsilon * jnp.sign(grad_inputs)

        return loss_func(params, seqs, labels) + loss_func(params, seqs_, labels)

    def train_mlp(self, seqs, labels):
        learning_rate = 1e-2
        n_training_steps = 25
        opt_init, opt_update = optax.chain(
            optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-4),
            optax.scale(-learning_rate) # minus sign -- minimizing the loss
        )
        self.key, key_ = jax.random.split(self.key, num=2)
        self.params = self.forward.init(self.key, jax.random.normal(key_, shape=(1900,)))
        opt_state = opt_init(self.params)

        loss_trace=[]
        for step in range(n_training_steps):
            loss, grad=jax.value_and_grad(self.adv_loss_func)(self.params, seqs, labels, self.deep_ensemble_loss)
            loss_trace.append(loss)

            updates, opt_state = opt_update(grad, opt_state, self.params)
            self.params = optax.apply_updates(self.params, updates)
        outs = self.forward.apply(self.params, self.key, seqs)

        #joint_outs = model_stack(outs)
        return loss_trace, outs

    def batch(self, seqs, labels):
        self.ensemble_seqs = jnp.tile(seqs, (5, 1 ,1))
        self.ensemble_labels = jax.lax.broadcast(labels, (5,))[...,jnp.newaxis]
        self.b_training_mlp = jax.vmap(self.train_mlp, (0, 0), (0, 0))
        self.bb_training_mlp = jax.vmap(self.b_training_mlp, (0, 0), (0, 0))


    def model_stack(self):
        mu = jnp.mean(self.outs[..., 0], axis=0)
        std = jnp.mean(self.outs[...,1] + self.outs[...,0]**2,axis=0) - mu**2
        return mu, std

    def call_train(self):
        self.loss_trace, self.outs = self.bb_training_mlp(self.ensemble_seqs, self.ensemble_labels) # batched
        self.joint_outs = self.model_stack()


    def bayesian_ei(self):
        mu, std = self.joint_outs
        best = jnp.max(mu)
        epsilon = 0.1
        z = (mu-best-epsilon)/std
        return (mu-best-epsilon)*norm.cdf(z) + std*norm.pdf(z)


seqs = ['MSADDGIVYCQMRQGPWEFHIVTVESSAYDWVVVPGARIALDKYNAACEQHWSCILRRGIDQKPYAPDMLKCQCSDMCHPSDSFTWEIDAEAWYCNTDNLFTGIALYKNNDDYPDWYPIRCLKHKNVTAAQVPLVHFNDNKFTHHVHNDMPACDFKFFKTPTVRHACQFGSIYHSKQSRMDYSDLMQDEKAKHLKESHNVVPDDGIIIDPYMDILFGGRMNNREHCAKNE',
        'EKMHIKESATRMGFQYEYKLPYCIWAFIIGRAWHFVSLHGDQWDCWKMTFVIYSACSNGHIDGCEVQHANLSSGVLPARWFDAFQQNMKGFHKMKCGGFCTYAFLWGLAMRIYVRNMGNLAIYQNGGTSEWLTEFWYRLAGAVWPFKQFSINGECEHFWWSFHPFTLFDNPPAKDRNVTAYLHFDAHFYSIAMVWLMSPVVKGDSPVNCCAVDVEQSGESWALLNNWCAP',
        'HSFHKYKHGNWKSEGDQCLKVGQLRDECPQVNTPMYCSWGPHYFSIFHWIIPVAKAYHMLHNIEQQVYRCHWQERYKELHDATKTHQLEWSFGKSVWCAHCKPYIGWYRSPAGWHMPIKPPATKNLWVVRHKSKRKEGTISWENTLTCVWFHEICYGHGVCHQVHPWVVDSNEEYEMQWMETEVGECSYPAERQGAWYSFTQQQKWICIHVCNMSSGRVFCWYVLQLFRN',
        'LDHAVLKILQAMGPWNNRVEHPRLGKRSTEWPAAIYEGEPRWRLKCDTTATYYKAFETRWYNCHMTLTCWWHGATIRSKLTTMCMMVTNGYRDFYRYNDWKGRKATKHHPMVCIYEILWIAFMGCLHMWAGARVSKIWVGFCIFFASCLQMSPLKDWHNKCAFGRNNPLGMKGWGMMIGNSFCHIVHEMDNKYYAGAPVDEPFMYNQQVFGFGAMHCLCMADFCNEWGIQ',
        'PERHHYIGFHCYMQLDAIQQNPHWNAHVLFRAFDYVSNYWTWITMYDKYQGFLGIYVTSCKVHEHGACKHCHWPICYDCGQHADKMLWRKSFALHGQSHAYRPLWDRDLTGVLGISIDLNQGIKVAEAEGEILYCNVTDMTVMMHQSVGVFWCHDMAYPQWTDWYSSDNMMNSIPEISHMKNYRVTMVHEPLFIWECVSEWTENAEHEGHLITVGSTGGKWDTGMEREVM',
        'DPSQTIHCGTTGMSWGTMFKRSYILIIRYGTPEATCPCIVNCQIVVYWGCMFKKDRDPRGTPIQSTENFFKHAMMEPSYAGGTAHMEKEIEYRSQDSWHAYFSYWVKVWCYVCIALSQIPNVAHHGMHLHASPEDKKCANNWRFRYVAFIRIAHGCSWCYRECYNFRYDRYIAWNPVHLESVPEWWAHPAFEIVKDTVDDNQYSGADERQGDPIGGQPCLLCATWEDSWT',
        'LIDLFSLTRKFSRMPCRHNMNESYKEEWCETNNVKEYPHEQLLDKRYDIITLDGCKRMYCRSESQRITELHFIRYNMLCWPDRCIPLQYSQYESNMPPPMMRMWGCYHFGTLMFMSYAMPPTGEKREIVGGKEDHSGDLEDAFTDEDFNMDPAHQDYRHIAGTWHEPMFEIRMRYELTCNNMWSPIYANNAGMKQLTICNNDKICPTEGRRRQREIFNYKLHGRDQCQHI',
        'SCDVGPHPLHGQCTGMAKQVMETANIPQCPIDDHVTRATMGLIDAGACDRDRVCVREIWNVYYDKSTMKIIMDPPSDTCKHKSFYGDMMSHQQMGWLSECIIANMQHNLPWQLWESWMIHSEICMIKQRKVMMFCGIQSKYTEDFARFHPFILANTQYIIFKRPTTWPRVYAFLHRCMVLGWSAYGMTAMIPNTKETIKLAHCEKWPLTGSYTPSFVIFDGWLARKCQWP',
        'DWIEHVHTFWVLMFISNYPQIVCGLINQIEPWKSKFHSLAGFNQGCQCEKNYQGPIQAINGINQLVTITTPINNQENVDKKPHPGSVHTKSDAITLRFNQGVHNIFMWDMATQGRASIPFLNNMNGGGLTDYSWEQVVTCHCHMTNDLELDPQMLYMWWIVSANAWMVNGMRRQHMACHWAQWEGFRWPRYVQSVPMKVLLTTQKIHWMQYFREKFCFILMKWQGYWYTV',
        'RHWRAPLLMYRDKEVQITWHFRFMYHCDALTCSEVHCHARNFMVFGYSTPQNYNPVILYWVTWANTCLTPKGAYCARQMRMYATVTMSKINQMTITYLVDRQRQHWGLAFRSDNTCNHKWYLKHRCKVWNWGWLIDCYDLDRNLPKQVSRNQSSKSLRDLFNYIHYHWAMLPINIYCYSGDIWTTISTDDQFHIPTFIPCGKTVHEDLQPYEMCGMWHQCEDADYTMQPV',
]
labels = jnp.array([25.217391304347824,
                    15.652173913043478,
                    23.478260869565219,
                    22.173913043478262,
                    23.913043478260871,
                    24.782608695652176,
                    26.956521739130434,
                    17.391304347826086,
                    19.130434782608695,
                    26.521739130434781
                   ])
seqs = get_reps(seqs)[0]

model = MLP(key, forward)
model.batch(seqs, labels)
model.call_train()

print(model.joint_outs)

    

    



    





    


