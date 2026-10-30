# Language models

Code

- [Show All Code](javascript:void(0))
- [Hide All Code](javascript:void(0))

Published

Last modified: 2026-10-30

Show code

``` downlit
# The Python chunks on this page import these; reticulate provisions them.
# Attaching reticulate gives the inline R its `py` object.
library(reticulate)
source("../tools/use-torch.R")
use_torch(c("numpy"))
```

Brian Hutchinson’s CSCI 481/581 sets the order of topics in these notes, which follow it with his permission; this page covers the part of his module on recurrent networks that deals with language modeling. The exercises are new.

# 1 Welcome

## 1.1 Plan for this lecture

The [recurrent networks chapter](../chapters/rnns.llms.md) read whole sequences and classified them, and ended by turning symbols into vectors with an [embedding layer](../chapters/rnns.llms.md#def-embedding). This lecture predicts text one character at a time:

1.  the probability of a string, built from the probability of each next character;
2.  a model that counts pairs of characters;
3.  a number that scores how well a model predicts held-out text;
4.  a recurrent network that predicts each next character;
5.  drawing new text from that network.

## 1.2 The text

Show code

``` python
import math
import re

import torch
from torch import nn

text = open("data/shakespeare.txt").read()
chars = sorted(set(text))
n_chars = len(chars)
index = {ch: i for i, ch in enumerate(chars)}
ids = torch.tensor([index[ch] for ch in text])
n_train = 90_000
train_ids, val_ids = ids[:n_train], ids[n_train:]
seq_len = 64
batch_size = 32
lm_lr = 0.005


def decode(t):
    return "".join(chars[i] for i in t.tolist())


class CharModel(nn.Module):
    # Embeds each character, reads the sequence with a GRU,
    # and returns logits over the next character at every position.

    def __init__(self, embed_size=16, hidden_size=128):
        super().__init__()
        self.embed = nn.Embedding(n_chars, embed_size)
        self.recurrent = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, n_chars)

    def forward(self, x, state=None):
        states, state = self.recurrent(self.embed(x), state)
        return self.out(states), state


def make_lm(seed=1):
    torch.manual_seed(seed)
    return CharModel()


def train_lm(net, n_steps, seed=0):
    # Adam on random windows of the training text, with gradient clipping to 1.
    # Each window's targets are the same window moved one character on.
    optimizer = torch.optim.Adam(net.parameters(), lr=lm_lr)
    generator = torch.Generator().manual_seed(seed)
    net.train()
    for step in range(n_steps):
        starts = torch.randint(0, n_train - seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([train_ids[s:s + seq_len] for s in starts])
        y = torch.stack([train_ids[s + 1:s + seq_len + 1] for s in starts])
        logits, _ = net(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, n_chars), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        _ = nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()


def lm_log_probs(net):
    # Natural log of the probability `net` gives each validation character
    # after the first, given all the validation characters before it.
    net.eval()
    with torch.no_grad():
        logits, _ = net(val_ids[:-1].unsqueeze(0))
    log_p = torch.log_softmax(logits[0], dim=1)
    return log_p[torch.arange(len(val_ids) - 1), val_ids[1:]]


def sample(net, prompt, n, temperature=1.0, seed=0):
    # Feeds `prompt`, then draws `n` characters one at a time,
    # feeding each one back in.
    generator = torch.Generator().manual_seed(seed)
    x = torch.tensor([[index[ch] for ch in prompt]])
    drawn = []
    net.eval()
    with torch.no_grad():
        logits, state = net(x)
        for _ in range(n):
            p = torch.softmax(logits[0, -1] / temperature, dim=0)
            nxt = torch.multinomial(p, 1, generator=generator)
            drawn.append(nxt.item())
            logits, state = net(nxt.view(1, 1), state)
    return prompt + decode(torch.tensor(drawn))


print(n_chars, len(text))
```

    61 100000

Show code

``` python
print(text[:120])
```

    First Citizen:
    Before we proceed any further, hear me speak.

    All:
    Speak, speak.

    First Citizen:
    You are all resolved ra

The exercises use the first 100,000 characters of the “tiny Shakespeare” text, which Karpathy [\[1\]](#ref-karpathy2015charrnn) collected from Shakespeare’s plays; this slice is the opening of *Coriolanus*. It has 61 distinct characters, counting capitals, punctuation, the space and the line break separately. `ids` numbers each character by its place in the sorted list `chars`. The first 90,000 characters are for training, and the remaining 10,000 for validation.

`CharModel` reads a sequence of character indices with an [embedding layer](../chapters/rnns.llms.md#def-embedding) and a [GRU](../chapters/rnns.llms.md#def-gru) layer, and maps the [hidden state](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-hidden-state) at each position to [logits](../chapters/feedforward-networks.llms.md#def-logits) over the next character. `train_lm()` runs [Adam](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-adam) with [learning rate](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-learning-rate) 0.005 on batches of 32 windows of 64 characters, with [gradient clipping](https://morrison-lab.github.io/lds/chapters/backpropagation.html#def-gradient-clipping) to 1. `lm_log_probs()` returns the log of the probability the network gives each validation character, given the ones before it, and [`sample()`](https://rdrr.io/r/base/sample.html) draws new text from the network one character at a time.

# 2 Counting

## 2.1 The probability of a string

> **NOTE:**
>
> **Exercise 1 (The probability of a word)** Let \\X_1, X_2, X_3\\ be the next three characters of a text. Use the [chain rule of probability](https://morrison-lab.github.io/lds/chapters/probability.html#thm-chain-rule) to write \\\Pr(X_1 = \texttt{t}, X_2 = \texttt{h}, X_3 = \texttt{e})\\ as a product of three probabilities, each about one character.

> **NOTE:**
>
> *Solution 1*. The [chain rule of probability](https://morrison-lab.github.io/lds/chapters/probability.html#thm-chain-rule) conditions each character on the ones before it:
>
> \\\begin{aligned} &\Pr(X_1 = \texttt{t}, X_2 = \texttt{h}, X_3 = \texttt{e}) \\ &\quad = \Pr(X_1 = \texttt{t}) \Pr(X_2 = \texttt{h} \mid X_1 = \texttt{t}) \Pr(X_3 = \texttt{e} \mid X_1 = \texttt{t}, X_2 = \texttt{h}). \end{aligned}\\
>
> Each factor is the probability of one character given the characters before it. A model that gives these conditional probabilities for every string of earlier characters gives the probability of every string.

> **NOTE:**
>
> **Definition 1 (Language model)** A **language model** gives, for each position \\t\\ in a sequence of symbols \\x_1, \dots, x_n\\, a probability distribution \\\Pr(X_t = x \mid x_1, \dots, x\_{t-1})\\ over the symbol \\x\\ at that position, given the symbols before it [\[2, Sec. 12.4\]](#ref-goodfellow2016deep). By the [chain rule of probability](https://morrison-lab.github.io/lds/chapters/probability.html#thm-chain-rule), the product of these probabilities over \\t = 1, \dots, n\\ is the probability of the whole sequence. The symbols can be characters or words; these notes use characters.

> **NOTE:**
>
> **Example 1 (A uniform language model)**  
>
> Show code
>
> ``` python
> uniform_the = (1 / n_chars) ** 3
> print(uniform_the)
> ```
>
>     4.40565509888493e-06
>
> The simplest [language model](#def-language-model) for this text gives each of the 61 characters probability \\1 / 61\\ at every position, whatever came before. It gives `"the"` probability \\(1 / 61)^3\\, or 1 in 226,981, the same as any other string of three characters, such as `"qzx"`.

## 2.2 Counting pairs

> **NOTE:**
>
> **Exercise 2 (What follows a q)** In the training text, what share of the `q`s are followed by a `u`?

> **NOTE:**
>
> *Solution 2*.
>
> Show code
>
> ``` python
> after_q = train_ids[1:][train_ids[:-1] == index["q"]]
> n_q = len(after_q)
> share_u = (after_q == index["u"]).float().mean().item()
> print(n_q, decode(after_q))
> ```
>
>     32 uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu
>
> 100% of the 32 `q`s in the training text are followed by a `u`. So the share of `q`s followed by `u` estimates \\\Pr(X_t = \texttt{u} \mid X\_{t-1} = \texttt{q})\\ as 1.00.

> **NOTE:**
>
> **Definition 2 (Bigram model)** A **bigram model** is a [language model](#def-language-model) in which the distribution of each symbol depends only on the symbol before it [\[2, Sec. 12.4.1\]](#ref-goodfellow2016deep). With \\K\\ possible symbols, it estimates
>
> \\ \Pr(X_t = b \mid X\_{t-1} = a) = \frac{N(a, b) + 1}{N(a) + K}, \\
>
> where \\N(a, b)\\ counts how often \\a\\ is followed by \\b\\ in a training text, and \\N(a)\\ counts how often \\a\\ is followed by any symbol. The added 1 in each count, called add-one or Laplace smoothing [\[3, Sec. 3.6\]](#ref-jurafsky2026slp), gives every pair a probability above 0, including pairs the training text never shows.

> **NOTE:**
>
> **Example 2 (A bigram model of the training text)**  
>
> Show code
>
> ``` python
> pair_counts = torch.ones(n_chars, n_chars)
> _ = pair_counts.index_put_(
>     (train_ids[:-1], train_ids[1:]), torch.ones(n_train - 1), accumulate=True
> )
> bigram = pair_counts / pair_counts.sum(dim=1, keepdim=True)
> p_u_after_q = bigram[index["q"], index["u"]].item()
> after_t = bigram[index["t"]]
> top_after_t = chars[after_t.argmax().item()]
> p_top_after_t = after_t.max().item()
> print(p_u_after_q, repr(top_after_t), p_top_after_t)
> ```
>
>     0.35483869910240173 'h' 0.32821977138519287
>
> `pair_counts` starts every count at 1 and adds the training pairs, so row \\a\\ of `bigram` is the [bigram model](#def-bigram-model)’s distribution of the character after \\a\\. Its most likely character after `t` is h, with probability 0.33. It gives a `u` after a `q` probability 0.35, against the share of 1.00 in [Solution 2](#sol-after-q). There are only 32 `q`s to count, so the 61 added ones are a large part of the row.

## 2.3 Pairs never seen

> **NOTE:**
>
> **Exercise 3 (Pairs the training text never shows)** How many of the pairs of neighbouring characters in the validation text never occur in the training text?

> **NOTE:**
>
> *Solution 3*.
>
> Show code
>
> ``` python
> seen = pair_counts > 1
> n_unseen = (~seen[val_ids[:-1], val_ids[1:]]).sum().item()
> print(n_unseen)
> ```
>
>     12
>
> 12 of the 9,999 pairs never occur in the training text. Without the added ones, the [bigram model](#def-bigram-model) would give each of them probability 0, and so the whole validation text probability 0.

# 3 Scoring a language model

## 3.1 Average log probability

> **NOTE:**
>
> **Exercise 4 (Scoring held-out text)** For each validation character after the first, take the natural log of the probability the [bigram model](#def-bigram-model) gives it, given the character before it. What is the average of the negatives of these logs?

> **NOTE:**
>
> *Solution 4*.
>
> Show code
>
> ``` python
> bigram_log_probs = torch.log(bigram[val_ids[:-1], val_ids[1:]])
> bigram_nll = -bigram_log_probs.mean().item()
> uniform_nll = math.log(n_chars)
> print(bigram_nll, uniform_nll)
> ```
>
>     2.3827016353607178 4.110873864173311
>
> The average is 2.38. The uniform model of [Example 1](#exm-language-model) gives every character probability \\1 / 61\\, so its average is \\\log 61 = 4.11\\. The lower average means the [bigram model](#def-bigram-model) gives the validation text higher probability. This average is the [cross-entropy loss](https://morrison-lab.github.io/lds/chapters/logistic-regression.html#def-cross-entropy-loss) of the model’s predictions of the validation characters.

> **NOTE:**
>
> **Definition 3 (Perplexity)** The **perplexity** of a [language model](#def-language-model) on a sequence \\x_1, \dots, x_n\\ is
>
> \\ \expf{-\frac{1}{n} \sum\_{t=1}^{n} \log \Pr(X_t = x_t \mid x_1, \dots, x\_{t-1})}, \\
>
> the exponential of the average negative log probability per symbol [\[3, Sec. 3.3\]](#ref-jurafsky2026slp). The DATA 471 notes define it for any classifier in [the same way](https://morrison-lab.github.io/lds/chapters/metrics.html#eq-perplexity), and find that a model which gives probability \\1/K\\ to each of \\K\\ symbols has perplexity \\K\\. Lower is better.

> **NOTE:**
>
> **Example 3 (Perplexity of the two counting models)**  
>
> Show code
>
> ``` python
> bigram_ppl = math.exp(bigram_nll)
> uniform_ppl = math.exp(uniform_nll)
> print(bigram_ppl, uniform_ppl)
> ```
>
>     10.834133238705657 61.00000000000001
>
> On the validation text after its first character, the uniform model has [perplexity](#def-perplexity) 61.0, the number of characters. The [bigram model](#def-bigram-model) has perplexity 10.8: on average it is as unsure of the next character as a model choosing evenly among about 11 characters.

# 4 A recurrent language model

## 4.1 Predicting every next character

> **NOTE:**
>
> **Exercise 5 (A recurrent language model)** Train `make_lm()` for 400 steps with `train_lm()`. Is its [perplexity](#def-perplexity) on the validation text lower than the [bigram model](#def-bigram-model)’s?

> **NOTE:**
>
> *Solution 5*.
>
> Show code
>
> ``` python
> lm = make_lm()
> untrained_ppl = math.exp(-lm_log_probs(lm).mean().item())
> train_lm(lm, 400)
> gru_ppl = math.exp(-lm_log_probs(lm).mean().item())
> print(untrained_ppl, gru_ppl)
> ```
>
>     62.06655063632067 5.192073376728205
>
> Show code
>
> ``` python
> # The prose below says the GRU beats the bigram model; stop the render if not.
> assert gru_ppl < bigram_ppl
> ```
>
> Yes. After training, the network’s [perplexity](#def-perplexity) on the validation text is 5.2, against 10.8 for the [bigram model](#def-bigram-model). Before training it was 62.1, close to the 61 of the uniform model. The bigram model sees one earlier character; the network’s [hidden state](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-hidden-state) can carry information from all of them.

> **NOTE:**
>
> **Definition 4 (Teacher forcing)** **Teacher forcing** trains a recurrent [language model](#def-language-model) on a text by feeding it the true text as input, and asking it at each position for the next symbol of the same text [\[2, Sec. 10.2.1\]](#ref-goodfellow2016deep). The targets are the inputs moved one position on. The model never reads its own predictions during training.

> **NOTE:**
>
> **Example 4 (Inputs and targets in `train_lm()`)**  
>
> Show code
>
> ``` python
> window_x = decode(train_ids[0:20])
> window_y = decode(train_ids[1:21])
> print(repr(window_x))
> ```
>
>     'First Citizen:\nBefor'
>
> Show code
>
> ``` python
> print(repr(window_y))
> ```
>
>     'irst Citizen:\nBefore'
>
> `train_lm()` uses [teacher forcing](#def-teacher-forcing). The two printed lines are the input and the target for a window that starts at the first character: the target is the input moved one character on. At each position, the [cross-entropy loss](https://morrison-lab.github.io/lds/chapters/logistic-regression.html#def-cross-entropy-loss) compares the network’s prediction with the next character of the true text, whatever the network predicted before.

# 5 Generating text

## 5.1 Sampling

> **NOTE:**
>
> **Exercise 6 (Writing new lines)** Draw 1,000 characters from the trained network of [Solution 5](#sol-gru-lm) with `sample(lm, "MENENIUS:\n", 1000)`. What share of the words in the result occur somewhere in the training text?

> **NOTE:**
>
> *Solution 6*.
>
> Show code
>
> ``` python
> train_words = set(re.findall(r"[a-z]+", text[:n_train].lower()))
>
>
> def real_word_share(s):
>     words = re.findall(r"[a-z]+", s.lower())
>     return sum(w in train_words for w in words) / len(words)
>
>
> drawn = sample(lm, "MENENIUS:\n", 1000)
> drawn_share = real_word_share(drawn)
> # The prose below says the draw has speaker names on lines of their own.
> assert re.search(r"^[A-Z]+:$", drawn[10:], re.MULTILINE)
> print(drawn[:300])
> ```
>
>     MENENIUS:
>     Well. be prest I am neme
>     MENy seoplin a diseeals:
>     He'd that in thriery ssy every more.
>
>     VALERIA:
>     In with ank untemt the payy, what wisds,
>     The ay opatues my proves to the aween his
>
>     MENIUS:
>     Why with one ourther it Rome he'll
>     Than stable
>     Let matters yourselave for tiee ageise their noblie be
>
> Show code
>
> ``` python
> print(drawn_share)
> ```
>
>     0.5668449197860963
>
> Counting a word as a run of letters, ignoring case, 57% of the words drawn occur in the training text. The rest are made-up words. Like the play, the draw puts names in capitals followed by a colon on lines of their own, beyond the one in the prompt.

> **NOTE:**
>
> **Definition 5 (Temperature)** Sampling at **temperature** \\T \> 0\\ draws the next symbol from the [softmax](https://morrison-lab.github.io/lds/chapters/multiclass-logistic-regression.html#def-softmax) of \\\vecf{z} / T\\, where \\\vecf{z}\\ is the vector of the model’s [logits](../chapters/feedforward-networks.llms.md#def-logits) [\[4\]](#ref-hinton2015distilling). At \\T = 1\\ this is the model’s own distribution. As \\T\\ falls toward 0 the distribution puts more of its weight on the most likely symbols, and as \\T\\ grows it approaches the uniform distribution.

> **NOTE:**
>
> **Example 5 (Drawing at three temperatures)**  
>
> Show code
>
> ``` python
> temperatures = [0.5, 1.0, 1.5]
> draws = {t: sample(lm, "MENENIUS:\n", 1000, temperature=t) for t in temperatures}
> shares = [real_word_share(draws[t]) for t in temperatures]
> print(shares)
> ```
>
>     [0.8787878787878788, 0.5668449197860963, 0.34545454545454546]
>
> Show code
>
> ``` python
> # The prose below says the share falls as the temperature rises.
> assert shares[0] > shares[1] > shares[2]
> print(draws[0.5][:200])
> ```
>
>     MENENIUS:
>     What he show the eward the consers the discess,
>     We all the city the wars your actrems.
>
>     VOLUMNIA:
>     I will you are my tribunes, and and to the say
>     That him such as live the well his shall more
>
> Show code
>
> ``` python
> print(draws[1.5][:200])
> ```
>
>     MENENIUS:
>     Well. but you sevewayts?
>
>     LARTWerpyis no.
>     Yeeals you ar kaquince?
>     N uf ssy every more.
>     Of power in reveraus asd vat thibp syreants fores,
>     Throw.
>
>     FilutherI: &oos,
>     Nor shear enVybelight sood.
>
> | [Temperature](#def-temperature) | Share of words found in the training text |
> |--------------------------------:|------------------------------------------:|
> |                             0.5 |                                       88% |
> |                             1.0 |                                       57% |
> |                             1.5 |                                       35% |
>
> At temperature 0.5 the network mostly writes words it has seen. At 1.5 most of its words are new, and the spelling breaks down.

## 5.2 Further reading

> **NOTE:**
>
> *Deep Learning*’s [section 12.4](https://www.deeplearningbook.org/contents/applications.html) covers \\n\\-gram and neural language models [\[2\]](#ref-goodfellow2016deep). Karpathy [\[5\]](#ref-karpathy2015unreasonable) trains character-level recurrent language models on Shakespeare, Wikipedia and source code, and shows what they write.

# 6 References

\[1\]

A. Karpathy, “Char-rnn: Multi-layer recurrent neural networks for character-level language models.” GitHub repository, 2015. Available: <https://github.com/karpathy/char-rnn>

\[2\]

I. Goodfellow, Y. Bengio, and A. Courville, *Deep learning*. MIT Press, 2016. Available: <https://www.deeplearningbook.org/>

\[3\]

D. Jurafsky and J. H. Martin, *Speech and language processing*, 3rd, draft. 2026. Available: <https://web.stanford.edu/~jurafsky/slp3/>

\[4\]

G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network.” 2015. Available: <https://arxiv.org/abs/1503.02531>

\[5\]

A. Karpathy, “The unreasonable effectiveness of recurrent neural networks.” Blog post, 2015. Available: <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>

Back to top
