---
order: 7
description: Combinatorics, graph theory, set theory, logic, recurrences, and complexity analysis — the discrete structures behind graph neural networks, tokenizers, sampling, and interview algorithm questions.
meta: Math for ML · supporting
---

# Discrete Mathematics: Counting, Graphs, and Structure

Continuous mathematics gives you gradients. Discrete mathematics gives you
*structure*: how many ways there are to do something, what connects to what, and
how long an algorithm will take. It shows up in ML more than people expect —
attention is a complete graph, a tokenizer is a greedy algorithm over a merge
lattice, beam search is a bounded tree search, and every "why is this $O(n^2)$?"
question about transformers is a discrete-maths question.

## Sets and relations

A **set** is an unordered collection of distinct elements. The operations —
union $\cup$, intersection $\cap$, difference $\setminus$, complement, and the
power set $2^S$ — are the vocabulary for talking about data.

Two facts that get used constantly:

- $|2^S| = 2^{|S|}$. The number of feature subsets grows exponentially, which is
  why exhaustive feature selection is impossible past ~20 features and why
  greedy/regularised methods exist.
- **Inclusion–exclusion**: $|A\cup B| = |A|+|B|-|A\cap B|$, generalising to
  alternating sums. This is exactly the computation behind the Jaccard index
  $J(A,B) = \frac{|A\cap B|}{|A\cup B|}$ used for deduplication, MinHash
  near-duplicate detection in pretraining corpora, and set-based retrieval
  metrics.

A **relation** on $S$ is a subset of $S\times S$. When it is reflexive,
symmetric, and transitive it is an **equivalence relation** and it partitions $S$
into disjoint classes — which is what clustering produces, what union-find
computes, and what "these two documents are duplicates" asserts.

A **partial order** (reflexive, antisymmetric, transitive) gives you a DAG, and
**topological sort** on that DAG is how autodiff decides the order to evaluate
nodes and how a build system decides what to compile first.

## Combinatorics: counting without enumerating

### The rules

| Rule | Statement | Example |
|---|---|---|
| Sum rule | disjoint choices add | 3 CNNs or 4 transformers → 7 architectures |
| Product rule | sequential choices multiply | 5 LRs × 3 batch sizes × 4 depths = 60 configs |
| Permutations | $P(n,k) = \frac{n!}{(n-k)!}$ | ordered top-$k$ rankings |
| Combinations | $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ | choosing a feature subset |
| With repetition | $n^k$ ordered, $\binom{n+k-1}{k}$ unordered | sequences of length $k$ over a vocab of $n$ |
| Multinomial | $\frac{n!}{n_1!\cdots n_k!}$ | arrangements with repeated items |

### Why this matters concretely

**Sequence space.** A vocabulary of 50,000 tokens and a context of 1,000 tokens
gives $50000^{1000} \approx 10^{4700}$ possible sequences. There are about
$10^{80}$ atoms in the observable universe. A language model cannot be a lookup
table; it *must* generalise. This counting argument is the cleanest one-line
justification for parametric models.

**Hyperparameter grids.** 6 hyperparameters at 5 values each is $5^6 = 15{,}625$
runs. At 2 GPU-hours each that is 31,250 GPU-hours. Random search and successive
halving are not laziness; they are the only options.

**Pairwise attention.** $\binom{n}{2}$ pairs among $n$ tokens is
$\frac{n(n-1)}{2} = O(n^2)$. Doubling context quadruples attention cost. Every
efficient-attention paper is an attempt to avoid computing all
$\binom{n}{2}$ interactions.

**Bagging.** Sampling $n$ items with replacement from $n$ leaves each item out
with probability $(1-1/n)^n \to e^{-1} \approx 0.368$. So each bootstrap sample
contains about 63.2% of the unique data, and the remaining 36.8% is the
**out-of-bag** set that random forests use for free validation. That number
falls straight out of a limit.

### Binomial coefficients and Pascal's identity

$$\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$$

Either you take element $n$ or you do not. This recurrence is the standard
warm-up for dynamic programming, and the same "include/exclude" decomposition
drives subset-sum, knapsack, and edit distance.

### Pigeonhole principle

$n$ items in $m < n$ boxes forces some box to hold at least $\lceil n/m\rceil$
items. Consequences you actually meet:

- **Hash collisions are unavoidable.** The hashing trick maps an unbounded
  feature space into $2^{20}$ buckets; collisions are guaranteed, and the
  practical claim is only that they are rare and roughly harmless.
- **Lossless compression cannot compress everything.** There are more $n$-bit
  strings than shorter ones.
- **Quantisation** maps $2^{16}$ float values into 256 int8 levels — the
  information loss is a pigeonhole certainty, and calibration is about choosing
  *which* collisions to accept.

The **birthday paradox** is the probabilistic version: among $k$ items drawn
from $N$ possibilities, a collision becomes likely at $k \approx \sqrt{N}$. With
64-bit hashes, expect collisions after ~$2^{32}$ = 4 billion documents — which is
a real consideration when deduplicating web-scale corpora, and the reason 128-bit
hashes are used there.

## Graph theory

A graph $G = (V, E)$ is a set of vertices and a set of edges. This is the single
most reusable structure in computer science, and machine learning is full of
graphs that people do not always name as such.

| Type | Definition | ML instance |
|---|---|---|
| Undirected | edges are unordered pairs | social network, molecule |
| Directed | edges are ordered | citation graph, causal DAG |
| Weighted | edges carry values | similarity graph, attention weights |
| Bipartite | two disjoint vertex sets | user–item recommendation |
| DAG | directed, no cycles | computation graph, Bayesian network |
| Tree | connected, acyclic, $|E| = |V| - 1$ | decision tree, parse tree, beam search tree |
| Complete | every pair connected | self-attention over a sequence |
| Hypergraph | edges join $>2$ vertices | group interactions |

### Representations, and their trade-offs

| Representation | Space | Edge query | Neighbour iteration | Used by |
|---|---|---|---|---|
| Adjacency matrix | $O(V^2)$ | $O(1)$ | $O(V)$ | dense graphs, attention masks |
| Adjacency list | $O(V+E)$ | $O(\deg)$ | $O(\deg)$ | sparse graphs, most GNN libraries |
| Edge list (COO) | $O(E)$ | $O(E)$ | $O(E)$ | PyTorch Geometric's `edge_index` |
| CSR/CSC | $O(V+E)$ | $O(\log \deg)$ | $O(\deg)$ | sparse matmul kernels, DGL |

Real graphs are sparse: a social network with $10^9$ users has average degree in
the hundreds, so $E \approx 10^{11}$ against $V^2 = 10^{18}$. Adjacency matrices
are not an option, and this is why GNN frameworks are built around
message-passing over edge lists rather than dense matrix multiplication.

### Traversal

```mermaid
flowchart TD
    Q["traversal problem"] --> A{"need shortest path<br/>in an unweighted graph?"}
    A -->|"yes"| BFS["BFS with a queue<br/>O of V plus E<br/>level by level"]
    A -->|"no"| B{"need topological order,<br/>cycle detection,<br/>or connected components?"}
    B -->|"yes"| DFS["DFS with a stack or recursion<br/>O of V plus E"]
    B -->|"no"| C{"weighted, non-negative?"}
    C -->|"yes"| DIJ["Dijkstra with a heap<br/>O of E log V"]
    C -->|"no"| BF["Bellman-Ford<br/>O of V times E<br/>handles negative weights"]
```

BFS explores by distance and therefore finds shortest paths in unweighted
graphs; DFS goes deep and is what you want for topological sort, cycle
detection, and strongly connected components (Tarjan/Kosaraju). Both are
$O(V+E)$.

### Graph algorithms with ML relevance

| Algorithm | Complexity | Why it appears in ML |
|---|---|---|
| BFS/DFS | $O(V+E)$ | $k$-hop neighbourhoods for GNN sampling |
| Dijkstra | $O(E\log V)$ | shortest-path features, routing |
| Union-find | $O(\alpha(n))$ near-constant | connected components, single-link clustering, dedup clusters |
| Kruskal / Prim MST | $O(E\log V)$ | single-linkage clustering is exactly MST cutting |
| PageRank | power iteration | node importance; the original was a Markov chain on a graph |
| Spectral clustering | eigendecomposition of the Laplacian | community detection, image segmentation |
| Max-flow / min-cut | $O(V^2E)$ or better | image segmentation (graph cuts), matching |
| Bipartite matching | Hungarian, $O(n^3)$ | DETR's set prediction loss, assignment problems |
| Viterbi | $O(TK^2)$ | best path in an HMM/CRF — dynamic programming on a trellis |

### The graph Laplacian

$$L = D - A, \qquad L_{\text{sym}} = I - D^{-1/2}AD^{-1/2}$$

where $D$ is the diagonal degree matrix. Key facts:

- $L$ is symmetric positive semi-definite.
- $\mathbf{x}^\top L \mathbf{x} = \sum_{(i,j)\in E} w_{ij}(x_i - x_j)^2$ — it
  measures how much a signal varies across edges. This is a **smoothness
  penalty**, which is why it appears in semi-supervised learning as a
  manifold-regularisation term.
- The multiplicity of eigenvalue 0 equals the number of connected components.
  The eigenvector for the second-smallest eigenvalue (the **Fiedler vector**)
  gives the best spectral bipartition.
- Graph convolutional networks are, in their original derivation, a first-order
  approximation to spectral filtering with $L_{\text{sym}}$. The famous GCN
  layer $H' = \sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}HW)$ comes directly
  from truncating a Chebyshev expansion of a spectral filter.

**Over-smoothing** is the graph-theoretic failure mode of deep GNNs: repeated
neighbourhood averaging is a random walk, and random walks converge to a
stationary distribution, so after enough layers every node representation
converges to the same vector. That is why most GNNs are 2–3 layers deep, and why
residual connections, jumping knowledge, and PairNorm exist.

## Recurrences and complexity

### Solving recurrences

Divide-and-conquer algorithms give recurrences of the form
$T(n) = aT(n/b) + f(n)$. The **master theorem** resolves them by comparing
$f(n)$ against $n^{\log_b a}$:

| Case | Condition | Result | Example |
|---|---|---|---|
| 1 | $f(n) = O(n^{\log_b a - \epsilon})$ | $T = \Theta(n^{\log_b a})$ | Karatsuba: $T=3T(n/2)+O(n) \Rightarrow n^{1.585}$ |
| 2 | $f(n) = \Theta(n^{\log_b a})$ | $T = \Theta(n^{\log_b a}\log n)$ | merge sort: $2T(n/2)+O(n)\Rightarrow n\log n$ |
| 3 | $f(n) = \Omega(n^{\log_b a+\epsilon})$, regularity | $T = \Theta(f(n))$ | work dominated by the combine step |

Strassen's matrix multiplication, $T(n) = 7T(n/2) + O(n^2)$, lands in case 1 and
gives $n^{\log_2 7} \approx n^{2.807}$ — asymptotically better than $n^3$, but
rarely used in ML because it is numerically less stable and the constant factors
lose to hardware-tuned $n^3$ kernels. A good reminder that asymptotics are not
the whole story.

### Complexity of things you actually run

| Operation | Complexity | Note |
|---|---|---|
| Dense matmul $(n\times m)(m\times p)$ | $O(nmp)$ | the cost model for all of deep learning |
| Self-attention, sequence $n$, dim $d$ | $O(n^2 d)$ time, $O(n^2)$ memory naively | FlashAttention keeps the time, drops memory to $O(n)$ |
| Feed-forward block, dim $d$, hidden $4d$ | $O(nd^2)$ | dominates attention until $n > d$ |
| Sorting $n$ items | $O(n\log n)$ | top-$k$ can be $O(n\log k)$ or $O(n)$ with quickselect |
| $k$-NN, brute force | $O(Nd)$ per query | ANN indexes (HNSW, IVF-PQ) trade recall for speed |
| $k$-means, one Lloyd iteration | $O(NKd)$ | why $K$ and $d$ both matter |
| Decision tree training | $O(Nd\log N)$ | sorting each feature at each node |
| Backprop through $L$ layers | same order as forward | reverse mode's key guarantee |
| Beam search, beam $B$, length $T$ | $O(BTV)$ scoring, $O(BT\log(BV))$ selection | linear in beam, not exponential |
| Exact Viterbi decoding | $O(TK^2)$ | vs $O(K^T)$ for brute force |

Notice the shape of the attention row. Attention is $O(n^2 d)$ and the FFN is
$O(nd^2)$; attention only dominates when $n > d$. For a model with $d = 4096$ and
$n = 512$, the FFN is the bottleneck, not attention — a fact that surprises
people who have absorbed "attention is quadratic" without the constant.

### P, NP, and why we approximate

- **P** — solvable in polynomial time.
- **NP** — solutions verifiable in polynomial time.
- **NP-complete** — the hardest problems in NP; a polynomial algorithm for one
  gives one for all.
- **NP-hard** — at least as hard as NP-complete, not necessarily in NP.

ML problems that are NP-hard, and what we do instead:

| Problem | Hardness | Practical approach |
|---|---|---|
| Optimal decision tree | NP-complete | greedy splitting (CART, ID3) |
| Exact $k$-means | NP-hard | Lloyd's algorithm with $k$-means++ init |
| Best feature subset | NP-hard | L1 regularisation, greedy forward/backward |
| Learning optimal Bayesian network structure | NP-hard | score-based greedy search, constraint-based |
| Exact MAP inference in a general graphical model | NP-hard | loopy BP, variational, sampling |
| Training a 3-node neural network to optimality | NP-hard | gradient descent, and hope |

That last one is worth sitting with: even a tiny network is NP-hard to train
optimally. Everything we do is a local method that works far better in practice
than the theory promises, and nobody fully understands why.

## Dynamic programming

DP applies when a problem has **optimal substructure** (the optimum is built
from optima of subproblems) and **overlapping subproblems** (the same
subproblems recur). It appears throughout NLP.

| DP algorithm | Recurrence idea | Where |
|---|---|---|
| Edit distance | insert/delete/substitute, take the min | spelling correction, WER, diff |
| Longest common subsequence | match or skip one side | ROUGE-L, diffing |
| Viterbi | best path to each state at each time | HMM POS tagging, CRF decoding |
| Forward–backward | sum over paths instead of max | HMM training, CTC loss |
| CTC | sum over all alignments of a label sequence | speech recognition without alignments |
| CKY parsing | best parse of each span | constituency parsing |
| Knapsack | include/exclude with a capacity | budgeted selection |

**Edit distance, worked.** With $D[i][j]$ the distance between the first $i$ and
first $j$ characters:

$$D[i][j] = \begin{cases} \max(i,j) & \text{if } \min(i,j)=0 \\ \min\bigl(D[i{-}1][j]+1,\; D[i][j{-}1]+1,\; D[i{-}1][j{-}1]+\mathbb{1}[a_i\ne b_j]\bigr) & \text{otherwise}\end{cases}$$

$O(mn)$ time, and $O(\min(m,n))$ space if you only need the number. Word error
rate in speech recognition is exactly this, computed over words rather than
characters.

**CTC** is the one to understand if you touch speech. It sums over *all*
alignments of a short label sequence to a long audio sequence using a
forward–backward DP, so you never need frame-level labels. The dynamic program
is what makes the loss differentiable and tractable.

## Logic and Boolean algebra

Propositional logic gives you $\land, \lor, \neg, \Rightarrow$, and the
equivalence $(p \Rightarrow q) \equiv (\neg p \lor q)$ that trips people up in
interviews. **De Morgan's laws** $\neg(p\land q) \equiv \neg p \lor \neg q$ are
worth reflexive fluency because they show up in query rewriting and in reasoning
about masks.

Where logic touches ML:

- **Attention masks** are Boolean matrices; causal masking is the predicate
  $j \le i$.
- **Neuro-symbolic systems** attach differentiable relaxations to logical
  operators (t-norms: $p \land q \approx pq$ or $\min(p,q)$).
- **SAT/SMT solvers** back constrained decoding and program synthesis.
- **Formal verification** of network properties (robustness certificates) is
  encoded as satisfiability over piecewise-linear constraints — which is exactly
  why ReLU networks are the ones we can verify.

## Number theory, the useful fragment

- **Modular arithmetic** underlies hashing (`h(x) mod m`), the hashing trick,
  and reproducible sharding of data across workers.
- **Primes** matter for hash table sizes — a prime modulus spreads clustered
  keys better than a power of two.
- **GCD/LCM** appear in scheduling and stride computations.
- **Universal and locality-sensitive hashing** are number-theoretic
  constructions; MinHash estimates Jaccard similarity and SimHash estimates
  cosine similarity, both by hashing rather than by comparing. Web-scale
  deduplication of pretraining data runs on exactly these.

## Discrete structures inside familiar models

| ML object | Discrete structure |
|---|---|
| Self-attention | complete weighted directed graph over tokens |
| Causal masking | a total order; the DAG of a chain |
| Byte-pair encoding | greedy merges over a frequency-ranked lattice |
| Beam search | breadth-limited tree search |
| MoE routing | bipartite assignment of tokens to experts, often solved with an auxiliary balanced-assignment objective |
| Decision tree | rooted tree with axis-aligned predicates |
| Random forest | forest of trees over bootstrapped multisets |
| Computation graph | DAG, evaluated in topological order |
| Tokenizer vocabulary | a trie |
| KV cache paging | a block table — an indirection layer, exactly like virtual memory |
| RadixAttention prefix cache | a radix tree over token prefixes |

The last two are worth noticing: modern inference servers borrow the operating
system's page table and the string algorithm's radix tree wholesale. Systems
work in ML is largely discrete algorithms applied to tensors.

## Self-check

1. A context window of 8k tokens grows to 32k. By what factor do attention FLOPs
   and attention memory grow, and does the FFN cost change?
2. Derive the 63.2% figure for bootstrap coverage.
3. Why do most GNNs stop at 2–3 layers? Answer in terms of random walks.
4. What is the complexity of Viterbi decoding, and what is it replacing?
5. You hash 5 billion documents with a 64-bit hash. Should you expect
   collisions? Show the birthday-bound reasoning.
6. Write the edit-distance recurrence and say what WER is in terms of it.
7. Explain what $\mathbf{x}^\top L\mathbf{x}$ measures and why it is used as a
   regulariser.

## Where to go next

- [Linear Algebra](./linear-algebra.md) — the Laplacian's spectrum, and matrices
  as graphs.
- [Probability](./probability.md) — random graphs, Markov chains, and the
  birthday bound.
- [Numerical Computing](./numerical-methods.md) — what the complexity table
  costs once floating point is involved.
