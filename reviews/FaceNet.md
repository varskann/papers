## FaceNet: A Unified Embedding for Face Recognition and Clustering
Florian Schroff, Dmitry Kalenichenko, James Philbin, CVPR 2015

### Summary
Learn Face Embeddings such that the squared L2 distances(Euclidean Space) in the embedding space directly correspond to face similarity: faces of the same person have small distances and faces of distinct people have large distances.

Embeddings can be used for Face Recognition, Clustering and Face Verification


__*Method*__:
1. Use a Deep Convolutional Natwork (Inception Net / Zeiler&Fregus Style Net), and set up an end-to-end learning setup with _Triplet Loss_
2. _**Triplet Loss**_: we want to ensure that an image x<sup>a</sup><sub>i</sub>(anchor) of a specific person is closer to all other images x<sup>p</sup><sub>i</sub>(positive) of the same person than it is to any image x<sup>n</sup><sub>i</sub>(negative) of any other person. A margin α is enforced between positive and negative pairs
3. _**Triplet Selection**_:  To ensure fast convergence it is crucial to select triplets that violate the triplet constraint, i.e., Select Hard and Semi-Hard triplets for training. Use Online Triplet Generation to select hard positive/negative exemplars from within a mini-batch

_**Evaluation**_

P<sub>same</sub> = All face pairs with same identity
P<sub>diff</sub> = All pairs of different identities

For a given face distance _d_:

- True Accepts : TA(d) = {(i, j) ∈ P<sub>same</sub>, with D(x<sub>i</sub>, x<sub>j</sub> ) ≤ d}
- False Accepts: FA(d) = {(i, j) ∈ P<sub>diff</sub>, with D(x<sub>i</sub>, x<sub>j</sub> ) ≤ d}

- Validation Rate: VAL(d) = |TA(d)| / |P<sub>same</sub>| 
- False Accept Rate: FAR(d) = |FA(d)| / |P<sub>diff</sub>| 


_**Experiments**_:
- Accuracy is directly correlated to the computation required(FLOPS)
- CNN Architecture: Deeper the Model, Lesser the  FAR
- Roubst to Image Quality of 20
- Large Embeddings perfom better, but require more training 
- More Training Data improves the VAL but not worth the effort

