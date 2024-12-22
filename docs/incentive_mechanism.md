# Incentive Mechanism

This subnet incentivizes responses that maximize intrachunk similarity and interchunk dissimilarity, without exceeding variable constraints, such as time or maximum chunk length.

The possible range of scores is heavily dependent on the given document. For example, a document consisting only of the letter 'a' repeating thousands of times would trivialize all intrachunk similarity to '1' and all interchunk dissimilarity to '0.' Conversely, if a document consisted of the random string, it would be impossibly hard to find any similarity.

While those are extreme examples, all documents vary in degree of similarity and dissimilarity. Therefore, in order to control for the document, responses of miners are compared relatively—between all miners who were asked the same question. As a result, any given query can only change the ranks of miners within the queried group.

## Group Tournament Ranking

This subnet uses a form of Group Tournament Ranking to control for the confounding effects of the input document. All validators maintain their own internal ranking of miners, from which they create groups and set weights.

### 1. Forming Groups

The initial group size currently starts at 2, then doubles until all (256) possible UIDs are included.

Example:

```txt
0 1 <-- Group 0
  1 2 3 4 <-- Group 1
      3 4 5 6 7 8 <-- Group 2
```

The numbers represent the global rank of the miner at the position (assuming stable ranks at a fixed interval of time). The indentation represents how the groups overlap.

### 2. Evaluating

The document is then sent to all miners in the selected group, alongside the constraints of maximum `chunk_size`, `time_soft_max` and `chunk_qty`. The responses are then scored as described in [Evaluation](./evaluation.md).

### 3. Ranking

Miners are ranked within their group based on the reward values they received for their responses. Then scores and global rankings for the validator's tournament are updated based on these local group rankings. More details can be found [in the ranking doc](./ranking.md).

### Example

Here is an example of this system with 12 miners and a sample size of 4:

![ranking_visualization](../assets/ranking_visualization.png)

> [!WARNING]
> The above example is for illustrative purposes only. The actual implementation has since changed as group sizes are now variable and group "rank values" are floats rather than integers. More details can be found [in the ranking doc](./ranking.md).

## Incentive Curve

When setting weights, the weight of the _nth_-best ranked miner will be twice that of the weight of the _(n+1)_-th ranked miner, or \((1/2)^n\), up to the top _k_ miners. Therefore, the weight function for the top _k_ miners is given by:

$$
w_n = \left(\frac{1}{2}\right)^{n-1} \quad \text{for} \quad n \leq k
$$

Right now, _k_ is set to 7.

Then, the incentive is distributed linearly to the rest of the active miners.

https://github.com/VectorChat/chunking_subnet/blob/425f46c33542e1d17e3fd6baf10b9266e46215bc/chunking/base/validator.py#L479-L543

Here is an example of the incentive curve with 5 miners:

![incentive_curve](../assets/incentive_curve.png)
