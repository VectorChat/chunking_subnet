# Ranking

The default validator uses a form of group tournament ranking to determine the ranking of miners. 

Specifically, validators maintain an internal ranking of all miners from which they decide whom to query and what weights to set. When selecting which miners to query, validators create groups of miners with adjacent ranks. They then select one of these groups at random and query all of the miners in that group. Once all of the miners' responses have been scored, the validator ranks them relative to one another and then adjusts these rankings to reflect their overall rankings. For organic queries, groups are created to include the miners specified by the user.

Here is an example of this system with 12 miners and a sample size of 4:

![ranking_visualization](../assets/ranking_visualization.png)

### Incentive Curve
When setting weights, the weight of the nth-best ranked miner will be twice that of the weight of the (n+1)th ranked miner, or (1/2)^n. This is so that improving a miner's rank by one spot will always result in more emissions than running more miners.

Here is an example of the incentive curve with 5 miners:

![incentive_curve](../assets/incentive_curve.png)