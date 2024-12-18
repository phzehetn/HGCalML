Snowflake model to be used for thesis
This model should perform identical to previous models, however: 

* It shows reproducibility with a well documented training procedure
* It includes some additional information that can be used for performance plots


Training procedure: 

1. 5 epochs on smaller data with 200 pile-up only in 30 degree phi region
2. 23 epochs with decreasing learning rate and fixed batchnorm after 3 epochs
3. Exact repetition of second step. No significant performance difference observed
suggesting that around 20 epochs are enough 
