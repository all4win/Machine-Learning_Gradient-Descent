# Machine-Learning_Gradient-Descent
An implementation of Gradient Descent by **Tiancheng Gong**

***
## Data
Original Data: Download raw data from [UCI Airfoil Self-Noise](http://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise/) as *dataset.csv*\
Processed Data: Adjust the value of attributes to the same scale [1, 100] and save as *dataset1.csv*\
| No. of Instances | No. of Attributes | Attribute Characteristics | Associated Tasks |
|:----------------:|:-----------------:|:-------------------------:|:----------------:|
| 1503             | 6                 | Real                      | Regression       |
***
## Approach
I used [*Stochastic Gradient Descent*](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) instead of [*Batch Gradient Descent*](https://en.wikipedia.org/wiki/Gradient_descent) because the former is greedy and thus usually converges within fewer iterations and requires less computation resource.
***
## Variables
| Scale (training size/total) | Learning Rate (lr) | No. of Iterations | Starting Point |
|:---------------------------:|:------------------:|:-----------------:|:--------------:|
| 0.8                         | 0.0001             | 1500              | zero vector    |
***
## Result
After running grad_descent, the accuracy of preiction is 3-4%. Check *result.txt*.
