Results with 30 epochs and default training parameters,
trained on GPU from Colab:

**Training only the FC layer on whole dataset**
CASE1:: test_accuracy: 66.05% ; test_XELoss: 1.0 -- time 00:05:36

**Training only the FC layer on 10% of dataset**
CASE1:: test_accuracy: 57.43% ; test_XELoss: 1.33 -- time 00:00:54

**Training only the CNN layers on whole rotation dataset**
ROTATION MODEL:: test_accuracy: 98.0% ; test_XELoss: 0.07 -- time 00:01:26

**Training FC only on 10% of dataset while using the CNN layers trained on rotation model**
CASE2:: test_accuracy: 73.08% ; test_XELoss: 0.77 -- total time 00:02:22

