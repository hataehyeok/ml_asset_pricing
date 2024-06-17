# Test

## Lists

- nn with nothing
- nn with batchnormalization
- nn with batchN, He init

## Results

Training with hidden_layers=[128, 64, 32], learning_rate=0.01, l1_lambda=0.001
R² score: -0.00019003558040320279
R² score: -0.00022022094712359497
R² score: -0.0026764402900874007

Training with hidden_layers=[32, 16, 8], learning_rate=0.01, l1_lambda=0.001
R² score: -0.003579853465275562
R² score: -8.638345741385223e-05
R² score: -0.006445952288908874

Training with hidden_layers=[128, 64], learning_rate=0.01, l1_lambda=0.001
R² score: -0.0003373899388645629
R² score: -0.004655685787650299
R² score: -0.0014945711091374214

Training with hidden_layers=[64, 32], learning_rate=0.01, l1_lambda=0.001
R² score: -0.00047730838332693537
R² score: -0.00604316555267026
R² score: -0.005249394461468437

Training with hidden_layers=[32, 16], learning_rate=0.01, l1_lambda=0.001
R² score: -0.002171391792431754
R² score: -0.0006860683035907478

Training with hidden_layers=[16, 8], learning_rate=0.01, l1_lambda=0.001
R² score: -0.0011282984714329203
R² score: -7.163918541341907e-05
