# Test

## Lists

- nn with nothing
- nn with batchnormalization
- nn with batchN, He init
- nn with batchN, He init, b4000
- nn with batchN, He init, b4000, dropout
- nn using trainset only

## Results

Training with hidden_layers=[128, 64, 32], learning_rate=0.01, l1_lambda=0.001

- R² score: -0.00019003558040320279
- R² score: -0.00022022094712359497
- R² score: -0.0026764402900874007
- R² score: -0.012853528406173487
- R² score: -0.001679358859759894
- R² score: -9.637879330393062e-06


Training with hidden_layers=[32, 16, 8], learning_rate=0.01, l1_lambda=0.001

- R² score: -0.003579853465275562
- R² score: -8.638345741385223e-05
- R² score: -0.006445952288908874
- R² score: -0.2275020717321221
- R² score: -0.252127583983607
- R² score: -5.9826907707449806e-05

Training with hidden_layers=[128, 64], learning_rate=0.01, l1_lambda=0.001

- R² score: -0.0003373899388645629
- R² score: -0.004655685787650299
- R² score: -0.0014945711091374214
- R² score: -0.018598837507907362
- R² score: -0.012685931852879584
- R² score: -0.0008535529816999787

Training with hidden_layers=[64, 32], learning_rate=0.01, l1_lambda=0.001

- R² score: -0.00047730838332693537
- R² score: -0.00604316555267026
- R² score: -0.005249394461468437
- R² score: -0.010855673824291756
- R² score: -0.05376580058817915
- R² score: -0.0004544163666126977

Training with hidden_layers=[32, 16], learning_rate=0.01, l1_lambda=0.001

- R² score: -0.002171391792431754
- R² score: -0.0006860683035907478
- not
- R² score: -5.169250457929561
- R² score: -0.02132134968502175
- R² score: -0.004577154318959664


Training with hidden_layers=[16, 8], learning_rate=0.01, l1_lambda=0.001

- R² score: -0.0011282984714329203
- R² score: -7.163918541341907e-05
- not
- R² score: -0.01846136925646169
- R² score: -0.012713475562173837
- R² score: -2.5241833423006454e-05
