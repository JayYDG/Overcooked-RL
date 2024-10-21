# Decentralized Critic:
ppo_fcnn.py
runner_ppo_fcnn.py

sample command for running:
```
python runner_ppo_fcnn_v2.py --exp-name fcnn_v2_3_num_envs --layout-name counter_circuit_o_1order --num-envs 3 --num-minibatches 1 --epoch 1 --learning-rate-shared 5e-5 --learning-rate-actor 5e-5 --learning-rate-critic 5e-3 --num-updates 5000
```


# Centralized Critic:
ppo_fcnn_v2.py
runner_ppo_fcnn_v2.py

sample command for running:
```
python runner_ppo_fcnn_v2.py --exp-name fcnn_final_higher_entropy_default_l --layout-name counter_circuit_o_1order --num-envs 6 --num-minibatches 1 --epoch 1 --learning-rate-shared 5e-5 --learning-rate-actor 5e-5 --learning-rate-critic 5e-3 --num-updates 5000 --extra-reward True --time-punish 0
```

# LSTM Based:
ppo_lstm_v2.py
runner_lstm_v2.py

# Plot results:
make_plots.ipynb - make result plot

# Outcome
<img src="exp1_imgs.gif"/>
