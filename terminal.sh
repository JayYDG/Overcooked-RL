# python runner_lstm_v2.py --learning-rate 5e-3
# python runner_lstm_v2.py --num-envs 30 --num-minibatches 5 --epoch 2 --learning-rate 2.5e-3
# python runner_lstm_v2.py --num-envs 30 --num-minibatches 5 --epoch 2 --learning-rate 5e-3
# python runner_lstm_v2.py --num-envs 5 --num-minibatches 1 --epoch 2 --learning-rate 1e-3
# python runner_lstm_v2.py --num-envs 10 --num-minibatches 2 --epoch 2 --learning-rate 1e-
# python runner_lstm_v2.py --num-envs 5 --num-minibatches 1 --epoch 4 --learning-rate 5e-4
# python runner_lstm_v2.py --num-envs 20 --num-minibatches 2 --epoch 4 --learning-rate-shared 5e-4 --learning-rate-actor 5e-4 --learning-rate-critic 5e-5
# python runner_lstm_v2.py --num-envs 10 --num-minibatches 2 --epoch 2 --learning-rate-shared 5e-4 --learning-rate-actor 5e-4 --learning-rate-critic 5e-5
# python runner_lstm_v2.py --num-envs 20 --num-minibatches 2 --epoch 4 --learning-rate-shared 5e-4 --learning-rate-actor 5e-4 --learning-rate-critic 5e-4
# python runner_lstm_v2.py --num-envs 20 --num-minibatches 2 --epoch 4 --learning-rate-shared 5e-3 --learning-rate-actor 5e-3 --learning-rate-critic 5e-5
# python runner_lstm_v2.py --num-envs 20 --num-minibatches 2 --epoch 4 --learning-rate-shared 5e-3 --learning-rate-actor 5e-5 --learning-rate-critic 5e-5
# python runner_lstm_v2.py --num-envs 2 --num-minibatches 1 --epoch 1 --exp-name no_extra_reward --learning-rate-shared 5e-5 --learning-rate-actor 5e-5 --learning-rate-critic 5e-5
# python runner_lstm_v2.py --num-envs 4 --num-minibatches 2 --epoch 2 --exp-name with_extra_reward_4_envs_2_minibathes_2_epoch-continue-2 --learning-rate-shared 5e-4 --learning-rate-actor 5e-4 --learning-rate-critic 5e-4
# python runner_ppo_cnn.py --num-envs 20 --num-minibatches 1 --epoch 1 --exp-name with_extra_reward_20_envs_1_minibathes_1_epoch_continue-2 --learning-rate-shared 5e-5 --learning-rate-actor 5e-5 --learning-rate-critic 5e-3
# python runner_ppo_fcnn_v2.py --exp-name fcnn_v2_3_num_envs --num-envs 3 --num-minibatches 1 --epoch 1 --learning-rate-shared 5e-5 --learning-rate-actor 5e-5 --learning-rate-critic 5e-3 --num-updates 5000
python runner_ppo_fcnn_v2.py --exp-name fcnn_v2_3_num_envs_fast --num-envs 3 --num-minibatches 1 --epoch 1 --learning-rate-shared 5e-4 --learning-rate-actor 5e-4 --learning-rate-critic 5e-3 --num-updates 5000
python runner_ppo_fcnn_v2.py --exp-name fcnn_v2_3_num_envs_all_fast --num-envs 3 --num-minibatches 1 --epoch 1 --learning-rate-shared 5e-4 --learning-rate-actor 5e-4 --learning-rate-critic 5e-2 --num-updates 5000
python runner_ppo_fcnn_v2.py --exp-name fcnn_v2_3_num_envs_6_num_envs --num-envs 6 --num-minibatches 1 --epoch 1 --learning-rate-shared 5e-5 --learning-rate-actor 5e-5 --learning-rate-critic 5e-3 --num-updates 3000
python runner_ppo_fcnn_v2.py --exp-name fcnn_v2_3_num_envs_1_num_envs --num-envs 1 --num-minibatches 1 --epoch 1 --learning-rate-shared 5e-5 --learning-rate-actor 5e-5 --learning-rate-critic 5e-3 --num-updates 3000