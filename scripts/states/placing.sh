time python enjoy.py \
    --env-name InmoovHandPlaceBulletEnv-v9 \
    --load-dir trained_models_0313_2_placeco_0316_3/ppo/ \
    --non-det 0 \
    --seed=1898 \
    --random_top_shape 1 \
    --renders 0 \
    --exclude_hard 0 \
    --obs_noise 1 \
    --n_best_cand 2 \
    --cotrain_stack_place 1 \
    --save_states 1 \
    --n_trials 20
    