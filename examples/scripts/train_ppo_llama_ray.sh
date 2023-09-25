set -x 
export PATH=$HOME/.local/bin/:$PATH

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/openllama2", "pip": "/openllama2/requirements.txt"}' \
    -- python3 examples/train_ppo_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --pretrain robertmyers/targon-7b \
    --critic_pretrain robertmyers/targon-7b \
    --model_revision v1.1.8 \
    --save_path /openllama2/ckpts \
    --micro_train_batch_size 1 \
    --train_batch_size 8 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --inference_tp_size 1 \
    --init_kl_coef 0.01 \
    --prompt_data robertmyers/gigatargon \
    --max_samples 80000 \
    --normalize_reward \
    --actor_init_on_gpu \
    --flash_attn \
    --gradient_checkpointing