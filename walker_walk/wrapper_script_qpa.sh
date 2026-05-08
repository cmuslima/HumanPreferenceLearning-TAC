#!/bin/bash
# submit_alpha_sweep.sh

# --- QPA TAC sweep ---
ALPHAS=(0.7)
for alpha in "${ALPHAS[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000M
#SBATCH --time=3:00:00
#SBATCH --array=5-9
#SBATCH --job-name=qpa_tac_alpha_${alpha}

module load python/3.10.13
cd /home/cmuslima/projects/aip-mtaylor3/cmuslima
source ENV/bin/activate
cd /home/cmuslima/scratch/HumanPreferenceLearning-TAC/QPA

start_time=\`date +%s\`
echo "Starting training... alpha=${alpha}, seed=\$SLURM_ARRAY_TASK_ID"

python train_QPA_TAC.py env=walker_walk alpha=${alpha} experiment=qpa_tac_alpha${alpha} seed=\$SLURM_ARRAY_TASK_ID agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=500000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=2000 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=10 data_aug_ratio=20 ensemble_size=1 explore=false her_ratio=0.5 wandb=true device='cuda'

end_time=\`date +%s\`
echo "Run time: \$((end_time-start_time))s"
EOF
    echo "Submitted QPA TAC job for alpha=$alpha"
done

# --- QPA sweep ---
ALPHAS=(1)


for alpha in "${ALPHAS[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000M
#SBATCH --time=3:00:00
#SBATCH --array=5-9
#SBATCH --job-name=qpa_alpha_${alpha}

module load python/3.10.13
cd /home/cmuslima/projects/aip-mtaylor3/cmuslima
source ENV/bin/activate
cd /home/cmuslima/scratch/HumanPreferenceLearning-TAC/QPA

start_time=\`date +%s\`
echo "Starting training... alpha=${alpha}, seed=\$SLURM_ARRAY_TASK_ID"

python train_QPA.py env=walker_walk alpha=${alpha} experiment=qpa_alpha${alpha} seed=\$SLURM_ARRAY_TASK_ID agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=500000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=2000 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=10 data_aug_ratio=20 ensemble_size=1 explore=false her_ratio=0.5 wandb=true device='cuda'

end_time=\`date +%s\`
echo "Run time: \$((end_time-start_time))s"
EOF
    echo "Submitted QPA job for alpha=$alpha"
done