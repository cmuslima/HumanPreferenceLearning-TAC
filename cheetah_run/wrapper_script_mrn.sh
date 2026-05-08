#!/bin/bash
# submit_alpha_sweep.sh

# --- MRN TAC sweep ---
ALPHAS=(0.1)
for alpha in "${ALPHAS[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000M
#SBATCH --time=3:00:00
#SBATCH --array=0
#SBATCH --job-name=MRN_tac_alpha_${alpha}

module load python/3.10.13
cd /home/cmuslima/projects/aip-mtaylor3/cmuslima
source ENV/bin/activate
cd /home/cmuslima/scratch/HumanPreferenceLearning-TAC/MRN

start_time=\`date +%s\`
echo "Starting training... alpha=${alpha}, seed=\$SLURM_ARRAY_TASK_ID"

python train_MRN_TAC.py env=cheetah_run alpha=${alpha} experiment=MRN_tac_alpha${alpha} seed=\$SLURM_ARRAY_TASK_ID agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=50 feed_type=1 num_meta_steps=5000 device=cuda

end_time=\`date +%s\`
echo "Run time: \$((end_time-start_time))s"
EOF
    echo "Submitted MRN TAC job for alpha=$alpha"
done

# --- MRN sweep ---
ALPHAS=(1)


for alpha in "${ALPHAS[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000M
#SBATCH --time=3:00:00
#SBATCH --array=0
#SBATCH --job-name=MRN_alpha_${alpha}

module load python/3.10.13
cd /home/cmuslima/projects/aip-mtaylor3/cmuslima
source ENV/bin/activate
cd /home/cmuslima/scratch/HumanPreferenceLearning-TAC/MRN

start_time=\`date +%s\`
echo "Starting training... alpha=${alpha}, seed=\$SLURM_ARRAY_TASK_ID"

python train_MRN.py env=cheetah_run alpha=${alpha} experiment=MRN_alpha${alpha} seed=\$SLURM_ARRAY_TASK_ID agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=50 feed_type=1 num_meta_steps=5000 device=cuda

end_time=\`date +%s\`
echo "Run time: \$((end_time-start_time))s"
EOF
    echo "Submitted MRN job for alpha=$alpha"
done