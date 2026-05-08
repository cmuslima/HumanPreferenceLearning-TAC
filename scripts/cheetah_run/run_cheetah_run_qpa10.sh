#!/bin/bash
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000M
#SBATCH --time=3:00:00
#SBATCH --array=0-4


module load python/3.10.13
cd /home/cmuslima/projects/aip-mtaylor3/cmuslima
source ENV/bin/activate
cd /home/cmuslima/scratch/HumanPreferenceLearning-TAC/QPA

start_time=`date +%s`
echo "starting training..."
echo "Starting task $SLURM_ARRAY_TASK_ID"

python train_QPA.py alpha=10 env=cheetah_run experiment=qpa10 seed=$SLURM_ARRAY_TASK_ID agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=2000 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=10 data_aug_ratio=20 ensemble_size=1 explore=false her_ratio=0.5 wandb=true device='cuda'
end_time=`date +%s`
runtime=$((end_time-start_time))

echo "run time"
echo $runtime
