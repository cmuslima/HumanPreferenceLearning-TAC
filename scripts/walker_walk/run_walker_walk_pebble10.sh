#!/bin/bash
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000M
#SBATCH --time=3:00:00
#SBATCH --array=0-4



module load python/3.10.13
cd /home/cmuslima/projects/aip-mtaylor3/cmuslima
source ENV/bin/activate
cd /home/cmuslima/scratch/HumanPreferenceLearning-TAC/PEBBLE

start_time=`date +%s`
echo "starting training..."
echo "Starting task $SLURM_ARRAY_TASK_ID"

python train_PEBBLE.py env=walker_walk alpha=10 experiment=pebble10 seed=$SLURM_ARRAY_TASK_ID agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=50 feed_type=1 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 device=cuda
end_time=`date +%s`
runtime=$((end_time-start_time))

echo "run time"
echo $runtime
