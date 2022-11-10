# save SAC agents at different timesteps into training
CUDA_VISIBLES=0 python train_sac.py --env quadruped_run --num-train-steps 2e6 --replay-buffer-capacity 2e6

# Use cached SAC agents to generate policy-dependent trajectories (100k for each actor)
CUDA_VISIBLES=0 python generate_trajectory_data.py --env quadruped_run --post-fix 100k

# Pretraining Policy Embedding GRU (PE_GRU) and Policy Embedding VAE (PE_VAE)
CUDA_VISIBLES=0 python pretraining_PE_GRU.py

# Train Ensemble Dynamics model with extended pretrained policy embedding
CUDA_VISIBLES=0 python train_GRU_PE_mbpo_model.py