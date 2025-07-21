echo "Training Autoencoder, this might take a long time"
# Add memory optimization environment variable
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python VAE/VAE_train.py --data_dir 'data/tabula_muris/all.h5ad' --num_genes 18996 --save_dir 'output/checkpoint/AE/my_VAE' --max_steps 3000 --state_dict 'data/annotation_model_v1' --max_minutes 180
echo "Training Autoencoder done"

cd ..
echo "Training diffusion backbone"
# Add memory optimization parameters for diffusion model trainin
export PYTORCH_ENABLE_MPS_FALLBACK=1 && python cell_train.py --data_dir 'data/tabula_muris/all.h5ad' --vae_path 'output/checkpoint/AE/my_VAE/model_seed=0_step=2999.pt' --model_name 'my_diffusion' --lr_anneal_steps 100000 --save_dir 'output/checkpoint/backbone' --microbatch 16
echo "Training diffusion backbone done"

echo "Training classifier"
python classifier_train.py --data_dir 'data/tabula_muris/all.h5ad' --model_path "output/checkpoint/classifier/open_problem_classifier" \
    --iterations 400000 --vae_path 'output/checkpoint/AE/my_VAE/model_seed=0_step=2999.pt' --num_class 22
echo "Training classifier, done"

# Generate samples
python cell_sample.py --model_path 'output/checkpoint/backbone/my_diffusion/model100000.pt' --sample_dir 'output/simulated_samples/muris' --num_samples 3000 --batch_size 1000

