torchrun -m --nproc_per_node 8 training.train_univit \
  --list_batch_sizes 64 \
  --output ./output/baseline