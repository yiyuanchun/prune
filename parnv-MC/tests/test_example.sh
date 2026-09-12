python3 ../parnv.py \
  --mnist-classifier /path/to/mnist_model.nnet \
  --verification-epsilon 0.02 \
  --verification-sample-index 0 \
  --dataset-split test \
  --dataset-root /path/to/mnist_data \
  -m marabou_with_ar \
  -a global \
  -r global
