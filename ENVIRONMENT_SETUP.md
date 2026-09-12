# 虚拟环境依赖模块

在项目虚拟环境中需要安装以下 Python 包：

```powershell
python -m pip install numpy scipy pandas matplotlib networkx onnx
python -m pip install packaging appdirs pyyaml ninja tqdm graphviz
python -m pip install pytest==8.1.1 pytest-order pytest-mock pylint
```

`auto_LiRPA` 相关代码还需要 PyTorch：

```powershell
python -m pip install "torch>=2.0.0,<2.9.0" "torchvision>=0.12.0,<0.24.0"
```

Marabou 验证流程还需要 `maraboupy`。：

```powershell
pip install maraboupy
```

如果希望把项目自带的 `auto_LiRPA` 安装到虚拟环境中，可选执行：

```powershell
python -m pip install -e .\parnv-MC\auto_LiRPA-master
```
# 验证神经网络命令行：
验证ACASXu神经网络：
```powershell
cd Prune/parnv-acasxu
python batch_verify_acasxu_inputs.py --quiet-runs --mode par
```

验证MNIST或CIFAR-10神经网络：
```powershell
cd Prune/parnv-MC
python batch_verify_inputs.py \                           
  --dataset mnist \
  --mode p \
  --networks-dir /home/gpu/yyc_projects/Prune/data/models/mnist \
  --dataset-root "/home/gpu/yyc_project /data" \
  --dataset-split train \
  --start-index 0 \
  --end-index 50 \
  --verification-epsilon 0.06 \
  --marabou-timeout-seconds 3600 \
  --property-timeout-seconds 3600

```