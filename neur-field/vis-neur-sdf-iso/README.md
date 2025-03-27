# 神经符号距离场的可视化

### 简介

这是一个可视化神经符号距离场的一些代码，目前我只加入了可视化等值面切片功能。代码部分来自于文章[*NeurCADRecon: Neural Representation for Reconstructing CAD Surfaces by Enforcing Zero Gaussian Curvature*]，预训练的 pytorch 模型来自于文章[*1‐Lipschitz Neural Distance Fields*]。

---

### 使用

- 输入：一个 pytorch 模型，是一个神经表示的符号距离场，满足输入 $n\times 3$的张量（点坐标），输出$n\times 1$的张量（符号距离）；
- 输出：一个 html 文件，可以用浏览器打开。

  
```bash
python vis_neural_sdf_iso.py -i /path/to/input.pt -o /path/to/output.html
```

---

### 可能的一些问题

Q1：如何获得 pytorch 模型？

A1：查看你的代码，以 *NeurCADRecon* 为例，在调用模型之前加入：
```python
# 保存模型
example = torch.rand(100,3).to(device)
traced_script_module = torch.jit.trace(net.decoder, example)
traced_script_module.save(output_dir)

# 调用模型
mesh = utils.implicit2mesh(net.decoder, None,
                          args.grid_res,
                          translate=-cp,
                          scale=1 / scale,
                          get_mesh=True, device=device, bbox=bbox)
```
或者以 *1‐Lipschitz Neural Distance Fields* 为例，在保存模型参数之前加入：
```python
# 保存模型
example = torch.rand(100,3).to(device)
traced_script_module = torch.jit.trace(net.decoder, example)
traced_script_module.save(output_dir)

# 保存模型参数
path = os.path.join(config.output_folder, "model_final.pt")
save_model(model, path)

def save_model(model, path):
    data = { "id": model.id, "meta" : model.meta, "state_dict" : model.state_dict()}
    torch.save(data, path)
```
当然也可以在读入模型之后加入：
```python
# 读入模型
model = load_model("model_final.pt", device)

# 保存模型
example = torch.rand(100,3).to(device)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save(output_dir)

def load_model(path, device:str):
    data = torch.load(path, map_location=device)
    model_type = data.get("id","Spectral")
    if model_type == "Spectral":
        model = DenseLipNetwork(*data["meta"])
    elif model_type == "SDP":
        model = DenseSDP(*data["meta"])
    elif model_type == "MLP":
        model = MultiLayerPerceptron(*data["meta"])
    elif model_type == "SIREN":
        model = SirenNet(*data["meta"])
    elif model_type == "MLPS":
        model = MultiLayerPerceptronSkips(*data["meta"])
    elif model_type == "PHASE":
        model = PhaseNet(*data["meta"])
    else:
        raise Exception(f"Model type {model_type} not recognized")
    model.load_state_dict(data["state_dict"])
    return model.to(device)
```
总之 `torch.jit.trace` 的是一个可以直接调用的模型。

Q2：可以使用保存的模型参数作为输入吗？

A2：不可以。保存的模型参数需要结合网络结构才可以调用。

---

### 参考文献
- Guillaume Coiffier, Louis Béthune. *1‐Lipschitz Neural Distance Fields*.
  - Computer Graphics Forum 2024
  - [[Paper](https://arxiv.org/abs/2407.09505)][[Project Page](https://gcoiffier.github.io/publications/onelipsdf/)][[Code](https://github.com/GCoiffier/1-Lipschitz-Neural-Distance-Fields)]
- Qiujie Dong, Rui Xu, Pengfei Wang, Shuangmin Chen, Shiqing Xin, Xiaohong Jia, Wenping Wang, Changhe Tu. *NeurCADRecon: Neural Representation for Reconstructing CAD Surfaces by Enforcing Zero Gaussian Curvature*.
  - SIGGRAPH 2024
  - [[Paper](https://dl.acm.org/doi/10.1145/3658171)][[Project Page](https://qiujiedong.github.io/publications/NeurCADRecon/)][[Code](https://github.com/QiujieDong/NeurCADRecon)]

[*NeurCADRecon: Neural Representation for Reconstructing CAD Surfaces by Enforcing Zero Gaussian Curvature*]: https://dl.acm.org/doi/10.1145/3658171

[*1‐Lipschitz Neural Distance Fields*]: https://arxiv.org/abs/2407.09505
