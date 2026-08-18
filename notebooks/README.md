# Notebooks

这里放 Jupyter 教程、推导和数值实验。可复用的物理公式、积分器和数据结构仍放在
`src/vectorviz/`；笔记本只调用公共 API。这样同一个计算既能被测试，也能被网页前端使用。

当前的 `vector_field.ipynb` 是项目早期留下的草稿，已从仓库根目录移到这里。后续笔记本按
`01_主题.ipynb`、`02_主题.ipynb` 的格式编号。

项目继续使用 uv，不需要 Conda。临时启动 JupyterLab：

```powershell
uv run --with jupyterlab jupyter lab notebooks
```

新笔记本遵守三条规则：

1. 第一段写清物理量、单位、坐标系和假设。
2. 图旁边给出数值检查，不能只写“看起来正确”。
3. 已经进入 `src/vectorviz/` 的功能不在笔记本里复制一份。
