# 系统架构

VectorViz 使用 `src` 布局，并通过稳定的小接口连接物理、积分、服务与前端。依赖方向只能从外层指向内层：核心模型不得导入 Web 框架或绘图库。当前核心保持为几个小模块；当网格场、导入器和相对论动力学落地后，再按领域拆成子包。

## 目录结构

```text
Vector_field_streamline/
├── pyproject.toml
├── mkdocs.yml
├── docs/
├── notebooks/                  # Jupyter 教程与可复现实验
├── src/
│   └── vectorviz/
│       ├── __init__.py          # 稳定公共导出
│       ├── core.py              # VectorField、Domain、排除几何
│       ├── fields.py            # 解析场与组合场
│       ├── tracing.py           # 自适应场线积分与结果类型
│       └── web/
│           ├── app.py           # HTTP API 与静态资源入口
│           ├── schemas.py       # 请求/响应契约
│           └── static/          # 无构建浏览器前端
└── tests/                       # 单元、验证、Web 契约测试
```

模块可以在保持下述依赖边界的前提下继续拆分。

## 分层与职责

| 层 | 负责 | 不负责 |
|---|---|---|
| `core.py` | 接口、区域、排除几何和共享数组约定 | 具体物理模型、HTTP、绘制 |
| `fields.py` | 批量求值解析场与组合场 | 播种、积分、颜色映射 |
| `tracing.py` | 方向归一化、双向积分、事件、终止信息 | 物理源公式、浏览器状态 |
| 验证代码 | 解析解、不变量、残差与收敛诊断 | 修改模型参数以“修好”结果 |
| `web/` | 输入校验、场景编排、静态资源、JSON 序列化 | 复制 NumPy/SciPy 算法到前端 |
| 前端 | 参数交互、图层和提示信息 | 物理公式与积分算法 |

## 核心数据流

```mermaid
sequenceDiagram
    participant UI as 浏览器
    participant API as Web API
    participant Field as VectorField
    participant Tracer as FieldLineTracer

    UI->>API: POST /api/scene
    API->>API: 校验预设、密度与分辨率
    API->>Field: 批量 evaluate(grid_points)
    Field-->>API: vectors (..., D)
    loop 每个种子
        API->>Tracer: trace(seed, BOTH)
        Tracer->>Field: evaluate(points)
        Field-->>Tracer: tangent vectors
        Tracer-->>API: TraceResult
    end
    API-->>UI: scalar + lines + sources + metadata
    UI->>UI: 颜色映射、箭头、场线和源图层
```

API 返回数值与科学元数据，而不返回预先栅格化的截图。这样前端可在不重复求解的情况下切换图层、调整颜色或查看探针。

## 核心抽象

### `VectorField`

所有场实现都遵守一个批量接口：

```python
vectors = field.evaluate(points)
```

- `points` 的形状是 `(..., dimension)`；
- 返回值形状与 `points` 相同；
- 计算使用浮点数组，不能静默把结果截断为整数；
- 无效点应通过明确异常、mask 或非有限值政策处理；
- 实现不得根据当前色图或相机改变结果。

解析场、数值积分场、规则网格插值场和组合场都通过这一接口进入积分器。

### `Domain`

`Domain` 描述积分允许进入的空间范围，并为越界事件提供单一事实来源。几何源的排除区域可以是独立 mask；不要把“离开计算域”和“命中奇点”混成同一个终止原因。

### `TraceResult`

轨迹结果至少保存：

- 按顺序排列的坐标点；
- 正向、反向或双向信息；
- 每个分支的终止原因；
- 弧长或积分参数；
- 可选的场强、误差估计和诊断元数据。

渲染器消费 `TraceResult`，不调用积分器内部方法。

## Web 适配层

Web 层把用户友好的预设名称转换成核心对象：

```text
electric_dipole -> PointChargeField(批量异号电荷)
magnetic_dipole -> MagneticDipoleField(...) 的 z=0 不变平面适配器
uniform         -> UniformField(...)
```

`resolution` 控制标量背景采样；`density` 控制播种数量或间距。这两个参数不能互相代替。完整 JSON 契约见 [HTTP API](api.md#http-api)。

## 缓存边界 { #cache-boundaries }

推荐使用三个彼此独立的缓存键：

1. **场缓存**：模型、物理参数、区域、采样网格和求解器版本；
2. **曲线缓存**：场缓存键、种子、积分选项和终止政策；
3. **显示缓存**：视口、色图、线宽、图层可见性和相机。

改变显示状态不使前两层失效；改变播种只使曲线缓存失效；改变源强或源位置才使场缓存失效。

## 扩展点

### 数值网格场

规则网格可由 `RegularGridField` 封装；非结构网格应保留单元拓扑，并在单元内插值。导入器将 FEMM、VTK 或测量数据转换成统一场接口，而不是让积分器依赖某种文件格式。

### 粒子与相对论轨迹

普通场线状态只有位置；带电粒子状态至少包括位置与速度；Kerr 光线状态包括四维位置与四动量。未来应抽象为通用 `Dynamics`：

```python
derivative = dynamics.rhs(parameter, state)
position = dynamics.position(state)
```

它们可以共享事件、曲线数据和渲染器，但不能都实现成 `VectorField`。原因和方程见[引力场与黑洞光线](tutorial/05-gravity-and-rays.md)。

## 架构约束的测试

集成测试至少应证明：

- 同一场对象既能批量采样，也能进入追踪器；
- Web 端点只通过公共 API 构建场，不访问私有积分细节；
- JSON 中的点、标量尺寸与 `domain` 一致；
- 改变前端显示选项不会改变核心轨迹坐标；
- Python 包可在仓库根目录之外导入，防止假安装掩盖 `src` 布局问题。

参见[开发指南](development.md)了解测试组织和提交检查。
