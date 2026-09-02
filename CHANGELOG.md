# Changelog

本文件记录 VectorViz 面向使用者可见的功能与契约变化。项目在 v1.0 前仍允许修正公共契约；每次修正都会同步更新 API 文档与相关教程。

## [Unreleased]

## [0.3.0] - 2026-09-02

### Added

- 场景元数据以 `coverage`、`equal_flux`、`feature` 枚举和独立的 `seed_description` 公开播种策略，并分别报告末端与起点端终止统计。
- 电荷场从全部正负源按绝对强度分配总种子预算；负源反向轨迹只在已有同源对的正向可渲染代表时抑制，并显式返回 `suppressed_count` 与 `rendered_line_count`。
- 双向场线响应可用 `start_termination` 描述点序起点端的终止原因，点序始终沿正场方向。

### Changed

- 默认 Halbach 阵列改为在 $y=\pm0.45$ m 的两条完整平行轨道上等距覆盖播种并双向追踪；默认预设的密度预算不再受八个显示源数量约束。
- 单磁偶极预设改为沿随偶极几何旋转的赤道线按距离覆盖播种并双向追踪；自定义多偶极仍使用逐源外向半球覆盖。
- 圆环电流预设使用公开磁通函数反解等 $\psi$ 轮廓；偶数预算成对镜像，奇数预算另保留一条轴线特征线。

## [0.2.2] - 2026-09-02

### Added

- `/api/presets` 为可编辑点源场景公开 `source_separation` capability；active 排除区域的源中心距离必须严格大于 0.322 m，422 会点名冲突源的原请求下标。
- 浏览器科学详情显示场模型、自由文本播种说明与终止统计；已知终止原因本地化，未知原因保留原始键。
- CI 新增独立的 macOS + Python 3.13 非浏览器兼容任务，并把它纳入 Pages 部署前置门禁。

### Changed

- 标量响应以 JSON `null` 表示 masked 格点，未遮罩项保留未按 `vmin`/`vmax` 裁剪的原始有限值；响应模型校验数组长度、mask/null 对应关系和色标边界。
- `density` 继续表示 6–40 的整数种子总预算，浏览器步长改为 1，使圆环奇数预算的轴线分支可达。
- 零强度磁偶极保留为响应 marker，但不再参与场、播种、排除区域、源间距或预算；全零 `magnetic_dipole`/`halbach_array` 请求返回 422。
- 数值位置编辑会回滚源间距冲突；拖动会吸附到合法位置，只在松开后请求一次，并在等待新场期间隐藏旧热图、色标和场线。

### Fixed

- 探针不再把色标百分位裁剪值冒充原始场值。
- 拖动源时不再把拖动前的场与拖动后的源位置混画。

## [0.2.1] - 2026-09-02

这是对已经交付的 v0.1.1–v0.2.1 工作进行的首个正式发布记录；它不把旧提交重新解释成曾经发布过的 tag。

### Added

- 建立 Ubuntu/Windows 多版本 CI、90% 分支覆盖率门槛、严格文档构建、GitHub Pages 部署及 wheel/sdist 隔离冒烟检查。
- 提供带事件终止、双向追踪和 opt-in 闭环检测的自适应场线积分器。
- 提供匀强场、点电荷、理想磁偶极、圆形电流线圈及复合场的批量解析求值。
- 新增圆环电流与八点面内磁偶极 Halbach 教学预设，以及无需构建工具链的浏览器可视化。
- 新增点源增删、拖动、磁偶极角度编辑、探针、键盘操作和 Python Playwright 前端语义测试。

### Changed

- HTTP 场景契约显式声明 SI 显示单位，并用逐源 `strength_unit` 区分 nC、A·m² 与 A。
- 电荷强度使用有符号数值；源类型、强度符号和 `angle_deg` 组合由严格请求模型校验。
- `density` 定义为整个场景的种子总预算，并在服务端与浏览器提交前共同校验。
- 场景元数据区分真实不变平面场线与投影流线，并公开播种说明和终止统计。

[Unreleased]: https://github.com/longwarriors/Vector_field_streamline/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/longwarriors/Vector_field_streamline/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/longwarriors/Vector_field_streamline/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/longwarriors/Vector_field_streamline/tree/v0.2.1
