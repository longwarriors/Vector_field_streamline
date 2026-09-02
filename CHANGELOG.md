# Changelog

本文件记录 VectorViz 面向使用者可见的功能与契约变化。项目在 v1.0 前仍允许修正公共契约；每次修正都会同步更新 API 文档与相关教程。

## [Unreleased]

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

[Unreleased]: https://github.com/longwarriors/Vector_field_streamline/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/longwarriors/Vector_field_streamline/tree/v0.2.1
