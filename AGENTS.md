# 测试与验证

- 按改动风险选最小、最快且足以验证行为的测试。不为每个函数机械地加单元测试，不重复测已被可靠覆盖的行为。
- 修缺陷，或改积分、播种、奇点处理、API 请求/响应契约等关键逻辑时，为重要失败场景留针对性回归测试；修缺陷的测试须确认在修复前会失败。
- 浏览器测试（`@pytest.mark.browser`，headless Chromium）只用于关键前端流程和前后端连接。
- 开发中只跑受影响的测试（如 `uv run pytest tests/test_x.py --no-cov`），不在每次编辑后跑全套。
- 提交前按 CI 顺序跑一次：`uv run ruff check .`、`uv run pytest`（含浏览器测试与 90% 覆盖率门槛；首次需 `uv run playwright install chromium`）、`uv run mkdocs build --strict`。
- CI（`.github/workflows/ci.yml`）另在 Windows/macOS/Ubuntu × Python 3.11/3.14 上跑 `-m "not browser"`，并对 wheel/sdist 做冒烟测试；改动打包或依赖时本地也跑 `uv build --clear` 与 `tests/package_smoke.py`。
- 报告实际跑了什么、结果如何、哪些没验证。
