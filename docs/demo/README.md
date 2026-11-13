# 在线工作台

当前 Gradio 前端的静态交互展示，使用 `examples/synthetic/history` 的人工合成成功和失败记录。

- 入口：`index.html`，失败页：`failure.html`。
- 支持历史切换、报告/证据/时间线标签，以及报告和 Trace 的独立滚动。
- 页面不加载 Gradio 客户端、不连接后端、不发出模型或搜索请求。真实调查使用项目的本地启动入口和许可门卫。
- 截图与页面来自同一轮实际渲染；前端布局变化时应重新导出，不把静态展示当作新的真实运行。

GitHub Pages 使用 `main` 分支的 `/docs` 目录；公开入口为 `/veripulse-multi-agent/demo/`。
