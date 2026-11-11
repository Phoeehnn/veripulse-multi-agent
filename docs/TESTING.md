# 公开版离线验证

Python 3.11.9 / Windows，全新虚拟环境。最终精选测试193通过、0失败、0跳过；pip check通过。

| 测试模块 | 用例数 |
|---|---|
| `test_experiment_budget` | 41 |
| `test_experiment_budget_recovery` | 27 |
| `test_fact_v2_quote_verifier` | 4 |
| `test_harness_budget` | 33 |
| `test_harness_ledger_verifier` | 18 |
| `test_persistent_experiment_budget` | 31 |
| `test_public_demo` | 3 |
| `test_validate` | 35 |
| `test_youtube_comments` | 1 |

选取原项目预算、引用、协作门卫和持久化钱包测试；YouTube时间窗及去重使用无真实身份的合成材料。未附原始CHEF/CED数据、历史场次和依赖这些材料的测试。

先在全新环境只安装requirements-demo.txt，运行三个演示专项并通过真实本机浏览器打开scripts/run_public_demo.py；再安装requirements-test.txt，运行全部精选测试。演示不加载模型权重，也不调用搜索服务。macOS/Linux提供等价命令，本轮实际验收平台为Windows。

UI导入隔离检查使用独立子进程，避免其他测试导入模型模块造成误判。网络检查允许本机回环，拒绝外部socket连接。测试失败与修复记录保留在本地验收档案，不作为模型性能实验。

单次pytest会使用新的traces/tmp目录。需要保留证据时先归档报告与SHA；进程结束后仅清理该次目录。
