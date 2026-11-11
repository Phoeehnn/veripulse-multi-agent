# 本地安装与离线演示

需要 Python 3.11。下载或克隆仓库后进入 `project-a` 目录。下列命令使用仓库相对路径；请将项目与缓存放在自己选择的位置。

## Windows PowerShell

```powershell
Set-Location path/to/project-a
$env:PYTHONDONTWRITEBYTECODE='1'
$env:GRADIO_ANALYTICS_ENABLED='False'
$env:TEMP="$PWD/traces/install-temp"
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR="$PWD/traces/pip-cache"
New-Item -ItemType Directory -Force $env:TEMP,$env:PIP_CACHE_DIR | Out-Null
py -3.11 -m venv .venv-demo
& ./.venv-demo/Scripts/python.exe -m pip install -r requirements-demo.txt
& ./.venv-demo/Scripts/python.exe -B scripts/run_public_demo.py
```

## macOS / Linux

```bash
cd path/to/project-a
export PYTHONDONTWRITEBYTECODE=1 GRADIO_ANALYTICS_ENABLED=False
mkdir -p traces/install-temp traces/pip-cache
export TMPDIR="$PWD/traces/install-temp"
export TEMP="$TMPDIR" TMP="$TMPDIR" PIP_CACHE_DIR="$PWD/traces/pip-cache"
python3.11 -m venv .venv-demo
.venv-demo/bin/python -m pip install -r requirements-demo.txt
.venv-demo/bin/python -B scripts/run_public_demo.py
```

打开 http://127.0.0.1:7883 。选择合成成功或失败案例，查看三个角色、报告、E/W 引文与时间线。退出用 Ctrl+C。端口被占用时停止自己的旧服务，或修改启动器端口。

页面复用真实项目的展示组件，合成数据不需要密钥、模型权重或第三方数据。输入框和调查按钮不执行真实任务；completed、passed、零费用及耗时为人工夹具字段。安装会下载开源依赖，启动后的演示不调用模型与搜索服务，服务只监听本机。

## 演示专项与核心离线测试

只安装演示依赖即可运行三个演示测试：

```powershell
# Windows
& ./.venv-demo/Scripts/python.exe -B tests/test_public_demo.py
```

```bash
# macOS / Linux
.venv-demo/bin/python -B tests/test_public_demo.py
```

预算、引用、仲裁门卫和持久化钱包等测试需要额外的导入依赖：

```powershell
& ./.venv-demo/Scripts/python.exe -m pip install -r requirements-test.txt
& ./.venv-demo/Scripts/python.exe -B -m pytest -q
```

```bash
.venv-demo/bin/python -m pip install -r requirements-test.txt
.venv-demo/bin/python -B -m pytest -q
```

测试使用合成数据和专属临时目录，禁用外部socket连接，允许事件循环所需的本机回环。无真实密钥或CHEF/CED数据要求。`chromadb` 等依赖用于加载现有模块；测试不下载Embedding/Reranker模型，也不运行付费网关。

## 完整调查与研究资产

`requirements.txt` 描述完整研究依赖，与轻量演示分开。真实调查还需要检索资源、外部凭证及匹配案件的运行许可，不能只填密钥就绕过门卫。默认从仓库根目录 `.env` 读取配置，也可设置 `PROJECTA_ENV_FILE` 指向自备文件；进程环境变量优先。所有子进程使用当前 `sys.executable`，需在具备对应依赖的环境中启动。

本次跨平台说明针对公开离线演示；Windows上完成实际验证，macOS/Linux提供等价启动命令。原真实调查入口的磁盘范围校验仍保留，本轮没有修改其运行策略，也未进行跨平台真实联网验收。

`experiments/archive/` 是依赖未公开数据的历史实验运行器，不属于演示步骤。题单、真实原文、钱包、旧许可不随仓库分发。源代码采用MIT许可，第三方依赖和数据各自保留其许可。
