# 第三方源码

`Retrieval-based-Voice-Conversion-WebUI` 是官方 RVC 仓库的浅克隆，用于：

- 核对当前实际存在的 CPU 依赖文件名（目前为 `requirments_cpu_py312.txt`）
- 在安装 `--with-rvc-cpu` / `-WithRvcCpu` 后加载标准 `.pth` 做实时推理（Windows 上该依赖包含 DirectML，默认走显卡；Linux 为 CPU）
- 作为后续训练自己的授权音色模型的上游工具

本项目不会把该仓库改造成默认必须完整安装才能做直通测试。
轻量直通环境使用项目根目录的 `.venv`；完整推理使用独立的 `.venv-rvc`。

如果目录不存在，可运行：

```bash
git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git \
  third_party/Retrieval-based-Voice-Conversion-WebUI
```
