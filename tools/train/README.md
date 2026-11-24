# 训练自己的授权音色模型（框架，默认不执行训练）

本目录只提供整理素材、检查环境和调用官方 RVC 训练流程的脚本框架。
**在你提供已授权的参考录音之前，不要开始训练。**

本工具不会为你下载或附带任何他人音色模型。

## 使用边界

- 只使用你本人的声音，或已获得明确授权的声音。
- 参考音频和训练过程应留在本地，不要上传到云端训练服务。
- 录音文件和训练后的 `.pth` 都应按敏感个人数据妥善保存。

## 参考音频建议

- 只有目标说话人，且已获得授权。
- 环境安静，麦克风距离稳定。
- 没有背景音乐、多人对话和明显回声。
- 避免削波、爆音、强混响和过度降噪。
- 包含自然说话、不同句式和适度的音高变化。
- 建议先准备 10–30 分钟干净语音进行测试。素材质量比单纯增加时长更重要。

把授权录音放到：

```text
testdata/authorized_reference/
```

## 推荐步骤（在直通链路跑通之后）

1. 整理授权参考录音，删除噪声和无关片段：

   ```bash
   python tools/train/prepare_dataset.py --i-have-authorization
   ```

2. 确认已安装完整 RVC 环境（Windows 为 DirectML，Linux 为 CPU），并已下载训练底模：

   ```bash
   ./scripts/install_linux.sh --with-rvc-cpu
   ./scripts/download_base_models.sh --with-pretrained
   ```

3. 查看特征提取与训练说明（脚本默认只做检查，不会在数据或环境不完整时开始训练）：

   ```bash
   python tools/train/extract_features.py
   python tools/train/train_model.py
   ```

4. 导出的 `.pth` 放到 `assets/weights/`，可选 `.index` 放到 `assets/indices/`。
5. 再回到实时程序做对比测试。

Windows + AMD 显卡（例如 RX 7800 XT）上，完整安装后训练默认走 DirectML；只有探测失败时才回退 CPU。Linux 开发环境使用 CPU。不要把 CUDA 配置照搬到 AMD 环境。
