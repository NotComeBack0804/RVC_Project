# 开源测试音色（官方演示模型）

这些是 RVC 官方发行包（`RVC*.7z`）里附带的演示音色，经 [licyk/rvc-model](https://huggingface.co/licyk/rvc-model) 镜像下载，并按该仓库 `manifest` 中的 SHA-256 校验。

许可见同目录 `LICENSE-RVC.txt`：上游为 MIT，并附带「仅供研究使用，使用及传播导出声音者自负全责」的条件。只用于测试本项目的转换链路，不要把输出冒充成真人原声，也不要用未授权的他人声音。

| 文件 | 用途 | 版本 | 采样率 | SHA-256 |
| --- | --- | --- | --- | --- |
| `assets/weights/kikiV1.pth` | 官方演示音色 | v1 | 40 kHz | `fb48c0a982317073c343e440e0c88337c3c605908429d81d420ea123bc83b5b5` |
| `assets/indices/kikiV1.index` | 对应检索索引 | v1 | 40 kHz | `5f20e83f2e232f2077fed5a064bc92865899e2330a1ff10f2c54b2bae834c647` |
| `assets/weights/keruanV1.pth` | 官方演示音色 | v1 | 40 kHz | `9b3808d83e99b7c7b01cc1e33cee79ab8f1ebed08708b72066917465ea336e56` |
| `assets/indices/keruanV1.index` | 对应检索索引 | v1 | 40 kHz | `8e8d2000be4c6dce12102fe417e03ac13c338328da424e33a8e71d94eb1b3380` |

同一来源里还有未下载的官方演示音色：`guanguanV1`、`zhanzhanv2-xi`（索引力约 239 MB）、`youzhanv2-xi`（无公开索引）。

注意：`pretrained/`、`pretrained_v2/` 里的 `f0G40k.pth` 等是训练底模，不是目标音色，不能当作实时变声的说话人模型加载。

## Shadowww T02（用户提供，说话人本人发布）

用户于 2026-11-24 发来的压缩包，内容为 [ShadowSH2306/shadowww-voices](https://huggingface.co/ShadowSH2306/shadowww-voices) 的 T02。已与官方 `SHA256SUMS` 核对一致。

| 文件 | SHA-256 |
| --- | --- |
| `assets/weights/shadowww-t02.pth` | `7166e6cc680178a1273929ebe00daedab1f8db8039bec312c96a23093ef2ac20` |
| `assets/indices/added_shadowww-t02.index` | `1b9d1b7168f89176ca51a937b1ae595fa6dab357745b05d1bd06a8c471a4e6fb` |

许可为 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)：可个人测试与非商用，需署名 “Voice model: Shadowww”；不得商用（除非另行获得 Shadowww 许可）；不得把输出冒充成 Shadowww 本人的真实录音，并应标明为 AI 生成。RVC v2，48 kHz，偏自然偏低的音色。
