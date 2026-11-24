# 自定义训练底模

把成对的生成器（G）和判别器（D）放在这里，然后在界面里点「刷新底模」即可选用，不必改代码。

推荐文件名：`f0G40k.pth` 与 `f0D40k.pth`（也可为 `32k` / `48k`）。

两种放法：

1. 直接放在本目录：`assets/pretrained_custom/f0G40k.pth` 和 `f0D40k.pth`
2. 建一个子目录，例如 `assets/pretrained_custom/MyBetterV2/`，里面放同一对文件。也可再放 `meta.json`：

```json
{"version": "v2", "sample_rate": "40k", "label": "我的新底模"}
```

也可在训练页用「导入底模」上传这两个 `.pth`，会自动复制到本目录的子文件夹中。

这里只放训练起点（pretrained G/D），不要放用户音色或他人声音模型。本目录下的模型文件不会提交到仓库。
