# 训练底模约定

训练需要一对官方风格的 **生成器 G** 和 **判别器 D** 作为起点。默认仍是 RVC v2 40k：

- `assets/pretrained_v2/f0G40k.pth`
- `assets/pretrained_v2/f0D40k.pth`

以后有更好的底模，按下面方式替换即可，不必改代码。这些是训练起点，不是用户音色；不要把用户 `.pth` 或他人声音模型放进仓库。

## 放置路径（扫描顺序）

1. `assets/pretrained_custom/`：自行投放或从界面导入，不会提交到仓库
2. `assets/pretrained_v2/`：官方 RVC v2
3. `assets/pretrained/`：官方 RVC v1

也会回退扫描 `third_party/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained_v2` 与 `pretrained`。

## 文件名

必须成对出现：

| 用途 | 生成器 | 判别器 |
| --- | --- | --- |
| 带音高（推荐，本工具默认） | `f0G40k.pth` | `f0D40k.pth` |
| 其它采样率 | `f0G32k.pth` / `f0G48k.pth` | `f0D32k.pth` / `f0D48k.pth` |
| 无音高变体 | `G40k.pth` | `D40k.pth` |

自定义子目录（例如 `assets/pretrained_custom/MyBetterV2/`）里也可以用 `G.pth` + `D.pth`，或任意 `*G*.pth` 与对应的 `*D*.pth`。可选 `meta.json`：

```json
{"version": "v2", "sample_rate": "40k", "label": "显示在下拉框里的名称"}
```

放入或导入后，在训练页点「刷新底模」，新组合会出现在下拉框中。

## 界面导入

在「一键训练」里选择生成器（G）和判别器（D）两个 `.pth`，再点导入。文件会复制到 `assets/pretrained_custom/<名称>/`。

## 官方缺省底模

若目录是空的，可运行：

```bash
./scripts/download_base_models.sh --with-pretrained
```

Windows 便携包已带默认 v2 40k 底模。若要训练 32k / 48k 或 v1，再下载对应文件，或自行放入上述目录。
