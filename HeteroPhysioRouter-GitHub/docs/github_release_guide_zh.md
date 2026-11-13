# PhysioGCR：命名、GitHub 简介与上传流程

## 一、统一名称

- 方法名和 GitHub 仓库名：**PhysioGCR**。
- 全称：**Physiological Group-Consistent Recognition**。
- 含义：基于传感器组一致性的生理信号识别。
- `Physio` 表示生理信号，`GCR` 对应 Group-Consistent Recognition。
- 正文第一次出现时写全称，后文、表格方法行和模型图统一写 `PhysioGCR`。

这是本研究提出的方法名称，不是已有领域术语。名称突出一个贯穿方法的原则：
用同一物理传感器分组组织表征、训练扰动和鲁棒性评价。多粒度表征、跨粒度交互、
质量估计和动态路由是实现中的组成部分；局部选择性修正是单独验证的机制扩展。

GitHub 仓库名称查询中未发现 `PhysioGCR` 同名仓库，但这不构成对所有论文、软件或
商标的穷尽查重。

## 二、创建仓库时填写什么

**Repository name**

```text
PhysioGCR
```

**Description**（可直接粘贴）

```text
Group-consistent multi-granular learning for robust physiological time-series recognition across Sleep-EDF, PAMAP2, and WESAD.
```

**Topics**（在仓库首页 About 设置中逐个添加）

```text
physiological-signals
time-series
multimodal-learning
wearable-sensing
sleep-staging
human-activity-recognition
missing-modality
pytorch
```

**仓库首页英文介绍**（README 已包含对应内容）

> PhysioGCR aligns the physical sensor groups used for representation,
> training-time corruption, and robustness evaluation. It retains sensor-group,
> temporal-patch, and global evidence through a multi-granular architecture and
> learns under structured group degradation. This repository provides the
> PyTorch implementation, controlled robustness experiments, and common-input
> benchmark adapters for Sleep-EDF, PAMAP2, and WESAD.

**中文介绍**（可用于中文项目说明）

> PhysioGCR 面向异构生理信号中的传感器缺失与信号退化，以传感器组一致性为设计原则，
> 将表征单元、训练扰动单元和评价单元对齐到同一物理分组。模型并行保留传感器组、
> 时间片段和全局证据，并通过跨粒度交互与组级扰动训练学习鲁棒表征。项目提供模型实现、
> 机制分析实验及 Sleep-EDF、PAMAP2、WESAD 的统一输入基准适配代码。

## 三、上传前修改的资料

当前整理目录是 `D:\HeteroPhysioRouter-GitHub`，无需为了上传而重命名本地目录。

1. 在 `CITATION.cff` 和 `pyproject.toml` 中，把 `OWNER` 换成 GitHub 用户名。
2. 把 `Anonymous authors` 换成实际作者信息；如果采用匿名发布则保留匿名信息并核对投稿政策。
3. 论文链接尚未公开时先不填写，不要写“TPAMI 已接收”等尚未发生的状态。
4. 确认 `THIRD_PARTY.md` 中的实现来源和许可证边界。
5. 数据、缓存和权重不随代码上传；README 已提供数据所有者的下载链接。

README 中推荐按以下顺序展示：方法介绍、框架图、安装、数据准备、最小运行示例、
完整 benchmark、复现边界、引用和许可证。最终 Fig. 1 确定后，再导出适合网页显示的
PNG 放入 `assets/` 并插入 README，无需把整个手稿和所有实验图上传。

## 四、推荐上传流程：Git + GitHub 网页

先安装 [Git for Windows](https://git-scm.com/download/win)，安装后重新打开 CMD。

在 GitHub 网页右上角选择 **New repository**，填入名称和 Description。
选择需要的可见性；**不要勾选自动添加 README、.gitignore 或 License**，因为本地已有这些文件。
创建空仓库后，使用下面的命令。

以下命令在 **CMD** 逐行执行；先把 `YOUR_USERNAME` 换成实际 GitHub 用户名。

```bat
cd /d D:\HeteroPhysioRouter-GitHub
git init
git branch -M main
git add .
git status --short
git diff --cached --stat
```

检查暂存内容主要是 `.py`、`.md`、`.toml`、`.json`、`LICENSE` 等代码与文档，
不应包含数据文件、checkpoint、环境目录或运行日志。然后执行：

```bat
git commit -m "Initial release of PhysioGCR"
git remote add origin https://github.com/YOUR_USERNAME/PhysioGCR.git
git push -u origin main
```

第一次提交如果 Git 提示缺少身份，按提示配置提交署名，例如：

```bat
git config user.name "YOUR_GITHUB_NAME"
git config user.email "YOUR_GITHUB_NOREPLY_EMAIL"
```

其中邮箱可使用 GitHub Settings → Emails 提供的隐私邮箱。浏览器登录或 Git Credential
Manager 会处理推送认证，不要把密码或 token 写入代码或仓库 URL。

## 五、发布 v0.1.0

上传代码后，在仓库网页选择 **Releases → Draft a new release**，填写：

- Tag：`v0.1.0`。
- Title：`PhysioGCR v0.1.0`。
- Release notes：使用下面的英文内容。

```text
Initial research code release of PhysioGCR.

Included:
- Multi-granular sensor-temporal representations and group-degradation training.
- Quality-aware routing and local correction diagnostics.
- Common-input benchmark adapters for Sleep-EDF, PAMAP2, and WESAD.
- Subject-disjoint evaluation, LOSO utilities, and metric aggregation.
- Dataset download instructions and documented third-party implementation sources.

Datasets and pretrained weights are not included. Literature-model adapters
are identified separately from exact original-paper reproductions.
```

## 六、后续更新

以后在当前目录修改代码后执行：

```bat
git add .
git diff --cached --stat
git commit -m "Update documentation and benchmark protocols"
git push
```

论文公开后增加论文 URL 和真实 BibTeX；结果表只引用已核验的实验记录，并写清完整输入、
缺失输入、失真输入和不同划分协议。Oracle 局部修正应标为机制诊断，不能展示成部署时可用的模型。
