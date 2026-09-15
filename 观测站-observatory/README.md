# 前沿观测站 / Frontier Observatory

> **定位**：把“快速变化的 2026 前沿”从静态 Markdown 里拆出来，建立机器可读、可审计、可定期复核的证据层。

教材中的稳定机制应多年可读；模型版本、协议版本、产品形态、系统能力却可能每几周变化。两者如果混在同一张手工表里，教材会发生 silent drift：正文仍然看起来完整，但代表模型、协议版本或工程事实已经过期。

本目录负责三件事：

```text
官方一手来源
    ↓
frontier-snapshot.json
    ↓
frontier_drift_audit.py
    ↓
教材第 12/17 篇、质量看板、实验路线的更新提示
```

## 文件

- [`frontier-snapshot.json`](frontier-snapshot.json)：当前机器可读快照；
- [`../工具-scripts/前沿漂移审计-frontier_drift_audit.py`](../工具-scripts/前沿漂移审计-frontier_drift_audit.py)：schema / freshness 审计；
- [`../教材-book/12-2026前沿-frontier-2026.md`](../教材-book/12-2026前沿-frontier-2026.md)：面向读者的前沿解释；
- [`../教材-book/17-闭源前沿模型与证据边界-frontier-closed-models.md`](../教材-book/17-闭源前沿模型与证据边界-frontier-closed-models.md)：闭源证据边界。

## 证据规则

观测站只记录：

1. 官方论文 / technical report；
2. 官方模型卡 / system card；
3. 官方产品 / developer 文档；
4. 官方规范；
5. 官方开源仓库。

厂商 benchmark 只能记录为“官方报告的结果”，不自动升级为独立第三方事实。闭源模型未披露的参数量、层数、MoE 结构、训练 recipe 一律标记为 `unknown`，不能通过社区猜测补齐。

## Freshness

`frontier-snapshot.json` 有 `as_of` 与 `revalidate_after_days`。CI 默认做 schema + age report，不因为互联网暂时不可用而阻塞仓库；需要发布新版本时可以本地执行严格 freshness 检查：

```bash
python 工具-scripts/前沿漂移审计-frontier_drift_audit.py --strict-age
```

这使“前沿章节是否过期”从感觉问题变成可见状态。
