---
name: クリップ生成リクエスト
about: 衆議院TV のある議員の質疑クリップを生成してほしいときに使います
title: "[clip] 委員会 / 議員名"
labels: clip-request
---

衆議院TV クリップの生成リクエストです。

- 日付: YYYY-MM-DD
- 委員会: ○○委員会
- 議員: ○○

```yaml
session_id: "00000"
member: 議員名
```

<!-- 上の YAML ブロックを clip-request ワークフローが読みます。session_id と member だけ正しければOKです。 -->
