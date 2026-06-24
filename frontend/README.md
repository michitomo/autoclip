# autoclip frontend

Vite + React + TS の操作 UI。日付 → 委員会 → 議員 を選んでクリップを生成する。

## 開発

```bash
# 1) バックエンド (別ターミナル)
cd ../backend
.venv/bin/uvicorn src.app:app --reload --port 8000

# 2) フロントエンド
npm ci        # 初回
npm run dev   # http://localhost:5173
```

`vite.config.ts` の proxy が `/api` を `http://127.0.0.1:8000` に転送するので、
dev 中は同一オリジンとして API を叩ける。

## ビルド

```bash
npm run build   # tsc + vite build → dist/
```
