# autoclip

衆議院TV（shugiintv.go.jp）の質疑答弁から、特定の議員のソーシャルメディア用クリップを自動生成するローカルファースト Webアプリ。

**ワークフロー**: 日付（既定: 今日）→ 委員会 → 議員 →（任意でトピック）を選ぶと、字幕付きクリップを生成する。
出力は (a) 質疑答弁のフル版、(b) Q&A 単位の切り抜きハイライト（JetCut でフィラー・無音を除去）。ハイライトは公開前に人手レビューする。

衆議院TV事務局より自己作成の許諾を確認済み。著作権法第40条1項（公開された政治上の演説の自由利用）＋第48条（出所明示）に基づき利用し、全出力に出所リンクを表示する。

既存の国会文字起こし・構造化パイプライン（scrape → Whisper → LLM 構造化）のコードを土台にしている。autoclip はその上に **動画クリップ + JetCut + 字幕焼き込み + UI/API** レイヤーを足したもの。

## 構成

```
autoclip/
  backend/    # Python (FastAPI) — 文字起こしパイプライン + 動画クリップ生成
  frontend/   # Vite + React + TS — 4ステップウィザード + JetCut レビュー UI
  data/       # セッション別 JSON (metadata/transcript/qa_pairs ...)
  media/      # ダウンロードした動画・生成クリップ
```

## セットアップ（backend）

```bash
cd backend
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
cp .env.example .env   # DEEPINFRA_API_KEY と OPENROUTER_API_KEY を設定
# ffmpeg が必要: brew install ffmpeg (macOS) / apt install ffmpeg (Linux)

# テスト
.venv/bin/python -m pytest -m "not integration" -q
```

ASR は **DeepInfra `whisper-large-v3-turbo`**（語単位タイムスタンプ対応を確認済み）。`ASR_PROVIDER=groq` で Groq に切替可能。
