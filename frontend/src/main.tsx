import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// 統合UI 一本 (corpus.json + catalog.json 駆動・サーバー無しでブラウザレンダ)。
createRoot(document.getElementById('root')!).render(
  <StrictMode><App /></StrictMode>,
)
