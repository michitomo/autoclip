import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// dev サーバーは /api を autoclip バックエンド (FastAPI :8000) に転送する。
// これで CORS を気にせず同一オリジンとして叩ける。
// 外部公開 (ngrok / Tailscale 等) 用に全インターフェース bind + 任意ホスト許可。
export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // 0.0.0.0 で listen (LAN / Tailscale / ngrok から到達可能に)
    // ngrok ドメイン + Tailscale IP を許可。
    // (Vite は既定で未知ホストを Blocked request で弾くため)
    allowedHosts: [
      '.ngrok-free.app', '.ngrok.app', '.ngrok.io', '.ngrok-free.dev',
      '100.118.210.81', // Tailscale (このMac)
    ],
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
