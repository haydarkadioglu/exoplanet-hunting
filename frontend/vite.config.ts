import { defineConfig } from 'vite'

export default defineConfig({
  base: '/exoplanet-hunting/',
  root: '.',
  build: {
    outDir: 'dist',
  },
  server: {
    port: 3000,
    open: true
  },
  assetsInclude: ['**/*.onnx']
})