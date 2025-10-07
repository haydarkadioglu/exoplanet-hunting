import { defineConfig } from 'vite'

export default defineConfig({
  base: '/',
  root: '.',
  build: {
    outDir: 'dist',
    minify: false, // Disable minification for debugging
    rollupOptions: {
      output: {
        manualChunks: undefined,
      },
    },
  },
  server: {
    port: 3000,
    open: true
  },
  assetsInclude: ['**/*.onnx']
})