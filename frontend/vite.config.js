import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000
  },
  build: {
    outDir: 'dist' // This will output your production build to the dist folder.
    // No need to specify an input; Vite will automatically use the index.html in the project root.
  }
})
