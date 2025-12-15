# 🚀 Deploying to Hugging Face Spaces

Deploy your Scientific RAG System to HF Spaces **for free** with **zero configuration**.

> 📁 **Note**: All deployment files are in the `ops/` folder for better organization.

## ✨ What Makes This Seamless

- ✅ **No app.py wrapper needed** - uses [demo/main.py](demo/main.py) directly
- ✅ **Auto-detects HF Spaces** - switches to in-memory Qdrant automatically
- ✅ **No environment variables required** - users provide API keys in UI
- ✅ **No Docker needed** - runs entirely on free CPU tier

## 📋 Prerequisites

1. **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co)
2. **Git with LFS**: Install [Git LFS](https://git-lfs.github.com/)

## 🎯 Quick Deployment (3 Steps)

### 1. Create Your Space

Go to [huggingface.co/new-space](https://huggingface.co/new-space):
- **Name**: `scientific-rag` (or your choice)
- **SDK**: Gradio
- **License**: MIT
- Click **Create Space**

### 2. Clone & Copy Files

```bash
# Clone the space
git clone https://huggingface.co/spaces/YOUR_USERNAME/scientific-rag
cd scientific-rag

# Copy everything from your project
cp ../scientific-rag/ops/requirements.txt .
cp ../scientific-rag/ops/README_SPACES.md README.md
cp -r ../scientific-rag/src .
cp -r ../scientific-rag/demo .
mkdir -p data/processed
cp ../scientific-rag/data/processed/chunks_arxiv.json data/processed/
```

### 3. Push to Deploy

```bash
git lfs install
git add .
git commit -m "Deploy Scientific RAG"
git push
```

**That's it!** Your space will build and launch automatically in 2-3 minutes.

## 🔄 Alternative: Direct Push from Repo

```bash
cd /path/to/scientific-rag

# Add HF Spaces as remote
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/scientific-rag

# Push directly (may need to filter large files)
git push space main
```

## 💡 No Secrets Needed

Users provide their own API keys in the UI - no environment variables or secrets to configure!

## ⚙️ Important Configuration Notes

### 1. Vector Database
- The app uses **in-memory Qdrant** (no Docker needed)
- Pre-processed chunks are loaded from `data/processed/chunks_arxiv.json`
- Embeddings are computed on first run (may take 1-2 minutes)

### 2. Dataset Size
- For faster startup, use a **subset of the dataset** (recommended: 1000-5000 chunks)
- Modify in [settings.py](src/scientific_rag/settings.py):
  ```python
  dataset_sample_size: int = 1000  # Limit for faster loading
  ```

### 3. Model Selection
- Use **lighter models** for free tier:
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (default)
  - Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (default)
  - LLM: Free models from OpenRouter (e.g., `meta-llama/llama-3.3-70b-instruct:free`)

### 4. Cold Start
- First launch takes 2-3 minutes to:
  - Download models
  - Load and index chunks
  - Initialize pipeline
- Subsequent queries are fast (~2-5 seconds)

## 🎨 Customize Your Space

### Update UI Theme
In [demo/main.py](demo/main.py), modify the `demo` object:
```python
demo = gr.Blocks(theme=gr.themes.Soft())  # Try: Soft, Base, Glass, etc.
```

### Change Space Appearance
Edit the YAML header in `README.md`:
```yaml
emoji: 🔬  # Change icon
colorFrom: blue  # Change gradient colors
colorTo: purple
```

## 📊 Monitor Your Space

- **Logs**: View real-time logs in the Space web interface
- **Usage**: Check visitor stats in Space settings
- **Sleep Mode**: Free spaces sleep after inactivity (restart on next visit)

## 🐛 Troubleshooting

### Space Fails to Build
- Check build logs for missing dependencies
- Ensure all files are committed
- Verify Python version (3.11+)

### Out of Memory
- Reduce dataset size: `dataset_sample_size: int = 500`
- Use smaller models
- Disable some features (e.g., reranking)

### Slow Performance
- Pre-compute and commit chunks: `data/processed/chunks_arxiv.json`
- Use faster embedding models
- Reduce `retrieval_top_k` and `rerank_top_k`

## 📚 Additional Resources

- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs)
- [Spaces GPU (paid)](https://huggingface.co/docs/hub/spaces-gpus)

## 💰 Upgrade Options

Free tier limitations:
- CPU only (no GPU)
- 16GB RAM
- Space sleeps after inactivity

**Upgrade to paid tier** for:
- GPU acceleration (starting at $0.60/hour)
- Persistent storage
- Always-on spaces
- More RAM/CPU

Go to your Space settings → `Hardware` to upgrade.
