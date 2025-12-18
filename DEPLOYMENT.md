# 🚀 Deploying to Hugging Face Spaces

Deploy your Scientific RAG System to HF Spaces **for free** with **zero configuration** - directly from the main branch!

## ✨ What Makes This Seamless

- ✅ **Deploy from main branch** - no separate deployment branch needed
- ✅ **Auto-detects HF Spaces** - switches to in-memory Qdrant automatically
- ✅ **No environment variables required** - users provide API keys in UI
- ✅ **No Docker needed** - runs entirely on free CPU tier
- ✅ **Pre-configured** - all deployment files in root directory

## 📋 Prerequisites

1. **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co)
2. **Git**: Ensure you have git installed locally

## 🎯 Deployment Methods

### Method 1: Direct Git Push (Recommended)

1. **Create Your Space** at [huggingface.co/new-space](https://huggingface.co/new-space):
   - **Name**: `scientific-rag` (or your choice)
   - **SDK**: Gradio
   - **License**: MIT
   - Click **Create Space**

2. **Add Space as Remote** (from your local repo):
   ```bash
   cd /path/to/scientific-rag
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/scientific-rag
   ```

3. **Push to Deploy**:
   ```bash
   git push space main
   ```

That's it! Your space will build and launch automatically in 2-3 minutes.

### Method 2: GitHub Integration

1. **Create Your Space** at [huggingface.co/new-space](https://huggingface.co/new-space)

2. **Connect GitHub**: In Space settings → "Linked Repositories"
   - Link your GitHub repository
   - Select `main` branch
   - Enable auto-deploy on push

3. **Done!** Every push to main will auto-deploy to Spaces

### Method 3: Clone and Push

```bash
# Clone the space
git clone https://huggingface.co/spaces/YOUR_USERNAME/scientific-rag
cd scientific-rag

# Pull from your repo
git pull https://github.com/YOUR_USERNAME/scientific-rag main

# Push to space
git push origin main
```

## 📁 Key Files for Deployment

All deployment files are now in the root directory:

- **`app.py`** - Main Gradio application (auto-detects Spaces environment)
- **`requirements.txt`** - Python dependencies for Spaces
- **`data/processed/chunks_arxiv.json`** - Pre-processed document chunks
- **`src/`** - Application source code

## 💡 No Secrets Needed

Users provide their own API keys in the UI - no environment variables or secrets to configure in Space settings!

## ⚙️ Configuration for Optimal Performance

### 1. Vector Database
- The app uses **in-memory Qdrant** on Spaces (no Docker needed)
- Pre-processed chunks are loaded from `data/processed/chunks_arxiv.json`
- Embeddings are computed on first run (may take 1-2 minutes)

### 2. Dataset Size
For faster startup on free tier, use a **subset** (recommended: 1000-5000 chunks).

Modify in [src/scientific_rag/settings.py](src/scientific_rag/settings.py):
```python
dataset_sample_size: int = 1000  # Limit for faster loading
```

Then regenerate chunks:
```bash
make chunk-data
```

### 3. Model Selection
Use **lighter models** for free tier CPU:
- Embeddings: `intfloat/e5-small-v2` (default - good balance)
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2` (default)
- LLM: Free models via OpenRouter (e.g., `meta-llama/llama-3.3-70b-instruct:free`)

All configurable in [src/scientific_rag/settings.py](src/scientific_rag/settings.py).

### 4. Startup Time
- **First launch**: 2-3 minutes
  - Download models (~200MB)
  - Load and index chunks
  - Initialize pipeline
- **Subsequent queries**: Fast (~2-5 seconds)
- **After sleep**: ~30 seconds to wake up

## 🎨 Customize Your Space

### Update README Header
Add YAML frontmatter to the top of README.md (for Spaces):

```yaml
---
title: Scientific RAG System
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
tags:
  - rag
  - scientific-papers
  - arxiv
  - retrieval
  - question-answering
python_version: 3.11
---
```

### Change UI Theme
In [app.py](app.py), modify the Blocks initialization:

```python
with gr.Blocks(theme=gr.themes.Soft(), title="Scientific RAG System") as demo:
```

Try different themes: `Soft`, `Base`, `Glass`, `Monochrome`, etc.

## 📊 Monitor Your Space

- **Logs**: View real-time logs in the Space web interface (top right)
- **Usage**: Check visitor stats in Space settings
- **Sleep Mode**: Free spaces sleep after inactivity (~48 hours)
  - Restart automatically on next visit
  - No data loss (models re-download on wake)

## 🐛 Troubleshooting

### Build Fails
**Problem**: Space shows build error

**Solutions**:
- Check build logs for missing dependencies
- Ensure `requirements.txt` is up to date
- Verify Python version compatibility (3.11+)
- Check file paths are correct (case-sensitive)

### Out of Memory
**Problem**: Space crashes with OOM error

**Solutions**:
- Reduce dataset size: set `dataset_sample_size = 500` in settings
- Use smaller embedding models
- Disable reranking temporarily
- Pre-compute embeddings locally and commit them

### Slow Performance
**Problem**: Queries take too long

**Solutions**:
- **Pre-process data**: Commit `chunks_arxiv.json` to skip processing
- **Reduce retrieval**: Lower `retrieval_top_k` to 5-10
- **Faster models**: Use `all-MiniLM-L6-v2` for embeddings
- **Cache embeddings**: They persist across queries after first run

### Module Import Errors
**Problem**: `ModuleNotFoundError` in logs

**Solutions**:
- Ensure all imports use absolute paths from `src/`
- Check `sys.path` is set correctly in `app.py`
- Verify package structure is intact

### Data Not Loading
**Problem**: No chunks available, retrieval fails

**Solutions**:
- Ensure `data/processed/chunks_arxiv.json` exists and is committed
- Check file size isn't truncated by Git LFS limits
- Re-run `make chunk-data` locally if needed

## 💰 Upgrade Options

### Free Tier Limitations
- **CPU only** (no GPU)
- **16GB RAM** (sufficient for most use cases)
- **50GB disk** (adequate for datasets + models)
- **Sleeps after inactivity** (~48 hours)

### Paid Tier Benefits ($0.03-1.00/hour)
- **GPU available** (faster embeddings and reranking)
- **More RAM** (32GB-256GB)
- **Always on** (no sleep mode)
- **Custom domains**

Upgrade in Space settings → "Hardware"

## 📚 Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs)
- [Spaces GPU Guide](https://huggingface.co/docs/hub/spaces-gpus)
- [Spaces Secrets Management](https://huggingface.co/docs/hub/spaces-overview#managing-secrets)

## 🎯 Production Checklist

Before deploying to production:

- [ ] Test locally with `make run-app`
- [ ] Reduce dataset size for free tier (`dataset_sample_size = 1000`)
- [ ] Pre-process and commit chunks (`data/processed/chunks_arxiv.json`)
- [ ] Test with limited resources (simulate CPU-only)
- [ ] Add error handling for API failures
- [ ] Update README with your Space URL
- [ ] Set appropriate rate limits if needed
- [ ] Monitor first few days for issues

## 🔄 Updating Your Deployment

To update your deployed Space:

```bash
# Make changes locally
git add .
git commit -m "Update: your changes"

# Push to both GitHub and Spaces
git push origin main
git push space main
```

Space will rebuild automatically (2-3 minutes).

---

**Questions?** Open an issue on [GitHub](https://github.com/DenkoProg/scientific-rag/issues)
