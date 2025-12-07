# Quick Start Guide - 100 Days of Data and AI

Get started in 10 minutes! 🚀

> **📝 Note for macOS users:** Use `python3` instead of `python` throughout this guide.

## Prerequisites

- Computer (Windows, Mac, or Linux)
- 20GB free disk space
- 16GB RAM recommended (8GB minimum)
- Internet connection
- GitHub account (optional, but recommended)

---

## Step 0: Fork the Repository (Recommended)

**Why fork?** Track your progress, build your portfolio, and practice Git!

### Option A: Fork (Recommended)

1. **Go to**: https://github.com/YOUR-ORG/100-days-data-ai
2. **Click "Fork"** button (top right)
3. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/100-days-data-ai.git
   cd 100-days-data-ai
   ```

### Option B: Download (No Git)

Download the ZIP file from GitHub and extract it.

> 💡 **Tip**: Using Git lets you commit your solutions daily and build a portfolio!

---

## Step 1: Install Python (5 minutes)

### Check if Python is already installed:
```bash
python --version
```

If you see `Python 3.11` or higher, skip to Step 2!

### Install Python:

**Windows:**
1. Download from [python.org/downloads](https://www.python.org/downloads/)
2. Run installer
3. ✅ **Check "Add Python to PATH"**
4. Click "Install Now"

**Mac:**
```bash
brew install python@3.11
```

**Linux:**
```bash
sudo apt update && sudo apt install python3.11 python3-pip python3-venv
```

---

## Step 2: Create Virtual Environment (2 minutes)

Open terminal/command prompt in the project folder:

```bash
# Create virtual environment
# On Mac/Linux:
python3 -m venv venv

# On Windows:
python -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

---

## Step 3: Install Core Packages (3 minutes)

With the virtual environment activated:

```bash
# Mac/Linux:
python3 -m pip install -r requirements.txt
# OR:
pip install -r requirements.txt

# Windows:
pip install -r requirements.txt
```

This installs the core packages needed for Phase 1 (Days 1-35):
- numpy, pandas, matplotlib, seaborn
- jupyter, plotly, scipy
- duckdb, statsmodels

---

## Step 4: Verify Setup (1 minute)

```bash
python resources/test_setup.py
```

You should see:
```
✅ Python version: 3.11.x
✅ numpy installed
✅ pandas installed
✅ matplotlib installed
...
✅ Core setup complete! Ready to start Phase 1
```

---

## Step 5: Start Day 1! 🎉

```bash
cd days/day-001-csv-vs-json
```

Open `README.md` to begin learning!

---

## Daily Routine

**Before starting each day:**
```bash
# Activate virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

**Then follow this pattern:**

1. **📖 Read** the `README.md` (10 min)
   - Learn the concepts
   - Study the examples

2. **💻 Code** the `exercise.py` (40 min)
   - Complete the exercises
   - Try before looking at solutions

3. **✅ Review** the `solution.py` (if stuck)
   - Compare your approach
   - Learn from examples

4. **🎯 Quiz** with `quiz.md` (10 min)
   - Test your understanding
   - Review if needed

**When done:**
```bash
deactivate  # Exit virtual environment
```

---

## Phase-by-Phase Setup

You don't need to install everything at once! Install packages as you progress:

### Phase 1: Data Engineering (Days 1-35)
✅ Already installed with `requirements.txt`

### Phase 2: Development Tools (Days 36-50)
```bash
# Install when you reach Day 44
pip install fastapi uvicorn pydantic pytest sqlalchemy psycopg2-binary
```

**Docker required for Days 44-49:**
- Download from [docker.com](https://www.docker.com/products/docker-desktop)

### Phase 3: Machine Learning (Days 51-70)
```bash
# Install when you reach Day 57
pip install scikit-learn torch torchvision transformers mlflow
```

### Phase 4: GenAI/LLMs (Days 71-92)
```bash
# Install when you reach Day 71
pip install langchain langchain-community chromadb faiss-cpu ollama langgraph
```

**Install Ollama for local LLMs:**
- Mac/Linux: `curl -fsSL https://ollama.com/install.sh | sh`
- Windows: Download from [ollama.com](https://ollama.com/download)

Then pull a model:
```bash
ollama pull llama2
```

### Phase 5: Specialized Topics (Days 93-100)
```bash
# Install when you reach Day 93
pip install opencv-python pillow whisper
```

---

## Recommended Tools

### Code Editor (Choose One):

**VS Code** (Recommended)
- Download: [code.visualstudio.com](https://code.visualstudio.com/)
- Install extensions: Python, Jupyter, Docker
- Free and feature-rich

**PyCharm Professional**
- Download: [jetbrains.com/pycharm](https://www.jetbrains.com/pycharm/)
- Full data science support
- Free for students

**Jupyter Lab**
```bash
pip install jupyterlab
jupyter lab
```

---

## Learning Tips

✅ **Code along** - Type examples yourself  
✅ **Take breaks** - 1 hour/day is the goal  
✅ **Practice more** - Experiment with variations  
✅ **Use the cheatsheet** - See `resources/cheatsheet.md`  
✅ **Track progress** - Commit daily to Git  
✅ **Join community** - Ask questions, share wins

---

## Mini Project Days (Extra Time)

These days include hands-on projects and may take 1.5-2 hours:
- **Day 7**: Format converter tool
- **Day 14**: Iceberg table operations
- **Day 21**: Medallion architecture
- **Day 28**: Spark ETL pipeline
- **Day 35**: Streaming pipeline
- **Day 42**: Data quality pipeline
- **Day 49**: Dockerized app
- **Day 56**: Data API
- **Day 63**: ML model with MLflow
- **Day 70**: Image classifier
- **Day 77**: Prompt engineering
- **Day 84**: RAG system
- **Day 91**: AI agent
- **Day 98**: Integration project

Don't rush these - they're where you consolidate learning!

---

## Troubleshooting

### "Python not found"
- Restart terminal after installation
- Try `python3` instead of `python`
- Check PATH environment variable

### Virtual environment not activating
- Make sure you're in the project folder
- On Windows: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### macOS: "externally-managed-environment" error
**Solution: Use `python3` instead of `python`**
```bash
python3 -m pip install -r requirements.txt
```

### Package installation fails
```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Docker permission denied (Linux)
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### Spark Java error
```bash
# Install Java 11
brew install openjdk@11  # macOS
sudo apt install openjdk-11-jdk  # Ubuntu

# Set JAVA_HOME
export JAVA_HOME=$(/usr/libexec/java_home)  # macOS
```

### Out of memory
- Close other applications
- Use smaller datasets for practice
- Reduce Spark memory: `export SPARK_DRIVER_MEMORY=2g`
- Use smaller LLM models (phi instead of llama2)

### Still stuck?
1. Check [resources/setup-guide.md](./resources/setup-guide.md)
2. Check [resources/cheatsheet.md](./resources/cheatsheet.md)
3. Google the error message
4. Ask in community forums

---

## Daily Workflow (If Using Git)

### Each Day:

1. **Start the day**:
   ```bash
   cd 100-days-data-ai
   source venv/bin/activate
   ```

2. **Complete the exercises** in `days/day-XXX-topic/exercise.py`

3. **Commit your work**:
   ```bash
   git add days/day-XXX-topic/
   git commit -m "Complete Day XXX: Topic Name"
   git push origin main
   ```

### Sample Commit Messages:
```bash
git commit -m "Complete Day 001: CSV vs JSON"
git commit -m "Complete Day 007: Mini Project - Format Converter"
git commit -m "Complete Day 050: Checkpoint - Data Engineering Review"
git commit -m "Complete Day 100: Capstone Review"
```

### Track Your Progress:
- GitHub profile shows daily commits (green squares!)
- Build a portfolio employers can see
- Backup all your work

---

## Learning Paths

### Path 1: Sequential (Recommended)
Complete all 100 days in order for comprehensive coverage.

### Path 2: Data Engineering Focus
- Days 1-50 (Data Engineering + Development)
- Skip to relevant projects

### Path 3: AI/ML Focus
- Days 1-7 (Data basics)
- Days 51-100 (ML + GenAI)

### Path 4: GenAI Focus
- Days 1-7 (Data basics)
- Days 51-63 (ML fundamentals)
- Days 71-100 (GenAI + LLMs)

---

## What's Next?

After completing 100 days, you'll be ready for:
- **Portfolio Projects**: Build 8 data engineering + 10 AI projects
- **Advanced Bootcamp**: 50 Days of Advanced Data and AI
- **Comprehensive Projects**: Full-stack data/AI applications
- **Job Applications**: You'll have skills + portfolio

---

## Quick Reference

- 📖 **Full Setup Guide**: [resources/setup-guide.md](./resources/setup-guide.md)
- 📝 **Cheatsheet**: [resources/cheatsheet.md](./resources/cheatsheet.md)
- 🧪 **Test Setup**: `python resources/test_setup.py`
- 📚 **Curriculum**: [CURRICULUM.md](./CURRICULUM.md)
- 🏠 **Main README**: [README.md](./README.md)

---

## System Requirements by Phase

| Phase | RAM | Disk | GPU | Docker |
|-------|-----|------|-----|--------|
| Phase 1 (Days 1-35) | 8GB | 5GB | No | No |
| Phase 2 (Days 36-50) | 8GB | 10GB | No | Yes |
| Phase 3 (Days 51-70) | 16GB | 15GB | Recommended | No |
| Phase 4 (Days 71-92) | 16GB | 20GB | Recommended | No |
| Phase 5 (Days 93-100) | 16GB | 20GB | Recommended | No |

---

## Community & Support

- **Discord**: [Join here](#)
- **GitHub Discussions**: Ask questions
- **Weekly Office Hours**: Live Q&A
- **Project Showcase**: Share your work

---

**Ready? Let's start with Day 1!** 🚀

```bash
cd days/day-001-csv-vs-json
cat README.md
```

Begin your 100-day journey to becoming a Data & AI engineer!
