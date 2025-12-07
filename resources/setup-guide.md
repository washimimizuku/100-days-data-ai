# Setup Guide - 100 Days of Data and AI

Complete setup instructions for all phases of the bootcamp.

## Prerequisites

- **Computer**: Windows, macOS, or Linux
- **RAM**: 16GB+ recommended (8GB minimum)
- **Disk Space**: 20GB+ free
- **Internet**: Stable connection for downloads

---

## Phase 1: Core Setup (Required for Days 1-35)

### 1. Install Python 3.11+

#### Windows
1. Download from [python.org](https://www.python.org/downloads/)
2. Run installer
3. ✅ **Check "Add Python to PATH"**
4. Click "Install Now"

#### macOS
```bash
# Using Homebrew (recommended)
brew install python@3.11

# Or download from python.org
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3-pip python3-venv

# Fedora
sudo dnf install python3.11 python3-pip
```

#### Verify Installation
```bash
python --version  # Should show 3.11+
pip --version
```

### 2. Install Git

#### Windows
Download from [git-scm.com](https://git-scm.com/download/win)

#### macOS
```bash
brew install git
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt install git

# Fedora
sudo dnf install git
```

#### Verify
```bash
git --version
```

### 3. Clone Repository

```bash
git clone <repository-url>
cd 100-days-data-ai
```

### 4. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip
```

### 5. Install Core Python Packages

```bash
pip install -r requirements.txt
```

This installs:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- jupyter
- plotly
- scipy
- statsmodels
- duckdb

### 6. Verify Setup

```bash
python resources/test_setup.py
```

You should see ✅ for all core packages.

---

## Phase 2: Data Engineering Tools (Days 22-35)

### Install Apache Spark

#### Option 1: PySpark (Recommended for learning)
```bash
pip install pyspark
```

#### Option 2: Full Spark Installation
1. Install Java 11 or 17
   ```bash
   # macOS
   brew install openjdk@11
   
   # Ubuntu
   sudo apt install openjdk-11-jdk
   ```

2. Download Spark from [spark.apache.org](https://spark.apache.org/downloads.html)

3. Extract and set environment variables
   ```bash
   export SPARK_HOME=/path/to/spark
   export PATH=$PATH:$SPARK_HOME/bin
   ```

### Install Kafka (Optional - for Days 30-32)

#### Using Docker (Recommended)
```bash
# Install Docker first (see Docker section)

# Run Kafka with Docker Compose
docker-compose up -d kafka
```

#### Manual Installation
Download from [kafka.apache.org](https://kafka.apache.org/downloads)

### Install Airflow (Optional - for Days 33-34)

```bash
# Install Airflow
pip install apache-airflow

# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com
```

---

## Phase 3: Docker Setup (Days 44-49)

### Install Docker Desktop

#### Windows/macOS
1. Download from [docker.com](https://www.docker.com/products/docker-desktop)
2. Install and start Docker Desktop
3. Verify:
   ```bash
   docker --version
   docker-compose --version
   ```

#### Linux
```bash
# Ubuntu
sudo apt install docker.io docker-compose

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

### Test Docker
```bash
docker run hello-world
```

---

## Phase 4: PostgreSQL (Days 47-48)

### Option 1: Docker (Recommended)
```bash
docker run --name postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=bootcamp \
  -p 5432:5432 \
  -d postgres:15
```

### Option 2: Local Installation

#### macOS
```bash
brew install postgresql@15
brew services start postgresql@15
```

#### Ubuntu
```bash
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

#### Windows
Download from [postgresql.org](https://www.postgresql.org/download/windows/)

### Install Python PostgreSQL Driver
```bash
pip install psycopg2-binary sqlalchemy
```

---

## Phase 5: ML/AI Tools (Days 51-70)

### Install ML Packages
```bash
# Core ML
pip install scikit-learn

# Deep Learning
pip install torch torchvision

# Hugging Face
pip install transformers datasets

# MLflow
pip install mlflow

# Computer Vision
pip install opencv-python pillow
```

### GPU Support (Optional but Recommended)

#### NVIDIA GPU Setup
1. Install NVIDIA drivers
2. Install CUDA Toolkit
3. Install PyTorch with CUDA:
   ```bash
   # Check CUDA version first
   nvidia-smi
   
   # Install PyTorch with CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

---

## Phase 6: GenAI/LLM Tools (Days 71-92)

### Install LLM Packages
```bash
# LangChain
pip install langchain langchain-community

# Vector databases
pip install chromadb faiss-cpu

# Ollama Python client
pip install ollama

# LangGraph
pip install langgraph
```

### Install Ollama (Local LLMs)

#### macOS/Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Windows
Download from [ollama.com](https://ollama.com/download)

#### Pull Models
```bash
# Pull Llama 2 (7B)
ollama pull llama2

# Pull smaller model for testing
ollama pull phi
```

### Alternative: OpenAI API

If you prefer using OpenAI instead of local models:
```bash
pip install openai

# Set API key
export OPENAI_API_KEY='your-key-here'
```

---

## Phase 7: AWS Setup (Days 89-90)

### Install AWS CLI

#### macOS
```bash
brew install awscli
```

#### Linux
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

#### Windows
Download from [AWS CLI installer](https://aws.amazon.com/cli/)

### Configure AWS
```bash
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (e.g., us-east-1)
# - Default output format (json)
```

### Install Boto3
```bash
pip install boto3
```

---

## IDE Setup

### VS Code (Recommended)

1. Download from [code.visualstudio.com](https://code.visualstudio.com/)

2. Install Extensions:
   - Python (Microsoft)
   - Jupyter (Microsoft)
   - Docker (Microsoft)
   - GitLens
   - Pylance
   - Black Formatter

3. Configure Python Interpreter:
   - `Cmd/Ctrl + Shift + P`
   - "Python: Select Interpreter"
   - Choose your venv

### Jupyter Lab (Alternative)
```bash
pip install jupyterlab
jupyter lab
```

---

## Verification Checklist

Run through this checklist to ensure everything is set up:

### Core (Required for Day 1)
- [ ] Python 3.11+ installed
- [ ] Git installed
- [ ] Virtual environment created and activated
- [ ] Core packages installed (numpy, pandas, etc.)
- [ ] `python resources/test_setup.py` passes

### Data Engineering (Days 22-35)
- [ ] PySpark installed
- [ ] Docker installed and running
- [ ] Can run `docker run hello-world`

### Development (Days 44-49)
- [ ] PostgreSQL accessible (Docker or local)
- [ ] Can connect to database

### ML/AI (Days 51-70)
- [ ] scikit-learn installed
- [ ] PyTorch installed
- [ ] Transformers installed

### GenAI (Days 71-92)
- [ ] LangChain installed
- [ ] Ollama installed
- [ ] Can run `ollama pull llama2`
- [ ] ChromaDB installed

### Cloud (Days 89-90)
- [ ] AWS CLI installed (optional)
- [ ] AWS account created (optional)
- [ ] Boto3 installed

---

## Troubleshooting

### Python Not Found
```bash
# macOS/Linux: Add to ~/.bashrc or ~/.zshrc
export PATH="/usr/local/bin/python3:$PATH"

# Windows: Add Python to PATH in System Environment Variables
```

### pip Install Fails
```bash
# Upgrade pip
pip install --upgrade pip

# Use --user flag
pip install --user package-name

# Clear cache
pip cache purge
```

### Docker Permission Denied (Linux)
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### Spark Java Error
```bash
# Install Java
brew install openjdk@11  # macOS
sudo apt install openjdk-11-jdk  # Ubuntu

# Set JAVA_HOME
export JAVA_HOME=$(/usr/libexec/java_home)  # macOS
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64  # Ubuntu
```

### Ollama Connection Error
```bash
# Start Ollama service
ollama serve

# In another terminal, pull model
ollama pull llama2
```

### Out of Memory (Spark/ML)
```bash
# Increase Spark memory
export SPARK_DRIVER_MEMORY=4g

# Use smaller batch sizes in ML training
# Use smaller models (e.g., phi instead of llama2)
```

---

## Resource Requirements by Phase

| Phase | RAM | Disk | GPU |
|-------|-----|------|-----|
| Phase 1 (Days 1-35) | 8GB | 5GB | No |
| Phase 2 (Days 36-50) | 8GB | 10GB | No |
| Phase 3 (Days 51-70) | 16GB | 15GB | Recommended |
| Phase 4 (Days 71-92) | 16GB | 20GB | Recommended |
| Phase 5 (Days 93-100) | 16GB | 20GB | Recommended |

---

## Quick Start Commands

```bash
# 1. Clone and setup
git clone <repo-url>
cd 100-days-data-ai
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install packages
pip install -r requirements.txt

# 3. Verify setup
python resources/test_setup.py

# 4. Start Day 1
cd days/day-001-csv-vs-json
jupyter notebook exercise.ipynb
```

---

## Getting Help

- **Documentation**: Check `resources/cheatsheet.md` for quick reference
- **Issues**: Open an issue on GitHub
- **Community**: Join Discord/Slack (link in README)

---

## Next Steps

Once setup is complete:
1. Read the [README.md](../README.md)
2. Review [CURRICULUM.md](../CURRICULUM.md)
3. Start with Day 1: CSV vs JSON

Good luck! 🚀
