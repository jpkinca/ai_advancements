# AI Trading Advancements - Quick Reference Guide

## 🚀 Essential Commands

### Repository Setup
```bash
# Create new repository on GitHub first, then:
cd "c:\Users\nzcon\VSPython\ai_advancements"
git init
git add .
git commit -m "Initial commit: AI Trading Advancements setup"
git remote add origin https://github.com/YOUR_USERNAME/ai-trading-advancements.git
git branch -M main
git push -u origin main
```

### Environment Setup
```bash
# Automated setup
python setup_repository.py

# Manual setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
cp .env.example .env
```

### Development Workflow
```bash
# Daily development routine
venv\Scripts\activate
black .
flake8 .
pytest
git add .
git commit -m "feat: your changes"
git push
```

## 🔧 Key Configuration Files

### .env Configuration
```bash
DATABASE_URL=postgresql://user:pass@host:port/db
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
ALPHA_VANTAGE_API_KEY=your_key
```

### Testing
```bash
# Run all tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test
pytest tests/test_integration.py::TestFAISSEngine
```

## 📊 Component Quick Start

### FAISS Engine
```bash
cd faiss/
python demo.py
python test_faiss_database.py
```

### Database Validation
```bash
python test_postgresql_connection.py
python verify_ai_database.py
```

### Quality Checks
```bash
black .          # Format code
flake8 .         # Lint code
mypy .           # Type check
pytest           # Run tests
```

## 🔍 Troubleshooting

### Common Issues
- **Import errors**: Check virtual environment activation
- **Database connection**: Verify DATABASE_URL format
- **FAISS issues**: Install faiss-cpu or faiss-gpu
- **Permission errors**: Run as administrator (Windows)

### Health Checks
```bash
python -c "import faiss; print('FAISS OK')"
python -c "import psycopg2; print('PostgreSQL OK')"
python test_postgresql_connection.py
```

## 📚 Documentation Structure

```
ai_advancements/
├── README.md                    # Main overview
├── SETUP_INSTRUCTIONS.md       # Complete setup guide
├── QUICK_REFERENCE.md          # This file
├── REPOSITORY_SETUP_COMPLETE.md # Setup summary
├── faiss/FAISS_IMPLEMENTATION_GUIDE.md
└── [component]/README.md       # Component-specific docs
```

## ⚡ Production Checklist

- [ ] Repository created on GitHub
- [ ] Environment variables configured
- [ ] Database connection tested
- [ ] All tests passing
- [ ] CI/CD pipeline enabled
- [ ] Documentation complete
- [ ] Security scanning enabled
- [ ] Trading disclaimers reviewed

## 🎯 Next Steps After Setup

1. **Configure GitHub repository settings**
2. **Set up branch protection rules**
3. **Add team members and collaborators**
4. **Configure monitoring and alerts**
5. **Begin development on specific components**

---

**Status**: Ready for production development and deployment
