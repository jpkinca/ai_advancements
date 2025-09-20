# Setup Guide for Hugging Face Authentication

## Step 1: Get Your Hugging Face Token

1. **Create a Hugging Face Account** (if you don't have one):
   - Go to https://huggingface.co/join
   - Sign up with your email

2. **Generate an Access Token**:
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Choose "Read" permissions (sufficient for model access)
   - Copy the token (starts with `hf_...`)

3. **Request Access to Llama Models**:
   - Go to https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
   - Click "Request access to this model"
   - Accept the license agreement
   - Wait for approval (usually instant)

## Step 2: Configure Your Token

### Option A: Environment Variable (Recommended)
```bash
# Windows PowerShell
$env:HUGGINGFACE_TOKEN="hf_your_token_here"

# Windows Command Prompt
set HUGGINGFACE_TOKEN=hf_your_token_here

# Linux/Mac
export HUGGINGFACE_TOKEN="hf_your_token_here"
```

### Option B: .env File
1. Copy `.env.example` to `.env`
2. Edit `.env` and add your token:
```
HUGGINGFACE_TOKEN=hf_your_token_here
```

### Option C: Direct Configuration
Edit `config.py`:
```python
LLM_CONFIG = {
    'llm_model': 'llama-3-8b',
    'huggingface_token': 'hf_your_token_here',
    # ... other settings
}
```

## Step 3: Test Authentication

Run the test to verify everything works:
```bash
cd c:\Users\nzcon\VSPython\ai_advancements\chain_of_alpha
python test_auth.py
```

## Alternative: Use Mock Mode

If you don't want to set up authentication right now, you can use mock mode:

1. Edit `config.py`:
```python
LLM_CONFIG = {
    'llm_model': 'mock',  # Uses mock responses for testing
    # ... other settings
}
```

2. Run the MVP:
```bash
python chain_of_alpha_mvp.py
```

Mock mode will generate realistic factor examples without requiring API access.

## Troubleshooting

1. **"Access to model is restricted"**: You need to request access to the Llama model
2. **"Token not found"**: Set the HUGGINGFACE_TOKEN environment variable
3. **"Invalid token"**: Check that your token starts with `hf_` and is copied correctly
4. **Memory issues**: Llama models require significant RAM. Consider using a smaller model or mock mode.

## Security Note

Never commit your API tokens to version control. Always use environment variables or `.env` files (and add `.env` to your `.gitignore`).