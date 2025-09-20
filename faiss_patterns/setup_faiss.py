#!/usr/bin/env python3
"""
Setup script for installing FAISS and testing the installation.
"""

import subprocess
import sys
import importlib

def install_package(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def test_faiss_installation():
    """Test if FAISS is properly installed."""
    try:
        import faiss
        print(f"✅ FAISS installed successfully! Version: {faiss.__version__ if hasattr(faiss, '__version__') else 'Unknown'}")
        
        # Test basic functionality
        d = 64
        index = faiss.IndexFlatL2(d)
        print(f"✅ Created FAISS index with dimension {d}")
        
        # Test adding and searching
        import numpy as np
        data = np.random.random((100, d)).astype('float32')
        index.add(data)
        print(f"✅ Added {len(data)} vectors to index")
        
        # Search
        k = 5
        distances, indices = index.search(data[:1], k)
        print(f"✅ Search successful! Found {len(indices[0])} results")
        
        return True
    except ImportError:
        print("❌ FAISS not installed")
        return False
    except Exception as e:
        print(f"❌ FAISS installation issue: {e}")
        return False

def main():
    print("FAISS Setup and Test Script")
    print("=" * 40)
    
    # Check current installation
    if test_faiss_installation():
        print("FAISS is already installed and working!")
        return
    
    print("Installing FAISS...")
    try:
        # Try CPU version first
        install_package("faiss-cpu")
        print("FAISS-CPU installed successfully!")
        
        # Test installation
        if test_faiss_installation():
            print("🎉 FAISS setup complete!")
        else:
            print("⚠️ FAISS installed but not working properly")
            
    except Exception as e:
        print(f"❌ Failed to install FAISS: {e}")
        print("You can try installing manually with:")
        print("  pip install faiss-cpu")
        print("  or for GPU: pip install faiss-gpu")

if __name__ == "__main__":
    main()
