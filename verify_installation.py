#!/usr/bin/env python3
"""
Quick verification script to check if TensorFlow and dependencies are installed correctly.
Run with: python verify_installation.py
"""

import sys

def check_imports():
    packages = {
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn'
    }
    
    print("=" * 50)
    print("VERIFYING INSTALLATION")
    print("=" * 50)
    print(f"Python Version: {sys.version}\n")
    
    all_good = True
    for package, name in packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {name:<20} {version}")
        except ImportError:
            print(f"❌ {name:<20} NOT INSTALLED")
            all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("✅ All packages installed successfully!")
        print("=" * 50)
        print("\nYou can now run: python TF.py")
        return 0
    else:
        print("❌ Some packages are missing. Run:")
        print("pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(check_imports())
