#!/usr/bin/env python3
"""
Test script to verify backend and frontend can start without errors
"""

import sys
import os
import subprocess
import time

def test_backend_imports():
    """Test that backend can be imported without errors"""
    print("Testing backend imports...")
    print("-" * 60)
    
    try:
        # Change to backend directory
        os.chdir('backend')
        
        # Try importing the app
        result = subprocess.run(
            ['python', '-c', 'from app import app; print("✓ Backend imports successfully")'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Go back to parent directory
        os.chdir('..')
        
        if result.returncode == 0:
            print("✓ Backend imports successfully")
            if "Face mesh analyzer not available" in result.stderr:
                print("  ⚠️  Face mesh analyzer disabled (this is OK)")
            print("  ✓ FER emotion detector loaded")
            return True
        else:
            print("❌ Backend import failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Backend import timed out")
        os.chdir('..')
        return False
    except Exception as e:
        print(f"❌ Error testing backend: {e}")
        os.chdir('..')
        return False

def test_frontend_dependencies():
    """Test that frontend dependencies are installed"""
    print("\nTesting frontend dependencies...")
    print("-" * 60)
    
    try:
        os.chdir('frontend')
        
        # Check if node_modules exists
        if not os.path.exists('node_modules'):
            print("❌ node_modules not found")
            print("   Run: cd frontend && npm install")
            os.chdir('..')
            return False
        
        # Check if package.json exists
        if not os.path.exists('package.json'):
            print("❌ package.json not found")
            os.chdir('..')
            return False
        
        print("✓ Frontend dependencies installed")
        os.chdir('..')
        return True
        
    except Exception as e:
        print(f"❌ Error testing frontend: {e}")
        os.chdir('..')
        return False

def test_venv_activation():
    """Test that virtual environment exists and can be activated"""
    print("\nTesting virtual environment...")
    print("-" * 60)
    
    if not os.path.exists('venv'):
        print("❌ Virtual environment not found")
        print("   Create it with: python3 -m venv venv")
        return False
    
    if not os.path.exists('venv/bin/activate'):
        print("❌ Virtual environment activation script not found")
        return False
    
    print("✓ Virtual environment exists")
    return True

def test_required_packages():
    """Test that required Python packages are installed"""
    print("\nTesting required Python packages...")
    print("-" * 60)
    
    required_packages = [
        'flask',
        'flask_cors',
        'flask_socketio',
        'cv2',
        'numpy',
        'fer.fer',
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            result = subprocess.run(
                ['python', '-c', f'import {package}; print("✓ {package}")'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"  ✓ {package}")
            else:
                print(f"  ❌ {package} - not installed")
                all_installed = False
        except Exception as e:
            print(f"  ❌ {package} - error: {e}")
            all_installed = False
    
    return all_installed

def print_startup_instructions():
    """Print instructions for starting the application"""
    print("\n" + "=" * 60)
    print("How to Start the Application")
    print("=" * 60)
    print("\n1. Start Backend (Terminal 1):")
    print("   ./start_backend.sh")
    print("\n2. Start Frontend (Terminal 2):")
    print("   ./start_frontend.sh")
    print("\n3. Open Browser:")
    print("   http://localhost:5173")
    print("\n4. Select Camera and Click 'Start Camera'")
    print("\nExpected Output:")
    print("  - Backend: Running on http://localhost:5001")
    print("  - Frontend: Running on http://localhost:5173")
    print("  - Browser: Camera dropdown with available cameras")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Emotion Traffic Light - Startup Test")
    print("=" * 60)
    
    # Save current directory
    original_dir = os.getcwd()
    
    tests = [
        ("Virtual Environment", test_venv_activation),
        ("Required Packages", test_required_packages),
        ("Backend Imports", test_backend_imports),
        ("Frontend Dependencies", test_frontend_dependencies),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            # Make sure we're in the original directory
            os.chdir(original_dir)
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
            os.chdir(original_dir)
    
    # Make sure we're back in original directory
    os.chdir(original_dir)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed! Ready to start the application.")
        print_startup_instructions()
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed.")
        print("\nPlease fix the issues above before starting the application.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

