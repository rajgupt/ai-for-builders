#!/bin/bash

# Validation script for AI Skills Hub landing page setup
# This script checks if all required files and configurations are in place

echo "🔍 Validating AI Skills Hub Landing Page Setup..."
echo "================================================"
echo ""

# Base directory
BASE_DIR="/Users/rajatgupta/repos/ai-for-builders/aitutor"
cd "$BASE_DIR" || exit 1

# Counter for checks
PASSED=0
FAILED=0

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo "✅ $1"
        ((PASSED++))
    else
        echo "❌ MISSING: $1"
        ((FAILED++))
    fi
}

# Function to check directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo "✅ $1/"
        ((PASSED++))
    else
        echo "❌ MISSING: $1/"
        ((FAILED++))
    fi
}

echo "📁 Checking Directory Structure..."
echo "-----------------------------------"
check_dir "docs"
check_dir "docs/stylesheets"
check_dir "docs/javascripts"
check_dir "docs/courses"
check_dir "docs/courses/foundation"
check_dir "docs/courses/core"
check_dir "docs/courses/advanced"
check_dir "docs/projects"
check_dir "docs/resources"
check_dir "notebooks"
echo ""

echo "📄 Checking Configuration Files..."
echo "-----------------------------------"
check_file "mkdocs.yml"
check_file "requirements.txt"
check_file "README.md"
echo ""

echo "📝 Checking Content Files..."
echo "-----------------------------------"
check_file "docs/index.md"
check_file "docs/stylesheets/custom.css"
check_file "docs/courses/foundation/index.md"
check_file "docs/courses/core/index.md"
check_file "docs/courses/advanced/index.md"
check_file "docs/projects/index.md"
check_file "docs/resources/index.md"
echo ""

echo "📓 Checking Notebooks..."
echo "-----------------------------------"
check_file "notebooks/quickstart.ipynb"
echo ""

echo "🔧 Checking GitHub Actions..."
echo "-----------------------------------"
check_file "../.github/workflows/deploy.yml"
echo ""

echo "📊 Validation Summary"
echo "================================================"
echo "✅ Passed: $PASSED"
echo "❌ Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 All checks passed! The landing page setup is complete."
    echo ""
    echo "Next Steps:"
    echo "1. Install dependencies: pip install -r requirements.txt"
    echo "2. Test locally: mkdocs serve"
    echo "3. Push to GitHub to trigger deployment"
    echo ""
    exit 0
else
    echo "⚠️  Some checks failed. Please review the missing files above."
    exit 1
fi
