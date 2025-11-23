#!/bin/bash
# Validation script to verify the assignment setup

echo "====================================="
echo "Generative Models Assignment Validator"
echo "====================================="

# Check if required files exist
echo ""
echo "Checking files..."
required_files=(
    "denoising_autoencoder.py"
    "dcgan.py"
    "generate_synthetic_data.py"
    "requirements.txt"
    "report.tex"
    "README.md"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file exists"
    else
        echo "✗ $file missing"
        exit 1
    fi
done

# Check outputs directory
echo ""
echo "Checking outputs..."
if [ -d "outputs" ]; then
    num_files=$(ls -1 outputs/ | wc -l)
    echo "✓ outputs/ directory exists with $num_files files"
else
    echo "✗ outputs/ directory missing"
    exit 1
fi

# Check for key output files
required_outputs=(
    "outputs/dae_loss_curves.png"
    "outputs/dae_reconstruction_comparison.png"
    "outputs/dcgan_loss_curves.png"
    "outputs/dcgan_final_generated.png"
)

for file in "${required_outputs[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file exists"
    else
        echo "✗ $file missing"
        exit 1
    fi
done

# Check Python syntax
echo ""
echo "Checking Python syntax..."
for pyfile in *.py; do
    if python3 -m py_compile "$pyfile" 2>/dev/null; then
        echo "✓ $pyfile syntax OK"
    else
        echo "✗ $pyfile has syntax errors"
        exit 1
    fi
done

# Check LaTeX file
echo ""
echo "Checking LaTeX file..."
if grep -q "\\documentclass" report.tex && grep -q "\\end{document}" report.tex; then
    echo "✓ report.tex appears valid"
else
    echo "✗ report.tex may be malformed"
    exit 1
fi

echo ""
echo "====================================="
echo "✓ All validation checks passed!"
echo "====================================="
echo ""
echo "Summary:"
echo "- All required Python scripts present"
echo "- All output visualizations generated"
echo "- LaTeX report created"
echo "- Ready for submission"
echo ""
echo "Next steps:"
echo "1. Upload report.tex to Overleaf and compile"
echo "2. Review generated images in outputs/"
echo "3. Run scripts with more epochs for better results (optional)"
