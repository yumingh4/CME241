#!/bin/bash
# Phase 3 Setup Script
# Run this to set up your environment

echo "======================================================================"
echo "  Phase 3: Buy vs. Rent RL - Setup Script"
echo "======================================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

echo "✓ pip found: $(pip3 --version)"
echo ""

# Create virtual environment (optional but recommended)
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo "✓ Virtual environment created and activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

echo ""

# Install requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "======================================================================"
echo "✓ Setup complete!"
echo "======================================================================"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate when done, run:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "  1. Copy calibrated_params.py from phase2: cp ../phase2/calibrated_params.py ."
echo "  2. Run the demo notebook: jupyter notebook phase3_demo.ipynb"
echo "  3. Or run scripts directly:"
echo "     - python buy_rent_environment.py"
echo "     - python modified_dqn.py"
echo "     - python evaluate_policies.py"
echo ""
echo "======================================================================"
