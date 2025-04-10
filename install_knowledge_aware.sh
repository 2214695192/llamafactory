#!/bin/bash

# Activate llama environment
source /home/zqq/anaconda3/bin/activate llama

# Create backup of current environment
pip freeze > requirements_backup.txt

# Install dependencies in the correct order with compatible versions
pip install tokenizers==0.19.0
pip install transformers==4.41.2
pip install peft==0.11.1
pip install accelerate==0.34.0
pip install numpy==1.24.3
pip install matplotlib==3.7.1
pip install tensorboard==2.15.0

# Verify installation
echo "Verifying installation..."
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import transformers; print('Transformers version:', transformers.__version__)"
python -c "import peft; print('PEFT version:', peft.__version__)"
python -c "import tokenizers; print('Tokenizers version:', tokenizers.__version__)"

echo "Installation complete! If you encounter any issues, you can restore the previous environment using: pip install -r requirements_backup.txt" 