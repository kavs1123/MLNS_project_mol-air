#!/usr/bin/env python

"""
This script fixes the type annotation in the molbloom package to be compatible with Python 3.7.
It replaces the Python 3.10+ union type syntax (str | None) with the typing.Union syntax.
"""

import os
from pathlib import Path
import re

def fix_molbloom_typing():
    try:
        # Find the molbloom __init__.py file in the conda environment
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if not conda_prefix:
            print("Could not determine conda environment path. Make sure you're in an active conda environment.")
            return False
            
        molbloom_path = Path(conda_prefix) / "lib" / "python3.7" / "site-packages" / "molbloom" / "__init__.py"
        
        if not molbloom_path.exists():
            print(f"Could not find molbloom package at {molbloom_path}")
            return False
            
        print(f"Found molbloom at {molbloom_path}")
        
        # Read the content of the file
        with open(molbloom_path, 'r') as f:
            content = f.read()
        
        # Replace the Python 3.10+ union type syntax with the typing.Union syntax
        if 'from typing import Union' not in content:
            content = re.sub(r'from typing import (.*)', r'from typing import \1, Union', content, count=1)
            if 'from typing import ' not in content:
                content = 'from typing import Union\n' + content
                
        # Replace str | None with Union[str, None]
        updated_content = re.sub(r'def canon\(smiles: str\) -> str \| None:', r'def canon(smiles: str) -> Union[str, None]:', content)
        
        if updated_content == content:
            print("No changes were needed for the molbloom package.")
            return True
            
        # Write the updated content back to the file
        with open(molbloom_path, 'w') as f:
            f.write(updated_content)
        
        print("Successfully fixed the molbloom package typing issue.")
        return True
        
    except Exception as e:
        print(f"An error occurred while fixing the molbloom package: {e}")
        return False

if __name__ == "__main__":
    success = fix_molbloom_typing()
    if success:
        print("You can now run your code without the typing error.")
    else:
        print("Failed to fix the molbloom package. You may need to modify it manually.")
        print("Open the molbloom/__init__.py file and change 'str | None' to 'Union[str, None]'")