# Folder Structure

The project has been reorganized into four main directories for better organization:

## Directory Structure

```
parser/
├── README.md                          # Main project README
├── README_UNIFIED_PARSER.md          # Unified parser documentation
├── requirements.txt                   # Python dependencies
├── scripts/                          # Core parser scripts
│   ├── __init__.py
│   ├── unified_parser.py             # Main unified parser
│   ├── document_parser.py            # Document parser
│   ├── graphics_handler.py           # Graphics/image handler
│   ├── visual_analyzer.py            # Visual content analyzer
│   ├── data_extractor.py             # Data extraction utilities
│   ├── batch_processor.py            # Batch processing
│   ├── one_click_processor.py        # One-click processing
│   ├── process_any_file.py           # Generic file processor
│   ├── config.py                     # Configuration
│   ├── setup.py                      # Setup script
│   ├── test_parser.py                # Parser tests
│   └── test_visual_analysis.py       # Visual analysis tests
│
├── demos/                            # Demo and example scripts
│   ├── demo_unified_parser.py        # Unified parser demo
│   ├── visual_demo.py                # Visual analysis demo
│   ├── example_usage.py              # Usage examples
│   └── quick_start.py                # Quick start guide
│
├── input/                            # Input files
│   ├── sample_document.png
│   ├── file-sample_150kB.pdf
│   └── sample_document.txt
│
└── output/                           # Output files and results
    ├── *.json                        # JSON results
    ├── *.csv                         # CSV exports
    ├── *.md                          # Markdown reports
    ├── assignment_output_batch/      # Batch processing outputs
    ├── demo_output_*/                # Demo outputs
    ├── visual_test_output/           # Visual test outputs
    └── sample_150kb_output/          # Sample outputs
```

## Directory Purposes

### 📁 `scripts/`
Core parser modules and utilities. Contains all the main parsing logic.

**Key Files:**
- `unified_parser.py` - Main parser that extracts images, text, and positions
- `document_parser.py` - Document parsing engine
- `graphics_handler.py` - Visual content detection
- `visual_analyzer.py` - Visual content analysis

### 📁 `demos/`
Demo scripts and examples showing how to use the parsers.

**Key Files:**
- `demo_unified_parser.py` - Demo of unified parser
- `visual_demo.py` - Visual analysis demo
- `example_usage.py` - Usage examples
- `quick_start.py` - Quick start guide

### 📁 `input/`
Input files for testing and processing.

**Contents:**
- Sample images (PNG, JPG)
- Sample PDFs
- Sample text files

### 📁 `output/`
All generated output files.

**Contents:**
- Parsed JSON results
- CSV exports
- Generated reports
- Batch processing results

## How to Use

### Running Scripts

From the root directory:

```bash
# Run a script
python3 scripts/unified_parser.py input/sample_document.png -o output/result.json

# Run a demo
python3 demos/demo_unified_parser.py

# Run tests
python3 scripts/test_parser.py
```

### Python API

```python
# Import from scripts directory
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')

from unified_parser import UnifiedParser

# Use the parser
parser = UnifiedParser()
result = parser.parse_with_positions('input/sample.png')
```

### Import Resolution

All scripts in the `scripts/` directory can import from each other directly. Demos need to add the scripts directory to the Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
```

## Benefits of This Structure

✅ **Organization** - Clear separation of concerns  
✅ **Maintainability** - Easy to find and update code  
✅ **Scalability** - Easy to add new scripts and features  
✅ **Clarity** - Clear purpose for each directory  
✅ **Clean** - Input and output files are separated  

## Migration Notes

If you have existing code that imports from the old structure:

**Old:**
```python
from unified_parser import UnifiedParser
```

**New:**
```python
import sys
sys.path.insert(0, 'scripts')
from unified_parser import UnifiedParser
```

Or run scripts from the root directory with:
```bash
python3 -m scripts.unified_parser input/sample.png
```
