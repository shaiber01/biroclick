# Preparing Papers for Reproduction

This guide explains how to prepare a scientific paper for use with the ReproLab system.

## Overview

The system supports two methods for loading papers:

### Method A: Markdown Loading (Recommended)
```
Paper (PDF) → marker/nougat → Markdown with figures → load_paper_from_markdown() → PaperInput
```
The loader automatically extracts and downloads figures from the markdown.

### Method B: Manual JSON Preparation
```
Paper (PDF) → Manual Extraction → JSON file → load_paper_input() → PaperInput
```
Requires manually specifying figure paths in a JSON file.

---

## Quick Start: Markdown Loading

The fastest way to prepare a paper is using the markdown loader:

> **Future Plan (v2):** Direct PDF loading via `load_paper_from_pdf()` will eliminate the manual conversion step. See `docs/guidelines.md` Section 14 for roadmap.

```python
from src.paper_loader import load_paper_from_markdown

# Convert PDF to markdown using marker
# $ marker_single paper.pdf --output_dir ./extracted/

# Load the markdown and automatically download figures
paper_input = load_paper_from_markdown(
    markdown_path="./extracted/paper.md",
    output_dir="./extracted/figures",
    paper_id="smith2023_plasmon",
    paper_domain="plasmonics"
)
```

The loader will:
1. Parse the markdown file
2. Extract all figure references (`![alt](url)` and `<img src="...">`)
3. Download figures to the output directory
4. Generate figure IDs from alt text or filenames
5. Return a validated `PaperInput` structure

### Supported Figure Formats

| Format | Extension | Vision Model Support |
|--------|-----------|---------------------|
| PNG | `.png` | ✅ Preferred |
| JPEG | `.jpg`, `.jpeg` | ✅ Preferred |
| GIF | `.gif` | ✅ Preferred |
| WebP | `.webp` | ✅ Preferred |
| BMP | `.bmp` | ⚠️ Supported |
| TIFF | `.tiff`, `.tif` | ⚠️ May need conversion |
| SVG | `.svg` | ⚠️ May need conversion |
| EPS | `.eps` | ⚠️ May need conversion |
| PDF | `.pdf` | ⚠️ May need conversion |

### Handling Relative Paths

The loader supports various path formats:

```python
# Figures with relative paths (resolved against markdown file location)
# Markdown: ![Figure 1](images/fig1.png)
paper_input = load_paper_from_markdown(
    markdown_path="papers/smith2023/paper.md",  # figures resolved from papers/smith2023/
    output_dir="papers/smith2023/figures"
)

# Figures with remote URLs needing a base URL
# Markdown: ![Figure 1](fig1.png)  (needs https://example.com/paper/fig1.png)
paper_input = load_paper_from_markdown(
    markdown_path="downloaded_paper.md",
    output_dir="./figures",
    base_url="https://example.com/paper/"
)

# Figures with absolute URLs (downloaded directly)
# Markdown: ![Figure 1](https://arxiv.org/html/1234/fig1.png)
paper_input = load_paper_from_markdown(
    markdown_path="paper.md",
    output_dir="./figures"
)
```

### Loading Supplementary Materials

If your paper has separate supplementary materials (common for scientific papers), convert the SI PDF to markdown separately and provide both:

```python
# Convert both PDFs
# $ marker_single paper.pdf --output_dir ./extracted/
# $ marker_single supplementary.pdf --output_dir ./extracted/

paper_input = load_paper_from_markdown(
    markdown_path="./extracted/paper.md",
    output_dir="./extracted/figures",
    supplementary_markdown_path="./extracted/supplementary.md",  # Optional SI
    paper_id="smith2023_plasmon"
)
```

The loader will:
- Load supplementary text into `paper_input['supplementary']['supplementary_text']`
- Extract supplementary figures into `paper_input['supplementary']['supplementary_figures']`
- Prefix supplementary figure IDs with "S" (e.g., "SFig1", "SFig2")

### Handling Long Papers

The loader automatically checks paper length and displays warnings:

| Length | Status | Recommendation |
|--------|--------|----------------|
| < 50K chars | ✅ Normal | Most papers fit here |
| 50-150K chars | ⚠️ Long | Consider trimming references |
| > 150K chars | ⚠️ Very long | May hit context limits; trim non-essential sections |

**Recommended trimming for long papers:**
1. **References section** (20-30% of text, not needed for reproduction)
2. **Acknowledgments** (not relevant)
3. **Author contributions** (not relevant)
4. **Detailed literature review** (keep only methodology-relevant citations)

**Example output for a long paper:**
```
Paper loaded from markdown:
  Title: Extended Review of Plasmonic Nanoantennas...
  ID: review_plasmonics_2023
  Main text: 180,432 chars (~45,108 tokens)
  Supplementary text: 42,156 chars (~10,539 tokens)
  Total: 222,588 chars (~55,647 tokens)
  Main figures: 24
  Supplementary figures: 8

📏 Length warnings:
  ⚠️  Main paper is long: 180,432 chars (~45,108 tokens). Consider trimming references section to reduce costs.
```

### Saving for Reuse

After loading, save the `PaperInput` to avoid re-downloading:

```python
from src.paper_loader import save_paper_input_json

save_paper_input_json(paper_input, "papers/smith2023/paper_input.json")

# Later, load directly from JSON (faster, no downloads)
from src.paper_loader import load_paper_input
paper_input = load_paper_input("papers/smith2023/paper_input.json")
```

---

## Step 1: Extract Paper Text

### Option A: Using `marker` (Recommended)

[Marker](https://github.com/VikParuchuri/marker) provides high-quality PDF-to-markdown conversion:

```bash
# Install marker
pip install marker-pdf

# Convert PDF to markdown
marker_single paper.pdf --output_dir ./extracted/
```

**Pros**: Preserves equations, tables, and structure  
**Cons**: May require manual cleanup of complex layouts

### Option B: Using `nougat` (For Complex Papers)

[Nougat](https://github.com/facebookresearch/nougat) is designed for scientific documents:

```bash
# Install nougat
pip install nougat-ocr

# Process PDF
nougat paper.pdf -o ./extracted/
```

**Pros**: Excellent equation handling, designed for arXiv papers  
**Cons**: Slower, GPU recommended

### Option C: Manual Copy-Paste

For simpler papers, copy-paste from PDF viewer:

1. Open PDF in a good viewer (Adobe, Chrome)
2. Select all text (Ctrl+A)
3. Paste into a text file
4. Clean up formatting artifacts

**Pros**: Quick, no dependencies  
**Cons**: May lose equations, tables

### Quality Checklist for Extracted Text

- [ ] All sections present (Abstract, Methods, Results, etc.)
- [ ] Key numerical values readable
- [ ] Equations converted reasonably (even as text descriptions)
- [ ] Figure captions included
- [ ] References section intact (optional, but useful for context)

---

## Step 2: Extract Figure Images

### Option A: Using PyMuPDF (Recommended)

```python
import fitz  # PyMuPDF

def extract_figures(pdf_path: str, output_dir: str):
    """Extract all images from a PDF."""
    doc = fitz.open(pdf_path)
    
    for page_num, page in enumerate(doc):
        images = page.get_images()
        
        for img_idx, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            
            # Convert CMYK to RGB if needed
            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            
            output_path = f"{output_dir}/page{page_num}_fig{img_idx}.png"
            pix.save(output_path)
            print(f"Saved: {output_path}")

# Usage
extract_figures("paper.pdf", "./figures/")
```

### Option B: Manual Screenshots

For papers with embedded plots that don't extract well:

1. Open PDF and zoom to 150-200%
2. Use screenshot tool (Snipping Tool, Cmd+Shift+4)
3. Capture each figure individually
4. Save as PNG (preferred) or JPEG

### Figure Naming Convention

Use consistent naming that matches the paper:

```
figures/
├── fig1_schematic.png
├── fig2a_absorption_spectrum.png
├── fig2b_transmission.png
├── fig3_field_enhancement.png
└── fig4_parametric_sweep.png
```

### Image Quality Guidelines

| Aspect | Requirement | Reason |
|--------|-------------|--------|
| Resolution | ≥512px on shortest side | Vision models work best with sufficient detail |
| Resolution | ≤4096px on longest side | Larger wastes tokens, may hit limits |
| File size | <5MB | Avoid excessive token costs |
| Format | PNG or JPEG | Widely supported |
| Content | Single figure per image | Clearer for comparison |

---

## Step 3: Digitize Key Figures (Optional but Recommended)

For quantitative comparison, digitize the data from key figures.

### Using WebPlotDigitizer

[WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/) is a free online tool:

1. **Upload figure image**
2. **Define axes**:
   - Click on two points on X-axis, enter their values
   - Click on two points on Y-axis, enter their values
3. **Extract data points**:
   - Use automatic extraction for clean plots
   - Manual point selection for noisy data
4. **Export as CSV**

### CSV Format

```csv
# fig2a_absorption.csv
# Wavelength (nm), Absorption (a.u.)
400, 0.12
410, 0.15
420, 0.23
...
```

### Which Figures to Digitize

Prioritize figures that show:
- **Spectra** (transmission, absorption, reflection)
- **Resonance positions** (where peak wavelengths matter)
- **Quantitative comparisons** (mode profiles, field enhancements)

Skip digitization for:
- Schematic diagrams
- 2D field maps (qualitative comparison is sufficient)
- Setup photographs

---

## Step 4: Create PaperInput Structure

### Option A: Automatic from Markdown (Recommended)

If your markdown from Step 1 contains embedded figure links, use the automatic loader:

```python
from src.paper_loader import load_paper_from_markdown, save_paper_input_json

# Load and download figures automatically
paper_input = load_paper_from_markdown(
    markdown_path="./extracted/paper.md",
    output_dir="./figures",
    paper_id="smith2023_plasmon",
    paper_domain="plasmonics"
)

# Save for later use
save_paper_input_json(paper_input, "./paper_input.json")
```

This is the fastest approach when figures are embedded in the markdown output.

### Option B: Manual Python Construction

For more control, or when figures need manual handling:

```python
from src.paper_loader import PaperInput, FigureInput, load_paper_text

# Load extracted text
paper_text = load_paper_text("./extracted/paper.md")

# Define figures manually
figures = [
    {
        "id": "fig2a",
        "description": "Absorption spectrum showing plasmon resonance at 520nm",
        "image_path": "./figures/fig2a_absorption.png",
        "digitized_data_path": "./figures/fig2a_absorption.csv"  # Optional
    },
    {
        "id": "fig3",
        "description": "Electric field enhancement |E|² map at resonance",
        "image_path": "./figures/fig3_field_map.png"
    }
]

# Create input
paper_input: PaperInput = {
    "paper_id": "smith2023_plasmon",
    "paper_title": "Tunable Plasmon Resonances in Gold Nanorods",
    "paper_domain": "plasmonics",
    "paper_text": paper_text,
    "figures": figures,
    "supplementary": []  # Add if available
}
```

### Option C: JSON File

Create a JSON file for the paper input:

```json
{
  "paper_id": "smith2023_plasmon",
  "paper_title": "Tunable Plasmon Resonances in Gold Nanorods",
  "paper_domain": "plasmonics",
  "paper_text": "Abstract: We demonstrate tunable...",
  "figures": [
    {
      "id": "fig2a",
      "description": "Absorption spectrum showing plasmon resonance at 520nm",
      "image_path": "./figures/fig2a_absorption.png",
      "digitized_data_path": "./figures/fig2a_absorption.csv"
    }
  ],
  "supplementary": []
}
```

Then load it:

```python
from src.paper_loader import load_paper_input
paper_input = load_paper_input("./paper_input.json")
```

---

## Step 5: Validate Your Input

Use the built-in validation functions:

```python
from src.paper_loader import (
    validate_paper_input,
    validate_figure_image,
    estimate_token_cost
)

# Check overall structure
errors = validate_paper_input(paper_input)
if errors:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")

# Check figure image quality
for fig in paper_input["figures"]:
    warnings = validate_figure_image(fig["image_path"])
    if warnings:
        print(f"Warnings for {fig['id']}:")
        for warning in warnings:
            print(f"  - {warning}")

# Estimate costs
cost_estimate = estimate_token_cost(paper_input)
print(f"Estimated cost: ${cost_estimate['estimated_cost_usd']:.2f}")
print(f"({cost_estimate['warning']})")
```

---

## Example Directory Structure

### Using Markdown Loader (Recommended)

```
my_reproduction/
├── paper.pdf                    # Original paper
├── extracted/
│   └── paper.md                 # Extracted text with figure links (from marker/nougat)
├── figures/                     # Auto-created by load_paper_from_markdown()
│   ├── Fig1.png                 # Downloaded figures
│   ├── Fig2a.png
│   ├── Fig2b.png
│   └── Fig3.png
├── paper_input.json             # Generated by save_paper_input_json()
└── outputs/                     # Created by ReproLab
    └── smith2023_plasmon/
        ├── plan.json
        ├── assumptions.json
        └── ...
```

### Using Manual Preparation

```
my_reproduction/
├── paper.pdf                    # Original paper
├── extracted/
│   └── paper.md                 # Extracted text (markdown)
├── figures/
│   ├── fig1_schematic.png       # Manually extracted figures
│   ├── fig2a_absorption.png
│   ├── fig2a_absorption.csv     # Digitized data
│   ├── fig2b_transmission.png
│   └── fig3_field_map.png
├── paper_input.json             # Manually created PaperInput definition
└── outputs/                     # Created by ReproLab
    └── smith2023_plasmon/
        ├── plan.json
        ├── assumptions.json
        └── ...
```

---

## Troubleshooting

### Markdown Loading Issues

| Problem | Solution |
|---------|----------|
| Figures not found | Check if paths are relative; provide `base_url` parameter |
| Download timeout | Increase `figure_timeout` parameter (default: 30s) |
| 403/404 errors | Some servers block automated downloads; manually download figures |
| Wrong figure IDs | Alt text missing; manually rename or edit `paper_input.json` |
| SVG/EPS figures | Convert to PNG for best vision model compatibility |
| Duplicate figure IDs | Loader auto-appends `_1`, `_2`, etc. to duplicates |
| Paper too long warning | Remove references, acknowledgments, or trim literature review |
| Supplementary not loading | Check path is correct; ensure file exists |
| SI figures mixed with main | SI figures are auto-prefixed with "S"; check `supplementary_figures` field |

### Text Extraction Issues

| Problem | Solution |
|---------|----------|
| Equations as images | Describe key equations in text ("Drude model: ε = ε∞ - ωp²/(ω² + iγω)") |
| Missing Greek letters | Use spelled-out names ("wavelength lambda = 500nm") |
| Garbled tables | Re-type critical values manually |
| Missing sections | Check if paper has supplementary material |

### Figure Extraction Issues

| Problem | Solution |
|---------|----------|
| Low resolution | Take manual screenshot at high zoom |
| Multiple plots in one image | Crop into separate files |
| Axes unreadable | Include axis values in description field |
| Color plots as grayscale | Re-export from PDF with color settings |

### Digitization Issues

| Problem | Solution |
|---------|----------|
| Noisy plot | Use manual point selection in WebPlotDigitizer |
| Log scale | Set axis type correctly in digitizer |
| Multiple curves | Digitize each curve to separate CSV |
| Data overlap | Focus on key wavelength regions only |

---

## Recommended Tools Summary

| Task | Tool | Install |
|------|------|---------|
| PDF → Text | marker | `pip install marker-pdf` |
| PDF → Text (complex) | nougat | `pip install nougat-ocr` |
| Markdown → PaperInput | `load_paper_from_markdown()` | Built-in (auto-downloads figures) |
| Extract images | PyMuPDF | `pip install pymupdf` |
| Digitize plots | WebPlotDigitizer | [Online tool](https://automeris.io/WebPlotDigitizer/) |
| Image editing | GIMP, Preview | Standard tools |

---

## Next Steps

After preparing your paper input:

1. **Run the system**: Pass `paper_input` to the ReproLab graph
2. **Review the plan**: Check the generated `plan.json` for accuracy
3. **Validate materials**: Confirm material data at the Stage 0 checkpoint
4. **Monitor progress**: Watch `progress.json` as stages complete
5. **Review report**: Examine the final `REPRODUCTION_REPORT.md`

