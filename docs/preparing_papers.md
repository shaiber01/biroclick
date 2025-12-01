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
5. **Handle duplicate IDs** by appending `_1`, `_2`, etc. (e.g., `fig1`, `fig1_1`, `fig1_2`)
6. Return a validated `PaperInput` structure

> **Note**: Duplicate figure IDs are common when PDF converters extract the same figure multiple times or when multi-panel figures have generic alt text. Check the output for `_1`, `_2` suffixes and consider editing the markdown to use unique alt text before loading.

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

#### Context Window Budget

Claude Opus 4.5 has a 200K token context window. Budget allocation:

| Component | Typical Tokens | Notes |
|-----------|---------------|-------|
| Paper text | 10K-50K | Main variable |
| System prompt | 2K-5K | Agent-specific |
| State context | 1K-3K | Grows with stages |
| Figure descriptions | 500-2K | Per figure ~100-200 tokens |
| Response space | 4K-8K | Leave room for output |
| **Safe paper limit** | **~150K tokens** | ~600K chars |

#### v1 Behavior: Paper Too Long

**In the current implementation (v1), papers exceeding the safe limit will cause the system to exit with an error.** The loader validates paper length and fails fast if it's too large.

```python
# v1 behavior - exits with error if too long
paper_input = load_paper_from_markdown(
    markdown_path="./extracted/paper.md",
    output_dir="./figures"
)
# Raises: ValueError("Paper exceeds maximum length (600K chars). 
#         Please manually trim references and non-essential sections before loading.")
```

**If you encounter this error:**
1. Manually remove the References section from your markdown
2. Remove Acknowledgments, Author Contributions, Funding sections
3. Remove detailed literature review paragraphs (keep methodology citations)
4. Re-run the loader

**Sections safe to remove manually:**
- References (20-30% of text)
- Acknowledgments
- Author contributions  
- Funding statements
- Detailed literature review

**NEVER remove:**
- Methods/Experimental
- Results
- Figure captions
- Key equations

#### Future Improvements (v2+)

The following features are planned for future versions:

- **Automatic trimming**: Smart removal of references and non-essential sections
- **Chunking strategy**: Split long papers by sections, load incrementally
- **Section filtering**: Load only sections relevant to target figures
- **Iterative loading**: Start with abstract+methods, add sections as needed

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

### Methods Section Identification

When using `load_paper_from_markdown()`, the system extracts the Methods section for the PromptAdaptorAgent. It searches for these common headers (case-insensitive):

| Header Pattern | Common In |
|----------------|-----------|
| `## Methods` | Most journals |
| `## Materials and Methods` | Biology, Chemistry |
| `## Experimental` | Physics, Engineering |
| `## Experimental Section` | ACS journals |
| `## Simulation Details` | Computational papers |
| `## Computational Methods` | Theory papers |
| `## Experimental Procedures` | Some Nature journals |

If none are found, the loader falls back to the first ~15,000 characters of the paper.

**Tip**: If your paper uses a non-standard header, consider manually adding `## Methods` above that section in the markdown before loading.

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

### Reproduction Quality Tiers

The quality of figure comparison depends on what data you provide. Understanding these tiers helps you decide where to invest preparation effort:

| Tier | Data Provided | Comparison Method | Confidence Level | Best For |
|------|---------------|-------------------|------------------|----------|
| **Gold** | Digitized CSV + images | MSE, R², correlation, peak extraction | Highest | Spectra, resonance positions, quantitative claims |
| **Silver** | Images only | Vision-based qualitative comparison | Medium | Field maps, complex multi-panel figures |
| **Bronze** | Text descriptions only | Manual comparison | Lowest | Schematics, when images unavailable |

**Gold Tier Benefits:**
- Automatic computation of mean squared error (MSE)
- Correlation coefficient and R² values
- Automatic peak detection and wavelength extraction
- Objective pass/fail based on thresholds
- Removes subjective visual judgment

**When to Invest in Gold Tier:**
- Main result figures supporting paper's key claims
- Figures showing resonance positions (wavelength shifts matter)
- Transmission/reflection/absorption spectra
- Any figure where you need <5% accuracy verification

**Silver Tier is Sufficient For:**
- Near-field maps (|E|² distributions)
- Mode profiles (shape matters more than exact values)
- Multi-panel figures with complex layouts
- Figures where qualitative agreement is acceptable

**Example Investment Strategy:**
```
Paper has 8 figures:
- Fig 1: Schematic → Skip (not reproducible)
- Fig 2a: Absorption spectrum → Gold tier (digitize - key validation)
- Fig 2b: Field map → Silver tier (image only)
- Fig 3a: Transmission spectrum → Gold tier (digitize - main result)
- Fig 3b: Transmission spectrum → Gold tier (digitize - main result)
- Fig 4: Dispersion diagram → Gold tier (digitize - quantitative claim)
- Fig S1: SEM image → Skip (not reproducible)
- Fig S2: Extended spectra → Silver tier (supporting data)

Digitization effort: 4 figures (~30-60 minutes)
Expected confidence: High for main claims
```

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
    # Supplementary uses nested SupplementaryInput structure (optional)
    "supplementary": {
        "supplementary_text": "",  # Add if available
        "supplementary_figures": [],
        "supplementary_data_files": []
    }
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
  "supplementary": {
    "supplementary_text": "",
    "supplementary_figures": [],
    "supplementary_data_files": []
  }
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
| Duplicate figure IDs | Loader auto-appends `_1`, `_2`, etc. to duplicates (see below) |
| Paper too long warning | Remove references, acknowledgments, or trim literature review |
| Supplementary not loading | Check path is correct; ensure file exists |
| SI figures mixed with main | SI figures are auto-prefixed with "S"; check `supplementary_figures` field |

#### Figure ID Collision Handling

When the markdown loader encounters figures with duplicate IDs, it automatically renames them to ensure uniqueness:

**How It Works:**
1. First figure with an ID keeps the original: `Fig1`
2. Subsequent figures get suffixed: `Fig1_1`, `Fig1_2`, etc.
3. This applies within a single paper's main text OR supplementary material
4. Main + supplementary figures don't conflict (SI prefixed with "S")

**Common Causes of Duplicate IDs:**
- Markdown has multiple `![Figure 1](...)` references
- PDF converter extracted figure twice (from different pages)
- Multi-panel figures (`a`, `b`, `c`) extracted separately without unique alt text
- Inline equations extracted as "Figure" with generic captions

**Example Collision Resolution:**
```
Markdown content:
![Figure 1](overview.png)      # → fig1
![Figure 1](detail_a.png)      # → fig1_1 (duplicate!)
![Figure 1](detail_b.png)      # → fig1_2 (duplicate!)
![Figure 2](spectrum.png)      # → fig2 (unique)

Resulting IDs:
- fig1, fig1_1, fig1_2, fig2
```

**How to Fix Collisions:**
1. **Best approach:** Edit the markdown before loading to use unique alt text
2. **After loading:** Edit the `paper_input.json` to rename IDs
3. **Accept as-is:** The auto-generated IDs work, but may be confusing in reports

**Checking for Collisions:**
```python
from src.paper_loader import load_paper_from_markdown

paper_input = load_paper_from_markdown(...)

# Check if any IDs were auto-renamed
for fig in paper_input['figures']:
    if '_1' in fig['id'] or '_2' in fig['id']:
        print(f"⚠️ Renamed figure: {fig['id']} from {fig.get('original_id', 'unknown')}")
```

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

