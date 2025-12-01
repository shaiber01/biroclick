#!/usr/bin/env python3
"""
Generate Python TypedDict types from JSON schemas.

Run this whenever JSON schemas are modified:
    python scripts/generate_types.py

This ensures schemas/generated_types.py stays in sync with the JSON schemas.

Prerequisites:
    pip install datamodel-code-generator

The generated types are imported by schemas/state.py for type-safe access
to schema-defined structures.
"""

import subprocess
import sys
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCHEMA_DIR = PROJECT_ROOT / "schemas"
OUTPUT_FILE = SCHEMA_DIR / "generated_types.py"

# Schemas to include in generation
# Order matters for dependency resolution
SCHEMAS = [
    "plan_schema.json",
    "progress_schema.json",
    "metrics_schema.json",
    "report_schema.json",
    "assumptions_schema.json",
    "prompt_adaptations_schema.json",
]


def main():
    """Generate Python types from JSON schemas."""
    print("=" * 60)
    print("ReproLab Type Generator")
    print("=" * 60)
    
    # Verify schema directory exists
    if not SCHEMA_DIR.exists():
        print(f"Error: Schema directory not found: {SCHEMA_DIR}")
        sys.exit(1)
    
    # Collect existing schemas
    existing_schemas = []
    missing_schemas = []
    
    for schema in SCHEMAS:
        schema_path = SCHEMA_DIR / schema
        if schema_path.exists():
            existing_schemas.append(schema_path)
            print(f"  ✓ Found: {schema}")
        else:
            missing_schemas.append(schema)
            print(f"  ✗ Missing: {schema}")
    
    if not existing_schemas:
        print("\nError: No schema files found!")
        sys.exit(1)
    
    if missing_schemas:
        print(f"\nWarning: {len(missing_schemas)} schema(s) not found, skipping them")
    
    # Build command
    cmd = [
        sys.executable, "-m", "datamodel_code_generator",
        "--output-model-type", "typing.TypedDict",
        "--output", str(OUTPUT_FILE),
        "--input-file-type", "jsonschema",
        "--use-default",  # Include default values
        "--target-python-version", "3.11",
    ]
    
    # Add all schema files
    for schema_path in existing_schemas:
        cmd.extend(["--input", str(schema_path)])
    
    print(f"\nGenerating types to: {OUTPUT_FILE}")
    print(f"Command: {' '.join(cmd[:6])}...")
    
    # Run generator
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"\nError running datamodel-code-generator:")
        print(result.stderr)
        print("\nMake sure datamodel-code-generator is installed:")
        print("  pip install datamodel-code-generator>=0.25.0")
        sys.exit(1)
    
    # Add header comment to generated file
    if OUTPUT_FILE.exists():
        content = OUTPUT_FILE.read_text()
        
        header = '''"""
Auto-generated TypedDict types from JSON schemas.

╔══════════════════════════════════════════════════════════════════════╗
║  DO NOT EDIT MANUALLY - Changes will be overwritten!                 ║
║                                                                       ║
║  Regenerate with: python scripts/generate_types.py                   ║
╚══════════════════════════════════════════════════════════════════════╝

Source schemas:
'''
        for schema in existing_schemas:
            header += f"  - {schema.name}\n"
        
        header += '''
These types are imported by schemas/state.py for type-safe access
to schema-defined structures (ExtractedParameter, Discrepancy, etc.)

Usage:
    from schemas.generated_types import ExtractedParameter, StageProgress
"""

'''
        OUTPUT_FILE.write_text(header + content)
        print("✓ Added header comment")
    
    print("\n" + "=" * 60)
    print(f"✓ Successfully generated {OUTPUT_FILE}")
    print("=" * 60)
    
    # Show summary
    if OUTPUT_FILE.exists():
        lines = len(OUTPUT_FILE.read_text().split('\n'))
        size = OUTPUT_FILE.stat().st_size
        print(f"\nOutput: {lines} lines, {size:,} bytes")


if __name__ == "__main__":
    main()

