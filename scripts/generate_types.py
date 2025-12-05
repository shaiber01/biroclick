#!/usr/bin/env python3
"""
Generate Python TypedDict types from JSON schemas.

Run this whenever JSON schemas are modified:
    python scripts/generate_types.py

This ensures schemas/generated_types matches the JSON schemas.
The output is a Python package (directory) at schemas/generated_types/.

Prerequisites:
    pip install datamodel-code-generator

The generated types are imported by schemas/state.py for type-safe access
to schema-defined structures (ExtractedParameter, Discrepancy, etc.)
"""

import subprocess
import sys
import shutil
import tempfile
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCHEMA_DIR = PROJECT_ROOT / "schemas"
OUTPUT_DIR = SCHEMA_DIR / "generated_types"
OUTPUT_FILE_LEGACY = SCHEMA_DIR / "generated_types.py"

# Schemas to include in generation
# Order matters for dependency resolution? (Not for independent modules)
SCHEMAS = [
    # Core state artifacts
    "plan_schema.json",
    "progress_schema.json",
    "metrics_schema.json",
    "report_output_schema.json",
    "assumptions_schema.json",
    "prompt_adaptations_schema.json",
    # Agent output schemas
    "planner_output_schema.json",
    "simulation_designer_output_schema.json",
    "code_generator_output_schema.json",
    "prompt_adaptor_output_schema.json",
    "supervisor_output_schema.json",
    "plan_reviewer_output_schema.json",
    "design_reviewer_output_schema.json",
    "code_reviewer_output_schema.json",
    "execution_validator_output_schema.json",
    "physics_sanity_output_schema.json",
    "results_analyzer_output_schema.json",
    "comparison_validator_output_schema.json",
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
    
    # Create a temporary directory for inputs to ensure clean generation
    with tempfile.TemporaryDirectory() as temp_input_dir_str:
        temp_input_dir = Path(temp_input_dir_str)
        
        # Collect and copy existing schemas
        existing_schemas = []
        missing_schemas = []
        
        print("Collecting schemas...")
        for schema in SCHEMAS:
            src_path = SCHEMA_DIR / schema
            if src_path.exists():
                dst_path = temp_input_dir / schema
                shutil.copy2(src_path, dst_path)
                existing_schemas.append(dst_path)
                print(f"  ✓ Found: {schema}")
            else:
                missing_schemas.append(schema)
                print(f"  ✗ Missing: {schema}")
        
        if not existing_schemas:
            print("\nError: No schema files found!")
            sys.exit(1)
        
        if missing_schemas:
            print(f"\nWarning: {len(missing_schemas)} schema(s) not found, skipping them")
        
        # Prepare output directory
        # Remove legacy file if it exists
        if OUTPUT_FILE_LEGACY.exists():
            if OUTPUT_FILE_LEGACY.is_dir():
                 shutil.rmtree(OUTPUT_FILE_LEGACY)
            else:
                 OUTPUT_FILE_LEGACY.unlink()
            print(f"\nRemoved legacy output file: {OUTPUT_FILE_LEGACY}")
            
        # Clean/Create output directory
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(exist_ok=True)
        
        # Build command
        # We run on the directory to allow modular generation
        cmd = [
            sys.executable, "-m", "datamodel_code_generator",
            "--output-model-type", "typing.TypedDict",
            "--output", str(OUTPUT_DIR),
            "--input-file-type", "jsonschema",
            "--use-default",  # Include default values
            "--target-python-version", "3.11",
            "--input", str(temp_input_dir),
        ]
        
        print(f"\nGenerating types to directory: {OUTPUT_DIR}")
        print(f"Command: {' '.join(cmd[:6])}...")
        
        # Run generator
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"\nError running datamodel-code-generator:")
            print(result.stderr)
            print(result.stdout)
            print("\nMake sure datamodel-code-generator is installed:")
            print("  pip install datamodel-code-generator>=0.25.0")
            sys.exit(1)
            
        # Create __init__.py to export all types
        init_file = OUTPUT_DIR / "__init__.py"
        
        header = '''"""
Auto-generated TypedDict types from JSON schemas.

╔══════════════════════════════════════════════════════════════════════╗
║  DO NOT EDIT MANUALLY - Changes will be overwritten!                 ║
║                                                                       ║
║  Regenerate with: python scripts/generate_types.py                   ║
╚══════════════════════════════════════════════════════════════════════╝

This package exports types for all schemas.
"""

'''
        imports = []
        for py_file in sorted(OUTPUT_DIR.glob("*.py")):
            if py_file.name == "__init__.py":
                continue
            module_name = py_file.stem
            imports.append(f"from .{module_name} import *")
            
        init_content = header + "\n".join(imports) + "\n"
        init_file.write_text(init_content)
        print("✓ Created __init__.py with exports")
    
    print("\n" + "=" * 60)
    print(f"✓ Successfully generated types in {OUTPUT_DIR}")
    print("=" * 60)
    
    # Show summary
    file_count = len(list(OUTPUT_DIR.glob("*.py")))
    print(f"\nOutput: {file_count} files generated")


if __name__ == "__main__":
    main()
