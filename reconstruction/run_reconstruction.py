import json
from reconctruction import reconstruct_document

# Load the layout JSON
with open('reconstruction_input/layout.json', 'r', encoding='utf-8') as f:
    layout_json = json.load(f)

print("Running document reconstruction...")
print(f"Pages folder: reconstruction_input/pages")
print(f"Output will be saved to: reconstruction_output/")

# Create output directory
import os
os.makedirs("reconstruction_output", exist_ok=True)

# Run reconstruction
reconstruct_document(
    layout_json=layout_json,
    pages_folder="reconstruction_input/pages",
    output_pdf_path="reconstruction_output/reconstructed_document.pdf",
    translate=False
)

print("\nTesting with translation...")
reconstruct_document(
    layout_json=layout_json,
    pages_folder="reconstruction_input/pages",
    output_pdf_path="reconstruction_output/reconstructed_document_translated.pdf",
    translate=True
)

print("\nReconstruction complete!")
print("Generated files:")
print("- reconstruction_output/reconstructed_document.pdf")
print("- reconstruction_output/reconstructed_document_translated.pdf")
