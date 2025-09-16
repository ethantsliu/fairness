from PIL import Image
import os

def combine_manuscript_figures_to_pdf():
    """Combine all four manuscript figures into one PDF"""
    
    # Get all manuscript PNG files from figures directory
    figures_dir = 'figures'
    manuscript_png_files = [
        'figure_traditional_risk_models.png',
        'figure_clinic_free_models.png', 
        'figure_reasonable_clinic_free_models.png',
        'figure_sociodemographic_models.png'
    ]
    
    print(f"Found {len(manuscript_png_files)} manuscript PNG files:")
    for file in manuscript_png_files:
        print(f"  - {file}")
    
    # Open the first image to get dimensions
    first_image_path = os.path.join(figures_dir, manuscript_png_files[0])
    first_image = Image.open(first_image_path)
    
    # Convert all images to RGB mode (required for PDF)
    images = []
    for png_file in manuscript_png_files:
        image_path = os.path.join(figures_dir, png_file)
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        images.append(img)
    
    # Save as PDF
    output_pdf = 'NHANES_Manuscript_Figures_Complete.pdf'
    first_image.save(output_pdf, save_all=True, append_images=images[1:])
    
    print(f"\nSuccessfully created: {output_pdf}")
    print(f"PDF contains {len(images)} pages")
    
    # Clean up
    for img in images:
        img.close()

if __name__ == "__main__":
    combine_manuscript_figures_to_pdf()
