from PIL import Image
import os

def combine_updated_pngs_to_pdf():
    """Combine all updated PNG files in the figures directory into one PDF"""
    
    # Get all updated PNG files from figures directory
    figures_dir = 'figures'
    updated_png_files = [
        'performance_comparison.png',
        'delta_auc_analysis.png', 
        'clinic_free_vs_traditional.png',
        'confidence_intervals.png',
        'comprehensive_dashboard_updated.png'
    ]
    
    print(f"Found {len(updated_png_files)} updated PNG files:")
    for file in updated_png_files:
        print(f"  - {file}")
    
    # Open the first image to get dimensions
    first_image_path = os.path.join(figures_dir, updated_png_files[0])
    first_image = Image.open(first_image_path)
    
    # Convert all images to RGB mode (required for PDF)
    images = []
    for png_file in updated_png_files:
        image_path = os.path.join(figures_dir, png_file)
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        images.append(img)
    
    # Save as PDF
    output_pdf = 'NHANES_Physical_Activity_Analysis_Updated_Complete.pdf'
    first_image.save(output_pdf, save_all=True, append_images=images[1:])
    
    print(f"\nSuccessfully created: {output_pdf}")
    print(f"PDF contains {len(images)} pages")
    
    # Clean up
    for img in images:
        img.close()

if __name__ == "__main__":
    combine_updated_pngs_to_pdf()
