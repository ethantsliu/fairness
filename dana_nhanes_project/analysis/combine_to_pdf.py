from PIL import Image
import os

def combine_pngs_to_pdf():
    """Combine all PNG files in the figures directory into one PDF"""
    
    # Get all PNG files from figures directory
    figures_dir = 'figures'
    png_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
    png_files.sort()  # Sort to ensure consistent order
    
    print(f"Found {len(png_files)} PNG files:")
    for file in png_files:
        print(f"  - {file}")
    
    # Open the first image to get dimensions
    first_image_path = os.path.join(figures_dir, png_files[0])
    first_image = Image.open(first_image_path)
    
    # Convert all images to RGB mode (required for PDF)
    images = []
    for png_file in png_files:
        image_path = os.path.join(figures_dir, png_file)
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        images.append(img)
    
    # Save as PDF
    output_pdf = 'NHANES_Physical_Activity_Analysis_Complete.pdf'
    first_image.save(output_pdf, save_all=True, append_images=images[1:])
    
    print(f"\nSuccessfully created: {output_pdf}")
    print(f"PDF contains {len(images)} pages")
    
    # Clean up
    for img in images:
        img.close()

if __name__ == "__main__":
    combine_pngs_to_pdf()
