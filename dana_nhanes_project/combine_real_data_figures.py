from PIL import Image
import os

def combine_real_data_figures_to_pdf():
    """Combine all real-data figures into one PDF"""
    
    # Get all real-data PNG files from figures directory
    figures_dir = 'figures'
    real_data_png_files = [
        'figure_3_roc_auc_real_data.png',
        'figure_4_delta_auc_real_data.png', 
        'figure_5_feature_importance_real_data.png',
        'performance_table_real_data.png'
    ]
    
    print(f"Found {len(real_data_png_files)} real-data PNG files:")
    for file in real_data_png_files:
        print(f"  - {file}")
    
    # Open the first image to get dimensions
    first_image_path = os.path.join(figures_dir, real_data_png_files[0])
    first_image = Image.open(first_image_path)
    
    # Convert all images to RGB mode (required for PDF)
    images = []
    for png_file in real_data_png_files:
        image_path = os.path.join(figures_dir, png_file)
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        images.append(img)
    
    # Save as PDF
    output_pdf = 'NHANES_CVD_Analysis_Real_Data_Complete.pdf'
    first_image.save(output_pdf, save_all=True, append_images=images[1:])
    
    print(f"\nSuccessfully created: {output_pdf}")
    print(f"PDF contains {len(images)} pages")
    
    # Clean up
    for img in images:
        img.close()

if __name__ == "__main__":
    combine_real_data_figures_to_pdf()
