import pandas as pd
import os

def process_annotations(file_path, sheet_name, threshold):
    """
    Process the annotation data based on a given dissimilarity score threshold.

    Parameters:
    file_path (str): Path to the Excel file.
    sheet_name (str): Name of the sheet in the Excel file.
    threshold (float): Threshold for the dissimilarity score.

    Returns:
    pd.DataFrame: Updated DataFrame with carried forward annotations.
    float: Accuracy of the annotations.
    int: Number of frames that need annotation.
    """

    # Load the data from the specified sheet
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    # Copy the original annotations
    original_annotations = data.iloc[:, 1:-2].copy()  # Excluding frame number and Optical Flow Score

    # Initialize the 'annotation needed' column with 0s
    data['annotation needed'] = 0

    # Classify frames based on the threshold
    data.loc[data['Optical Flow Score'] > threshold, 'annotation needed'] = 1

    # Carry forward annotations for frames where no new annotation is needed
    for col in original_annotations.columns:
        for i in range(1, len(data)):
            if data.loc[i, 'annotation needed'] == 0:
                data.loc[i, col] = data.loc[i - 1, col]

    # Ensure columns match for comparison
    updated_annotations = data.iloc[:, 1:-2]  # Updated annotations, excluding frame number and last two columns
    matching_columns = updated_annotations.columns.intersection(original_annotations.columns)
    updated_annotations = updated_annotations[matching_columns]
    original_annotations = original_annotations[matching_columns]

    # Calculate accuracy
    correct_annotations = (original_annotations == updated_annotations).all(axis=1).sum()
    total_annotations = len(data)
    accuracy = correct_annotations / total_annotations

    # Count the number of frames that need annotation
    num_frames_to_annotate = data['annotation needed'].sum()

    return data, accuracy, num_frames_to_annotate

# Example usage
file_path = 'C:/Users/anton/Desktop/OF/heichole_optical_flow_scores/hei_chole24_optical_flow_scores.xlsx'  # Replace with your file path
sheet_name = 'inst'  # Replace with the name of your sheet
threshold = 80 # Set your desired threshold
updated_data, accuracy, num_frames_to_annotate = process_annotations(file_path, sheet_name, threshold)

# Generate output file path
file_name, file_extension = os.path.splitext(file_path)
output_file_path = f"{file_name}_{sheet_name}_accuracy_{threshold}{file_extension}"

# Save the updated data back to an Excel file
updated_data.to_excel(output_file_path, index=False)

print(f'File path: {file_path}')
print(f'Sheet Name: {sheet_name}')
print(f'Accuracy: {accuracy}')
print(f'Threshold: {threshold}')
print(f'Number of frames needing annotation: {num_frames_to_annotate}')
print(f'Updated file saved as: {output_file_path}')
