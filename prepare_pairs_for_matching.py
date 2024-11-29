import os


def main():

    # Define the input directory
    # Replace with your actual directory path
    input_dir = "input_pairs"
    output_paris_filename = "input_pairs.txt"

    # Get all image filenames from the directory
    all_images = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])

    if len(all_images) < 2:
        print("Not enough images in the directory to create pairs.")
    else:
        # Create the pairs.txt file
        with open(output_paris_filename, "w") as f:
            for i in range(len(all_images) - 1):
                img1 = all_images[i]
                img2 = all_images[i + 1]
                f.write(f"{img1} {img2}\n")

    print("pairs.txt file created successfully!")


if __name__ == "__main__":
    main()

