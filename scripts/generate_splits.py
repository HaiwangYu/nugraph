import argparse
import nugraph as ng
import warnings

# This script calls the generate_samples function as a static method,
# which is the correct way as revealed by the help() output.

# Ignore a common warning from h5py to keep the output clean
warnings.filterwarnings('ignore', '.*is an HDF5 object reference.*')

def main():
    parser = argparse.ArgumentParser(
        description="Uses the NuGraph library to generate sample splits in an HDF5 file."
    )
    parser.add_argument("--data-path", required=True, 
                        help="Path to the HDF5 file to modify in-place.")
    args = parser.parse_args()

    print(f"Calling static method generate_samples() on file: {args.data_path}")
    
    # Call the method directly on the class, not on an instance
    ng.data.NuGraphDataModule.generate_samples(data_path=args.data_path)

    print("✅ Done. The file should now contain the required sample splits.")

if __name__ == "__main__":
    main()