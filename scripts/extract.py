from pathlib import Path
from geosave_engine.geodata.utils.archives import extract_zip, cleanup_zip

def extract_and_rearrange_dw(zip_path: Path, extract_to: Path) -> None:

    expert_zip = extract_to / "Experts_tiles.zip"
    non_expert_zip = extract_to / "Non_expert_tiles.zip"
    validation_zip = extract_to / "validation_set_tiles.zip"
    
    try:
        extract_zip(zip_path, extract_to, skip_if_extracted=False)
        extract_zip(expert_zip, extract_to / "train", skip_if_extracted=False)
        extract_zip(non_expert_zip, extract_to / "train", skip_if_extracted=False)
        extract_zip(validation_zip, extract_to / "val", skip_if_extracted=False)

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # cleanup_zip(zip_path)
        cleanup_zip(expert_zip)
        cleanup_zip(non_expert_zip)
        cleanup_zip(validation_zip)

def extract_and_rearrange_dw_test(zip_path: Path, extract_to: Path) -> None:
    
    test_zip = extract_to / "dw_test.zip"

    try:
        extract_zip(zip_path, extract_to, skip_if_extracted=False)
        extract_zip(test_zip, extract_to / "test", skip_if_extracted=False)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cleanup_zip(test_zip)

if __name__ == "__main__":
    # Example usage
    zip_file = Path("data/dynamicworld.zip")
    extract_folder = Path("data/dynamicworld_raw")
    extract_and_rearrange_dw(zip_path=zip_file, extract_to=extract_folder)

    zip_file = Path("data/dynamicworld_test.zip")
    extract_and_rearrange_dw_test(zip_path=zip_file, extract_to=extract_folder)
