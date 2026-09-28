from pathlib import Path
import pickle
from huggingface_hub import HfApi
from datetime import date
from io import BytesIO
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pickle
from shapely.geometry.base import BaseGeometry
from shapely.geometry import shape



def load_majortom_intermediates(
    intermediate_root: str | Path,
) -> tuple[list[dict], list[dict]]:

    intermediate_root = Path(intermediate_root)

    all_samples = []
    all_metadata = []

    sample_files = sorted(intermediate_root.glob("*_samples.pkl"))

    for samples_path in sample_files:

        phisat_pid = samples_path.name.removesuffix("_samples.pkl")
        metadata_path = intermediate_root / f"{phisat_pid}_metadata.pkl"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing metadata intermediate for {phisat_pid}: "
                f"{metadata_path}"
            )

        with samples_path.open("rb") as f:
            samples = pickle.load(f)

        with metadata_path.open("rb") as f:
            metadata = pickle.load(f)

        all_samples.extend(samples)
        all_metadata.extend(metadata)

    return all_samples, all_metadata


def numpy_to_bytes(array: np.ndarray) -> bytes:
    """
    Serialize a NumPy array using the .npy format.
    Preserves dtype and shape.
    """
    buffer = BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def geometry_to_wkt(value) -> str | None:

    if value is None:
        return None

    if isinstance(value, BaseGeometry):
        return value.wkt

    if isinstance(value, dict):
        return shape(value).wkt

    raise TypeError(
        f"Unsupported geometry type: {type(value)}"
    )


def write_majortom_parquets(
    tmp_root: str | Path,
    product_id: str,
    repo_id: str,
) -> None:
    """
    Load all PhiSat -> Major-TOM intermediates and write:

        images_<product_id>.parquet
        metadata_<product_id>.parquet

    One row corresponds to one Major-TOM sample.

    Intermediate files are deleted only after both Parquet files
    have been successfully written.
    """

    tmp_root = Path(tmp_root)

    samples, metadata = load_majortom_intermediates(tmp_root)

    if len(samples) != len(metadata):
        raise ValueError(
            f"Found {len(samples)} samples but "
            f"{len(metadata)} metadata records"
        )

    if not samples:
        print(
            f"No Major-TOM intermediates found in "
            f"{tmp_root}"
        )
        return

    parquet_url = ( f"https://huggingface.co/datasets/" f"{repo_id}/resolve/main/" f"images/images_{product_id}.parquet" )

    image_rows = []

    for sample in samples:

        row = {}

        for column_name, value in sample.items():

            if isinstance(value, np.ndarray):
                row[column_name] = numpy_to_bytes(value)

            else:
                row[column_name] = value

        image_rows.append(row)

    
    metadata_rows = []

    geometry_columns = {
        "phisat2_corrected_geometry",
        "phisat2_original_geometry",
        "geometry",
    }

    for parquet_row, sample_metadata in enumerate(metadata):

        row = {}

        for column_name, value in sample_metadata.items():

            if column_name in geometry_columns:
                row[column_name] = geometry_to_wkt(value)

            else:
                row[column_name] = value

        row["parquet_url"] = parquet_url
        row["parquet_row"] = parquet_row
        
        metadata_rows.append(row)

    
    images_table = pa.Table.from_pylist(image_rows)
    metadata_table = pa.Table.from_pylist(metadata_rows)

    images_path = (tmp_root / f"images_{product_id}.parquet")
    metadata_path = (tmp_root / f"metadata_{product_id}.parquet")

    
    pq.write_table(
        images_table,
        images_path,
        compression="zstd",
    )

    pq.write_table(
        metadata_table,
        metadata_path,
        compression="zstd",
    )

    print(
        f"Wrote {len(samples)} samples:\n"
        f"  {images_path}\n"
        f"  {metadata_path}"
    )

    
    for path in tmp_root.glob("*.pkl"):
        path.unlink()

    print( f"Deleted intermediates from {tmp_root}")


def upload_majortom_parquets(
    tmp_root: str | Path,
    product_id: str,
    repo_id: str,
):
    tmp_root = Path(tmp_root)

    images_path = tmp_root / f"images_{product_id}.parquet"
    metadata_path = tmp_root / f"metadata_{product_id}.parquet"

    if not images_path.exists():
        raise FileNotFoundError(images_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    api = HfApi()

    # Upload images into the "images" folder
    api.upload_file(
        path_or_fileobj = images_path,
        path_in_repo = f"images/{images_path.name}",
        repo_id = repo_id,
        repo_type = "dataset",
    )

    # Upload metadata into the "metadata" folder
    api.upload_file(
        path_or_fileobj= metadata_path,
        path_in_repo = f"metadata/{metadata_path.name}",
        repo_id = repo_id,
        repo_type = "dataset",
    )

    print(
        f"Uploaded {images_path.name} and {metadata_path.name} "
        f"to {repo_id}"
    )


if __name__ == '__main__':

    tmp_root = Path("/home/sromagnoli/Phiesta/tmp")
    product_id = "6039"
    repo_id = "Sara-inblue/Phisat_to_MajorTom"
    upload_majortom_parquets(tmp_root, product_id, repo_id)