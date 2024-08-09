# This will call the deep_learning.py cli programmatically to save the
# inference results for each probe / head / user, etc.

# Note: THIS ASSUMES THAT ANY DIRECTORY CONTAINING REPLICATES
# CONTAINS ONLY REPLICATES

import tbitk.ai.dl_cli as dl_cli

from pathlib import Path

# Note that these are relative paths!
ROOT_INPUT_DATA_DIR = Path("../../data/unskilled_operator-20220421/")
RAW_DATA_DIR = ROOT_INPUT_DATA_DIR / "raw"
PREPROCESSED_DATA_DIR = ROOT_INPUT_DATA_DIR / "preprocessed"

ROOT_OUTPUT_DATA_DIR = Path("./data/uos_inference_results/").resolve()

MODEL_ROOT_DIR = Path("data/inference_models")
CLARIUS_MODEL_PATH = (
    MODEL_ROOT_DIR / "clarius-l7hd/20220408/duke_study_clarius_small_8c8772.pt"
).resolve()
BUTTERFLY_MODEL_PATH = (
    MODEL_ROOT_DIR / "butterfly-iq/20220408/duke_study_butterfly_small_8c8772.pt"
).resolve()
GENERAL_MODEL_PATH = (
    MODEL_ROOT_DIR / "general_purpose/20220408/duke_study_general_small_8c8772.pt"
).resolve()


PROBE_SUBDIRS = ["interson-spl01", "clarius-l7hd", "butterfly-iq"]
PROBE_SUBDIR_TO_MODEL_PATH = {
    "clarius-l7hd": [CLARIUS_MODEL_PATH, GENERAL_MODEL_PATH],
    "butterfly-iq": [BUTTERFLY_MODEL_PATH, GENERAL_MODEL_PATH],
    "interson-spl01": [GENERAL_MODEL_PATH]
}


def get_paths_to_applicable_models(probe_subdir):
    return PROBE_SUBDIR_TO_MODEL_PATH[probe_subdir]


def get_save_inf_res_cmdline_list(
    globs, model_path, root_dir, combine_videos=False, combined_name=None, verbose=False
):
    if not isinstance(globs, list):
        raise TypeError(f"globs argument must be a list of strings, got {type(globs)}")

    cmd = ["save_inference_results"]

    # General options
    cmd.append("--print_found_files")

    # Root directory for the inference results
    cmd.append("--root_dir")
    cmd.append(str(root_dir))

    # Append all input files patterns
    cmd.append("--file_patterns")
    for g in globs:
        cmd.append(g)

    # Specify the model path
    cmd.append("--model_path")
    cmd.append(str(model_path))

    cmd.append("--keep_root")

    # Combine the videos and specify the name if necessary
    if combine_videos:
        cmd.append("--combine_sources")

        cmd.append("--combined_res_name")
        cmd.append(combined_name)
    else:
        cmd.append("--keep_file_structure")

    if verbose:
        print("\n\nConstructed args:", cmd)

    return cmd

# Output directory structure is ROOT_OUTPUT_DATA_DIR / probe / model / participant_number / inf_res
for probe_subdir in PROBE_SUBDIRS:
    working_dir = PREPROCESSED_DATA_DIR / probe_subdir

    model_paths = get_paths_to_applicable_models(probe_subdir)

    # First, detect directories with replicates, as these will
    # need to be done separately.
    paths_to_replicates = list(working_dir.glob("**/[1-9]-[1-9]-[2-9].*"))
    dirs_with_replicates = {p.parents[0].resolve() for p in paths_to_replicates}

    # Generate inference results for the paths with replicates,
    # if there are any.
    print("#" * 10, "Inference for replicates", "#" * 10)
    for dir_with_replicate in dirs_with_replicates:
        # Get all of the replicates
        filenames_in_replicate_dir = [p.name for p in dir_with_replicate.glob("*.mha")]
        unique_prefixes = {p[:-5] for p in filenames_in_replicate_dir}
        globs_for_replicates = [
            dir_with_replicate / (prefix + "*.mha") for prefix in unique_prefixes
        ]

        for replicate_glob in globs_for_replicates:
            for model_path in model_paths:
                model_name = model_path.stem
                participant_number = dir_with_replicate.name
                inf_res_save_dir = (
                    ROOT_OUTPUT_DATA_DIR
                    / probe_subdir
                    / model_name
                    / participant_number
                )
                combined_name = replicate_glob.name.replace("*.mha", "all_inf_res")
                args_list = get_save_inf_res_cmdline_list(
                    [str(replicate_glob)],
                    model_path,
                    inf_res_save_dir,
                    combined_name=combined_name,
                    combine_videos=True,
                    verbose=True,
                )
                dl_cli.main(args_list)

    # Now the easy part, for the rest of the directories,
    # just look for any remaining subdirectories to run the inference on
    # but dont combine the results. We can pass the globs directly to the
    # deep_learning.py commandline
    all_participant_subdirs = {str(p.resolve()) for p in working_dir.glob("[1-9]")}
    dirs_with_no_replicates = all_participant_subdirs - {
        str(p) for p in dirs_with_replicates
    }
    globs_for_remaining_dirs = []
    for dir_with_no_replicates in dirs_with_no_replicates:
        glob_for_dir = str(Path(dir_with_no_replicates) / "**/*.mha")
        globs_for_remaining_dirs.append(glob_for_dir)

    print("#" * 10, "Inference for non-replicates", "#" * 10)
    for model_path in model_paths:
        model_name = model_path.stem
        inf_res_save_dir = ROOT_OUTPUT_DATA_DIR / probe_subdir / model_name
        args_list = get_save_inf_res_cmdline_list(
            globs_for_remaining_dirs,
            model_path,
            inf_res_save_dir,
            verbose=True,
        )
        dl_cli.main(args_list)
