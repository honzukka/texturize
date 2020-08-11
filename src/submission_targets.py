import os
import glob
import itertools

import torch

from texturize.logger import ansi, ConsoleLog
from texturize import api, io, commands
from texturize.procam import (
    Metadata,
    load_matrix_wrap as load_matrix,
    Procam,
    ProcamSimple
)


def texturize_target(submit_config, config, rel_dir):
    # Create the output folder and initialize source & output paths
    in_dir, out_dir = get_in_and_out_dir(submit_config, rel_dir)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    config['SOURCE'] = [
        os.path.join(in_dir, source) for source in config['SOURCE']
    ]
    config['output'] = os.path.join(out_dir, config['output'])
    if config['matrix']:
        config['matrix'] = os.path.join(in_dir, config['matrix'])
    if config['backgrounds']:
        config['backgrounds'] = [
            os.path.join(in_dir, background)
            for background in config['backgrounds']
        ]
    else:
        config['backgrounds'] = [None]

    # Setup the output logging and display the logo!
    log = ConsoleLog(config.pop("quiet"), config.pop("verbose"))

    # Separate command-specific arguments.
    sources, _, seed, root_output, backgrounds = [
        config.pop(k)
        for k in ("SOURCE", "TARGET", "seed", "output", "backgrounds")
    ]
    _, _, _ = [config.pop(k) for k in ("weights", "zoom", "help")]

    # If there's a random seed, use the same for all images.
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    device = config["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    matrix_path = config.pop("matrix")
    crop = config.pop("crop")
    brightness = config.pop("brightness")

    src_files = itertools.chain.from_iterable(glob.glob(s) for s in sources)
    bg_files = itertools.chain.from_iterable(glob.glob(b) for b in backgrounds)
    inputs = itertools.product(bg_files, src_files)
    for bg_path, src_path in inputs:
        # Create output subfolder.
        root = os.path.dirname(root_output)
        bg_name = os.path.splitext(os.path.basename(bg_path))[0]
        src_name = os.path.splitext(os.path.basename(src_path))[0]
        out_dir = os.path.join(root, "{}-{}".format(bg_name, src_name))
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, os.path.basename(root_output))

        # Load procam-related stuff.
        if matrix_path:
            metadata = Metadata.create_from_matrix_file(matrix_path)
            matrix = torch.from_numpy(load_matrix(matrix_path)).to(device)
            config["procam"] = Procam(matrix, metadata, crop=crop)
        else:
            if bg_path:
                background = io.load_tensor_from_image(
                    io.load_image_from_file(bg_path),
                    device=device, linearize=True
                )
                config["size"] = (background.shape[3], background.shape[2])
            else:
                background = None
            config["procam"] = ProcamSimple(background, brightness)

        # Load the images necessary.
        source_img = io.load_image_from_file(src_path)

        # Setup the command specified by user.
        cmd = commands.Remix(source_img)

        # Process the files one by one, each may have multiple variations.
        try:
            config["output"] = output
            config["output"] = config["output"].replace(
                "{source}", os.path.splitext(os.path.basename(src_path))[0]
            )

            result, filenames = api.process_single_command(cmd, log, **config)
            log.notice(ansi.PINK + "\n=> result:", filenames, ansi.ENDC)
        except KeyboardInterrupt:
            print(ansi.PINK + "\nCTRL+C detected, interrupting..." + ansi.ENDC)


def get_in_and_out_dir(submit_config, rel_dir, quiet=False):
    out_dir = os.path.join(submit_config.run_dir, 'output')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    in_dir = os.path.join(get_project_dir(submit_config), rel_dir)

    if not quiet:
        print('in_dir: {}'.format(in_dir))
        print('out_dir: {}'.format(out_dir))

    return in_dir, out_dir


def get_project_dir(submit_config):
    return os.path.abspath(
        os.path.join(submit_config.run_dir_root, os.pardir)
    )
