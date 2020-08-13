from aalto_submit import AaltoSubmission
from submit_utils import EasyDict


def main():
    submit_texturize()


def submit_texturize():
    submit_config = create_submit_config(
        run_func='submission_targets.texturize_target',
        task_description='texturize_gram_dark_spot_test',
        time='0-02:00:00',
        n_cores=1,
        memory=16,
        flag='GPU'
    )

    # Define the parameters that are passed to the run function.
    run_func_args = EasyDict()

    config = {
        'SOURCE': [
            'img/tex/big_pebbles.png', 'img/tex/bricks.png', 'img/tex/dirt.png',
            'img/tex/flowers.png', 'img/tex/flowers2.png', 'img/tex/gravel.png',
            'img/tex/marble.png', 'img/tex/wood.png'
        ],
        'output': '{command}_{source}_{variation}_{octave}_{type}.png',
        'size': (256, 256),
        'model': 'VGG11',
        'variations': 1,
        'quality': 4.0,
        'precision': None,
        'mode': "gram",
        'octaves': 4,
        'verbose': False,
        'seed': None,
        'device': None,
        'quiet': False,
        'layers': None,
        'TARGET': None,
        'zoom': 2,
        'weights': (1.0,),
        'help': False,
        'matrix': None,
        'crop': None,
        'backgrounds': [
            'img/bg/scene-human.png'
        ],
        'brightness': 5.0
    }

    run_func_args.config = config
    run_func_args.rel_dir = (
        'data/texture_synthesis'
    )

    submission = AaltoSubmission(run_func_args, **submit_config)
    submission.run_task()


def create_submit_config(
    run_func, task_description, time, n_cores, memory, flag
):
    submit_config = EasyDict()

    # Define the function that we want to run.
    submit_config.run_func = run_func

    # Define where results from the run are saved and give a name for the run.
    submit_config.run_dir_root = '<RESULTS>/graphics/horesoj1/procam/runs'
    submit_config.task_description = task_description
    # Cluster username (only necessary if different to local username)
    submit_config.username = 'horesoj1'

    # Define parameters for run time, number of GPUs, etc.
    submit_config.time = time  # In format d-hh:mm:ss.
    submit_config.num_gpus = 1
    submit_config.num_cores = n_cores
    submit_config.cpu_memory = memory  # In GB.
    # submit_config.gpu_type = 'pascal'

    submit_config.run_dir_extra_ignores = [
        'lib', '*.slrm', '*.sh', '.pytest_cache', 'pytest.ini', '__pycache__',
        '.ipynb_checkpoints', 'models', '.mypy_cache', 'output', 'texturize.egg-info'
    ]
    submit_config.use_singularity_on_dgx = False

    # Define the envinronment where the task is run.
    # Pick one of: 'L' (local), 'DGX' (dgx01, dgx02), 'DGX-COMMON' (dgx-common)
    # , GPU (nodes gpu[1-10], gpu[28-37]) or CPU (batch partition).
    submit_config.env = flag
    if submit_config.env != 'L':
        submit_config.modules = ['miniconda']
        submit_config.conda_env_name = 'texturize'

    return submit_config


if __name__ == "__main__":
    main()
