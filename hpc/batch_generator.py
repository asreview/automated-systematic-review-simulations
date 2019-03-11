import pandas as pd
import numpy as np
from batchgen import batch_from_strings
from sklearn.model_selection import ParameterGrid


def generate_params(params, fix_args, params_file_path):
    def create_df_parameter_grid(params):
        " create a parameter grid "
        grid = ParameterGrid(params)
        df = pd.DataFrame(list(grid))
        return df

    def create_df_args(fix_args, freq):
        '''go throw the params and make a df'''
        df_args = pd.DataFrame()
        for key in fix_args:
            df_args[key] = np.tile(fix_args[key], freq)
        return df_args

    df_pg = create_df_parameter_grid(params)
    df_args = create_df_args(fix_args, df_pg.shape[0])

    df_all = pd.concat([df_pg, df_args], axis=1)
    df_all.index.name = 'T'
    df_all.to_csv(params_file_path)

    return df_all


def generate_commands(data_file_path, params_file_path):
    """Create commands from a parameter (CSV) file.
       Arguments
       ---------
       data_file_path: str
           File with data to simulate.
       params_file_path: str
           File with parameter grid (CSV).
       config_file: str
       """
    params = pd.read_csv(params_file_path)
    base_job = "python3 -m asr simulate "
    param_names_all = list(params.columns.values)
    param_names = [p for p in param_names_all if p not in ['T', 'simulation']]

    script = []
    for row in params.itertuples():
        param_val = map(lambda p: ' --' + p + ' ' + str(getattr(row, p)), param_names)
        param_val_all = " ".join(list(param_val))
        job_str = base_job + data_file_path + param_val_all
        job_str += " --log_file output/results" + str(getattr(row, "T")) + ".log"
        script.append(job_str)

    print(script)
    return script


def pre_compute_defaults():
    """Define default the pre-compute commands

    Returns
    -------
    str:
        List of lines to execute before computation.
    """
    #Todo install asr package
    #check if results.log is a file or folder
    mydef = """module load eb
    module load Python/3.6.1-intel-2016b

    cd $HOME/asr
    mkdir -p "$TMPDIR"/output
    rm -rf "$TMPDIR"/results.log
    cp -r $HOME/asr/pickle "$TMPDIR"
    cd "$TMPDIR"
    """
    return mydef


def post_compute_defaults():
    """Definition of post-compute commands
    Returns
    -------
    str:
        List of lines to execute after computation.
    """
    mydef = 'cp -r "$TMPDIR"/output  $HOME/asr'
    return mydef


def generate_shell_script(data_file_path, params_file_path, config_file):
    """Create job script including job setup information for the batch system as well as job commands.
    Arguments
    ---------
    data_file_path: str
        File with systematic review data.
    params_file_path: str
        File with parameter grid (CSV).
    config_file: str
        configuration information for the batch system
    """
    script = generate_commands(data_file_path, params_file_path)
    script_commands = "\n".join(script)
    pre_com_string = pre_compute_defaults()
    print(pre_com_string)
    print(type(pre_com_string))
    post_com_string = post_compute_defaults()

    batch_from_strings(command_string=script_commands, config_file=config_file,
                           pre_com_string=pre_com_string,
                           post_com_string=post_com_string,
                           force_clear=True)

