import datetime
import os
import pandas as pd
import torchio as tio
import warnings
import wandb
import torch
import random
import numpy as np
import yaml

from net.config.parser import setup_config
from dataset.statistics.split import perform_crossfold_split
from dataset.statistics.prepare import perform_prepare
from dataset.preprocess import perform_preprocessing
from dataset.generate_targets import perform_generate
from logger_setup import setup_logger
from dataset.utility.loaders import get_train_val_dl, get_test_dl
from net.model.utility.get_fresh_model import get_fresh_model
from net.phases.train import train_one_ep
from net.phases.infer import infer_one_ep
from net.model.utility.checkpoints import load_checkpoint, save_checkpoint
from net.config.wandb import initialize_wandb
# Plotting imports made conditional to avoid pyvista/trame conflicts when not needed
try:
    from net.plot.volumes import perform_plot_3d, start_game_3d
except ImportError:
    perform_plot_3d = None
    start_game_3d = None
try:
    from scripts.script_concept_fig import create_disc_figure
    from scripts.script_weeks_fig import create_weeks_plot
except ImportError:
    create_disc_figure = None
    create_weeks_plot = None
from utils import *

def pipe(dataframe, experiment_dir, params, transformations, global_wandb_steps, validate=True):
    train_dl, val_dl = get_train_val_dl(dataframe, params, transformations=transformations)
    if len(val_dl) == 0: val_dl = get_test_dl(dataframe, params, transformations=transformations)
    logger.info(f"Training - validation subset sizes: {len(train_dl.dataset), len(val_dl.dataset)}")
    # Initialize model, loss, optimizer
    model, criteria, optimizer, multi_noise_loss, best_val_loss = get_fresh_model(params)
    logger.info(optimizer)
    for criterion in criteria:
        logger.info(criterion)

    train_losses = []
    val_losses = []
    val_loss = torch.inf
    # Epoch training loop
    for epoch in range(0, params.training.epochs):
        logger.info(f" \n[Epoch {epoch + 1}/{params.training.epochs}] \n ")

        # Train for one epoch
        train_loss, global_wandb_steps = train_one_ep(
            model, 
            train_dl, 
            criteria, 
            multi_noise_loss if params.training.mnl else None,
            optimizer, 
            params.device, 
            global_wandb_steps,
            params.wandb.log,
            progress_bar=params.training.progress_bar
        )

        if validate:
            val_loss, global_wandb_steps, ep_scores = infer_one_ep(
                model, 
                val_dl, 
                criteria, 
                multi_noise_loss if params.training.mnl else None,
                params.device,
                global_wandb_steps, 
                params.wandb.log,
                lmks = params.target.lmks,
                experiment_dir=experiment_dir,
                eval=False, 
                **namespace_to_dict(params.validation)
            )
        
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss, 
                                os.path.join(experiment_dir, f'best.pt'))

        # Save last model
        save_checkpoint(model, optimizer, epoch, val_loss, 
                        os.path.join(experiment_dir, f'last.pt'))

        logger.info(f"Train loss: {train_loss}")
        logger.info(f"Validation loss: {val_loss}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if params.wandb.log:
            global_wandb_steps = log_epoch_to_wandb(train_loss, val_loss, ep_scores, global_wandb_steps)

    return train_losses, val_losses

print("Torch cuda is available? ", torch.cuda.is_available())
# Suppress UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
# Initialize parameters
# params = parameters_parsing()
params = setup_config()
# Apply reproducibility settings
# Set random seed from parameters for reproducibility
torch_seed = params.repro.model_seed
# Set environment variables for deterministic behavior
os.environ["PYTHONHASHSEED"] = str(torch_seed)  # Python hash randomization
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # CUDA determinism configuration
# Set PyTorch seeds for reproducibility across CPU and GPU
torch.manual_seed(torch_seed)  # CPU seed
torch.cuda.manual_seed(torch_seed)  # GPU seed (current device)
torch.cuda.manual_seed_all(torch_seed)  # All GPU devices
# Set NumPy and random module seeds
np.random.seed(torch_seed)  # NumPy random seed
random.seed(torch_seed)  # Python random module seed
# Enable deterministic algorithms in PyTorch
torch.backends.cudnn.deterministic = params.repro.deterministic  # Use deterministic CUDA algorithms
torch.backends.cudnn.benchmark = params.repro.benchmark  # Disable auto-tuning (slower but reproducible)
# torch.use_deterministic_algorithms(True)  # Force deterministic behavior globally

# Convert to dictionary (if Namespace or similar)
params_as_dict = namespace_to_dict(params)  # or: params.__dict__ if vars() doesn't work
if params.mode in ['test', 'interactive_plot', 'interactive_game'] or params.resume:
    experiment_dir = params.checkpoint
else:
    experiment_dir, experiment_name = create_experiment_id(params.prefix)
    # Write to JSON
    with open(os.path.join(experiment_dir, 'params.yaml'), 'w') as f:
        yaml.dump(params_as_dict, f, indent=6)

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# save_experiment_parameters(experiment_dir, experiment_name, params, stamp)

# Initialize logger
logger = setup_logger(experiment_dir)
logger.info(f"Experiment directory created at: {experiment_dir}")
logger.info(f"Experiment started and parameters are saved. This is the experiment dir: {experiment_dir} ")
logger.info(f"System path set to: {params.ds.sys}")
logger.info(f"Device set to: {params.device}")
logger.info("Searching for master dataframe in system+dataset_root... \n If you did not prepare such dataframe or you think it misses some samples, please re-run this script in prepare mode and specify datasets")

if 'script' in params.mode:
    import seaborn as sns
    
    logger.info(f"Running a script mode: {params.mode}")

    sns.set_style(
        'whitegrid',
        {'font.family': 'sans-serif', 'font.sans-serif': 'Verdana'}
    )
    sns.set_theme(
        'paper',
        'whitegrid',
        font_scale=2,
        palette='husl'
    )

if params.mode == 'script_concept':
    logger.info("Running script_concept_fig.py")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    create_disc_figure(params.script.save_dir, params.script.temp_dir, params.loss.w, **namespace_to_dict(params.script.concept))
    exit(0)  # Exit after running the script to prevent further execution

if params.mode == 'prepare':
    main_dataframe_path = perform_prepare(params) # Create info frames

# Load or create sinfo dataframe
main_dataframe_path = os.path.join(params.ds.sys, params.ds.root, params.ds.dataframe + '.csv')
if os.path.exists(main_dataframe_path):
    main_dataframe = pd.read_csv(main_dataframe_path)
else:
    logger.error(f"Master dataframe not found @ {main_dataframe_path}.")
    raise FileNotFoundError("Master dataframe not found.")

if params.mode == 'presplit':
    splitdataframe_path = perform_crossfold_split(main_dataframe, params)

if params.mode == 'preprocess':
    perform_preprocessing(main_dataframe, params, logger)

logger.info(f"Master dataframe path: {main_dataframe_path}")
logger.info(f"Total number of input in master dataframe: {len(main_dataframe)}")
logger.info(f"Training - validation - test subset sizes: {len(main_dataframe[main_dataframe['set'] == 0]), len(main_dataframe[main_dataframe['set'] == 1]), len(main_dataframe[main_dataframe['set'] == 2])}")

if params.mode == 'script_weeks':
    logger.info("Running script_weeks_fig.py")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    create_weeks_plot(params.script.save_dir, main_dataframe)
    exit(0)  # Exit after running the script to prevent further execution

# Define transformations for 3D images
transforms = [
                tio.RescaleIntensity((0, 1)), 
                # tio.Flip(axes=('L',)),
              ] if params.training.rescale_inputs else []
transformations = tio.Compose(transforms)
logger.info("Transformations are created.")

if params.mode == 'test_loaders':
    logger.info(f"I have been asked to test loaders... Here it goes.")
    train_dl, val_dl = get_train_val_dl(main_dataframe, params, transformations=transformations)
    test_loaders(train_dl)
    test_loaders(val_dl)

logger.info(f"I am cleaning the dataframe from non-existing files and nonfrontal cases... {len(main_dataframe)}")
main_dataframe = update_dataframe(main_dataframe, params)
logger.info(f"After cleaning, total number of input in master dataframe: {len(main_dataframe)}")
logger.info(f"!__[U]__! Training - validation - test subset sizes: {len(main_dataframe[main_dataframe['set'] == 0]), len(main_dataframe[main_dataframe['set'] == 1]), len(main_dataframe[main_dataframe['set'] == 2])}")

if params.mode in ['generate', 'interactive_plot', 'interactive_game']:
    logger.info(f"Running one of the interactive modes...")
    main_dataframe = main_dataframe.loc[main_dataframe['npid'].isin(params.test_patients)]
    if len(main_dataframe) == 0:
        raise KeyError("No test patient is given. Aborting mission...")
    if params.mode == 'interactive_plot':
        perform_plot_3d(main_dataframe, experiment_dir, params)
    elif params.mode == 'generate':
        perform_generate(main_dataframe, experiment_dir, params)
    elif params.mode == 'interactive_game':
        start_game_3d(main_dataframe, experiment_dir, params)

global_wandb_steps = {'train_loss': 0, 
                      'val_loss': 0, 
                      'epoch': 0}

if params.mode == 'train':
    logger.info("...... Training mode has started  .......")
    if params.wandb.log: initialize_wandb(params, experiment_name)
    # Initialize model, loss, optimizer
    model, criteria, optimizer, multi_noise_loss, _ = get_fresh_model(params)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    epoch = 0
    if params.resume:
        experiment_dir = params.checkpoint
        model_dir = experiment_dir + '/' + params.validation.use_model + '.pt'
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
        model, optimizer, epoch, best_val_loss = load_checkpoint(model, optimizer, model_dir) 
        logger.info(f"Resuming training from epoch {epoch} with best validation loss {best_val_loss}.")
        logger.info(f"Resuming training from checkpoint loaded at {stamp}.")
    
    train_losses, val_losses = pipe(main_dataframe, experiment_dir, params, transformations, global_wandb_steps)
    # Combine losses into a DataFrame
    loss_df = pd.DataFrame({
        'epoch': epoch + np.arange(len(train_losses)),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    # Save to CSV or any preferred format
    loss_df.to_csv(os.path.join(experiment_dir, f"losses_{stamp}.csv"), index=False)

    model_dir = experiment_dir + '/' + params.validation.use_model + '.pt'
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
    
    model, optimizer, epoch, best_val_loss = load_checkpoint(model, optimizer, model_dir) 
    logger.info(f"\n--- Testing {params.validation.use_model} model from {model_dir} --- \n")
        
    test_dl = get_test_dl(main_dataframe, params, transformations=transformations)
    if len(test_dl) == 0: _, test_dl = get_train_val_dl(main_dataframe, params, transformations=transformations)
    eval_dict = namespace_to_dict(params.validation)
    loss_dict = namespace_to_dict(params.loss)
    eval_dict['loss_params'] = loss_dict
    test_loss, global_wandb_steps, test_scores  = infer_one_ep(
                model, 
                val_dl, 
                criteria, 
                multi_noise_loss if params.training.mnl else None,
                params.device,
                global_wandb_steps, 
                params.wandb.log,
                lmks = params.target.lmks,
                experiment_dir=experiment_dir,
                eval=False, 
                **eval_dict
            )
    
    logger.info(f"Test loss: {test_loss}")
    logger.info(f"Test scores: \n {test_scores.drop(['nsid'], axis=1).mean()}")
    counter = 1
    save_scores = os.path.join(experiment_dir, "test_scores.csv")
    while os.path.exists(save_scores):
        save_scores = os.path.join(experiment_dir, f"test_scores_{counter}.csv")
        counter += 1
    test_scores.to_csv(save_scores, index=False)

    counter = 1
    save_mean_scores =os.path.join(experiment_dir, "test_scores_mean.csv")
    while os.path.exists(save_mean_scores):
        save_mean_scores = os.path.join(experiment_dir, f"test_scores_mean_{counter}.csv")
        counter += 1
    test_scores.drop(['nsid'], axis=1).mean().to_csv(save_mean_scores)

    if params.wandb.log: wandb.finish()

if params.mode == 'test':
    params.wandb.log = False
    logger.info("I am starting to test")
    # Initialize model, loss, optimizer
    model, criteria, optimizer, multi_noise_loss, _ = get_fresh_model(params)
    experiment_dir = params.checkpoint
    model_dir = experiment_dir + '/' + params.validation.use_model + '.pt'
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
    
    model, optimizer, epoch, best_val_loss = load_checkpoint(model, optimizer, model_dir) 
    logger.info(f"\n--- Testing {params.validation.use_model} model from {model_dir} --- \n")
    
    # test_patients = getattr(params, 'test_patients', [''])
    logger.info(f"The selected patient(s) are {params.test_patients} .")
    main_dataframe.loc[main_dataframe['npid'].isin(params.test_patients), "set"] = 2
    # Filter test patients first
    test_dl = get_test_dl(main_dataframe.loc[main_dataframe['npid'].isin(params.test_patients)], params, transformations=transformations)
    logger.info(f"!__[U]__! Test subset size: {len(test_dl)}")
    if len(test_dl) == 0: 
        test_dl = get_test_dl(main_dataframe, params, transformations)
        logger.info(f"!__[U]__! Couldn't find test patients, switch to entire test subset. Subset size: {len(test_dl)}")

        if len(test_dl) == 0:         
                _, test_dl = get_train_val_dl(main_dataframe, params, transformations=transformations)
                logger.info(f"!__[U]__! Switched to val subset. Subset size: {len(test_dl)}")

    eval_dict = namespace_to_dict(params.validation)
    loss_dict = namespace_to_dict(params.loss)
    eval_dict['loss_params'] = loss_dict
    test_loss, global_wandb_steps, test_scores  = infer_one_ep(
                model, 
                test_dl, 
                criteria, 
                multi_noise_loss if params.training.mnl else None,
                params.device,
                global_wandb_steps, 
                params.wandb.log,
                lmks = params.target.lmks,
                experiment_dir=experiment_dir,
                eval=True, 
                **eval_dict
            )
    
    logger.info(f"Test loss: {test_loss}")
    logger.info(f"Test scores: \n {test_scores.drop(['nsid'], axis=1).mean()}")
    save_scores = os.path.join(experiment_dir, "test_scores.csv")
    while os.path.exists(save_scores):
        save_scores = os.path.join(experiment_dir, f"test_scores_{stamp}.csv")
    test_scores.to_csv(save_scores, index=False)

    save_mean_scores =os.path.join(experiment_dir, "test_scores_mean.csv")
    while os.path.exists(save_mean_scores):
        save_mean_scores = os.path.join(experiment_dir, f"test_scores_mean_{stamp}.csv")
    test_scores.drop(['nsid'], axis=1).mean().to_csv(save_mean_scores)
