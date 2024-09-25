import numpy as np
import pandas as pd
import os
import shutil

def get_project_root():
    """Returns project root folder."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(os.path.dirname(current_dir))
    except NameError:
        return os.path.abspath(os.getcwd())

BASE_DIR = get_project_root()
TEST_RESULT_DIR = os.path.join(BASE_DIR, 'test_result')

def delete_files_in_record(directory):
    # Delete all files and subfolder in the recorded directory. 
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try: 
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason {e}')

def clear_files_in_folder():
    folders = ['loss', 'reward', 'coordinates', 'state']
    
    for folder in folders:
        clear_dir = os.path.join(TEST_RESULT_DIR, folder)
        if os.path.exists(clear_dir):
            delete_files_in_record(clear_dir)
            print(f"All files in {clear_dir} are deleted")
        else:
            print(f"The directory {clear_dir} does not exist")

def normalize_to_range(value, min_val, max_val, new_min, new_max, clip=False):
    
    # Normalize value to a specified new range by supplying the current range.
    
    value = float(value)
    min_val = float(min_val)
    max_val = float(max_val)
    new_min = float(new_min)
    new_max = float(new_max)

    if clip:
        return np.clip((new_max - new_min) / (max_val - min_val) * (value - max_val) + new_max, new_min, new_max)
    else:
        return (new_max - new_min) / (max_val - min_val) * (value - max_val) + new_max
    
def save_loss_plot(q_values, next_q_values, expect_q_values, loss, episode):
        
    data = {
        'Q_values': q_values,
        'Next_Q_values' : next_q_values,
        'Expected_Q_values' : expect_q_values,
        'MSE_Loss': loss,
    }
    df = pd.DataFrame(data)
    file_path = os.path.join(TEST_RESULT_DIR, 'loss', f'DQN Q_Values and Loss episode{episode}.csv')
    df.to_csv(file_path, index=False)
    
def save_reward_plot(reward_data, episode_count):
    file_path = os.path.join(TEST_RESULT_DIR, 'reward', f'reward_plot.csv')

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        new_data = pd.DataFrame({'Rewards': [reward_data], 'Episode' : [episode_count]})
        df = pd.concat([df, new_data], ignore_index=True)
    
    else:
        df = pd.DataFrame({'Rewards': [reward_data], 'Episode' : [episode_count]})
    df.to_csv(file_path, index=False)

def save_transition_plot(state, actions,episode):
    data = {
        'State': state,
        'Action': actions,
    }
    df = pd.DataFrame(data)
    file_path = os.path.join(TEST_RESULT_DIR, 'state', f'transitions{episode}_plot.csv')
    df.to_csv(file_path, index=False)
    
def save_x_y_plot(x_y,episode):
    data = {
        'x_y': x_y,
    }
    df = pd.DataFrame(data)
    file_path = os.path.join(TEST_RESULT_DIR, 'coordinates',f'coordinates{episode}_plot.csv')
    df.to_csv(file_path, index=False)
