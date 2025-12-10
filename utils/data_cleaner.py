# run in root "python utils/data_cleaner.py". gives /cleaned/ directory in 2018-AVTrack360 to replace the old data with interpolated and padded data

import json
import numpy as np
import os
from pathlib import Path

def interpolate_to_uniform_grid(raw_data, video_length_in_s, dt=0.1):
    """
    Interpolate viewport data to uniform time grid with padding.
    
    Args:
        raw_data: List of dicts with 'sec', 'yaw', 'pitch', 'roll'
        video_length_in_s: vid length item from json dict
        dt: Time step in seconds (default 0.1 = 100ms)
    
    Returns:
        List of dicts with uniform timestamps starting at 0.0
    """
    if len(raw_data) < 2:
        return []
    
    # extract from original
    timestamps = np.array([frame['sec'] for frame in raw_data])
    yaw_values = np.array([frame['yaw'] for frame in raw_data])
    pitch_values = np.array([frame['pitch'] for frame in raw_data])
    roll_values = np.array([frame['roll'] for frame in raw_data])
    
    # 0.0 to last timestep based on dt and video length
    # goes to the last .1 of the video, regardless of data ending early
    num_steps = int(np.ceil(video_length_in_s / dt)) + 1
    uniform_times = np.arange(num_steps) * dt
    uniform_times = uniform_times[uniform_times <= video_length_in_s]
    
    # Interpolate and pad beginning and end
    yaw_interp = np.interp(uniform_times, timestamps, yaw_values)
    pitch_interp = np.interp(uniform_times, timestamps, pitch_values)
    roll_interp = np.interp(uniform_times, timestamps, roll_values)
    
    # return output list
    cleaned_data = []
    for t, yaw, pitch, roll in zip(uniform_times, yaw_interp, pitch_interp, roll_interp):
        cleaned_data.append({
            'sec': float(t),
            'yaw': float(yaw),
            'pitch': float(pitch),
            'roll': float(roll)
        })
    
    return cleaned_data

def clean_user_file(input_path, output_path):
    """
    Clean a single user json file.
    
    Args:
        input_path: Path to original json
        output_path: Path to cleaned json
    """
    # get dirty file
    with open(input_path, 'r') as f:
        data = json.load(f)

    # make a new list, get all the dirty data, interpolate it, and append it in the same structure as the original
    cleaned_data = []
    for video_entry in data['data']:
        video_id = video_entry.get('filename', '')
        hmd = video_entry.get('hmd', '')
        video_length = video_entry.get('video_length_in_s', None)
        raw_frames = video_entry.get('pitch_yaw_roll_data_hmd', [])

        if video_length is None:
            if raw_frames:
                video_length = raw_frames[-1]['sec']
            else:
                video_length = 0.0

        cleaned_frames = interpolate_to_uniform_grid(raw_frames, video_length)
        
        cleaned_data.append({
            'filename': video_id,
            'hmd': hmd,
            'pitch_yaw_roll_data_hmd': cleaned_frames,
            'video_length_in_s': video_length
        })

    # get data and label, make the file if it isn't replacing one, and write to file
    output_data = {'data': cleaned_data}

    if 'label' in data:
        output_data['label'] = data['label']

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

def clean_all_users(input_dir, output_dir, user_ids):
    """
    Clean all user files
    
    Args:
        input_dir: original directory of json dataset
        output_dir: new directory for cleaned json dataset
        user_ids: List of user IDs to process
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print(f"Cleaning viewport data...")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Users:  {len(user_ids)} total")
    print(f"{'-'*60}")
    
    success_count = 0
    error_count = 0
    
    # get file, check if it exists, create a cleaned copy, exception for any errors, then print any error count
    for user_id in user_ids:
        input_file = input_path / f"{user_id}.json"
        output_file = output_path / f"{user_id}.json"
        
        if not input_file.exists():
            print(f"User {user_id:2d}: File not found")
            error_count += 1
            continue
        
        try:
            clean_user_file(str(input_file), str(output_file))
            success_count += 1
            
        except Exception as e:
            print(f"User {user_id:2d}: Error: {e}")
            error_count += 1

    print(f"Complete: {success_count} successful, {error_count} errors")

if __name__ == "__main__":
    INPUT_DIR = "./data/2018-AVTrack360/Head_rotation"
    OUTPUT_DIR = "./data/2018-AVTrack360/cleaned"
    USER_IDS = range(2, 50)

    clean_all_users(INPUT_DIR, OUTPUT_DIR, USER_IDS)