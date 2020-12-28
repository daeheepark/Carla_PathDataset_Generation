import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import os

import copy
import pickle as pkl
from torchvision import transforms

import argparse
from tqdm import tqdm

p_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([23.0582], [27.3226]),
    transforms.Lambda(lambda x: F.log_softmax(x.reshape(-1), dim=0).reshape(x.shape[1:]))
])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([23.0582], [27.3226])
])

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def generateDistanceMaskFromColorMap(src, scene_size=(64, 64)):
    img = cv2.imread(src)
    img = cv2.resize(img, scene_size)
    raw_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(raw_image, 60, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_not(thresh)
    raw_image = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    raw_map_image = cv2.resize(raw_image.astype(np.float32), dsize=(100, 100), interpolation=cv2.INTER_LINEAR)
    raw_map_image[raw_map_image < 0] = 0  # Uniform on drivable area
    raw_map_image = raw_map_image.max() - raw_map_image  # Invert values so that non-drivable area has smaller values

    image = img_transform(raw_image)
    prior = p_transform(raw_map_image)

    return image, prior


def extract_carla_from_path(agent_past, agent_future, agent_translation):
    future_agents_traj = agent_future
    past_agents_traj = agent_past
    
    # Data clean fix (ignore this)
    for i in range(len(past_agents_traj)):
        center_point = ((past_agents_traj[i][-2]+future_agents_traj[i][0])/2)
        past_agents_traj[i][-1] = center_point

    # Compensate little deviations (~1e-13) in the ego's current locations.
    # So the locations are always (0, 0) exactly.
    ego_current = past_agents_traj[0, -1, :].copy()
    past_agents_traj[:, :, :] -= ego_current
    future_agents_traj[:, :, :] -= ego_current
    scene_size = [-56, 56]
    # Scaling trajectory from (-200,200) to (scene_size[0], scene_size[1])
    future_agents_traj = future_agents_traj*(scene_size[1] - scene_size[0])/400
    past_agents_traj = past_agents_traj*(scene_size[1] - scene_size[0])/400

    filename = path[len(os.path.dirname(path))+1:-4]
    scene_id = [ 'train', 'town01','traj', filename]

    # Filter the agents whose locations at t=0, are out of ROI
    # ROI === [(-56, 56), (-56, 56)]
    agents_current = past_agents_traj[ :, -1, :]
    oom_mask = np.all(np.abs(agents_current) < np.abs(scene_size[0]), axis=-1)
    num_agents = oom_mask.sum(axis=-1)
    past_agents_traj_filt = past_agents_traj[oom_mask]
    future_agents_traj_filt = future_agents_traj[oom_mask]
    
    # Apply sampling intervals
    past_agents_traj_filt = past_agents_traj_filt[:, -1::-sampling_interval][:, ::-1]
    future_agents_traj_filt = future_agents_traj_filt[:, sampling_interval-1::sampling_interval]

    # Make encode_coordinates, decode_rel_pos, decode_start_pos
    agents_current = past_agents_traj_filt[:, -1, :]
    decode_start_pos = past_agents_traj_filt[:, -1, :]
    decode_start_vel = past_agents_traj_filt[:, -1, :] - past_agents_traj_filt[:, -2, :]

    # Set dtypes float64 to float32
    past_agents_traj_filt = past_agents_traj_filt.astype(np.float32)
    future_agents_traj_filt = future_agents_traj_filt.astype(np.float32)
    decode_start_pos = decode_start_pos.astype(np.float32)
    decode_start_vel = decode_start_vel.astype(np.float32)

    past_agents_traj_len = np.full((num_agents, ), int(20//sampling_interval), np.int64)
    
    future_agents_traj_len = np.full((num_agents, ), int(30//sampling_interval), np.int64)
    future_agent_masks = np.full((num_agents, ), True, np.bool)
    condition =1 
    
    return past_agents_traj_filt, past_agents_traj_len, future_agents_traj_filt, future_agents_traj_len, future_agent_masks, decode_start_pos, decode_start_vel

def get_agent_mask(agent_past, agent_future, agent_translation):
    
    map_width = 112
    map_height = 112
    time_rate = 0.5

    num_agents = len(agent_past)
    future_agent_masks = [True] * num_agents

    past_agents_traj = [[[0., 0.]] * 4] * num_agents
    future_agents_traj = [[[0., 0.]] * 6] * num_agents

    past_agents_traj = np.array(past_agents_traj)
    future_agents_traj = np.array(future_agents_traj)

    past_agents_traj_len = [4] * num_agents
    future_agents_traj_len = [6] * num_agents

    decode_start_vel = [[0., 0.]] * num_agents
    decode_start_pos = [[0., 0.]] * num_agents

    ego_cur_pose = copy.deepcopy(agent_translation[0])
    for idx, path in enumerate(zip(agent_past, agent_future)):
        past = path[0]
        future = path[1]
        if isinstance(future[0], int) or len(future[0])!=2:
            # print(path[1])
            future_agent_masks[idx] = False
            continue
            # future = []
            # print(path[1])
            # for i in range(len(path[1])-1):
                # future.append(path[1][i+1])
                # future = np.multiply(np.array(future),10)
            # future = np.array(future)
        past -= ego_cur_pose
        future -= ego_cur_pose
        pose = past[-1]
        decode_start_pos[idx] = agent_translation[idx]

        # agent filtering
        side_length = map_width // 2
        if len(past) == 0 or len(future) == 0 \
                or np.max(pose) > side_length or np.min(pose) < -side_length:
            future_agent_masks[idx] = False

        # agent trajectory
        if len(past) < 4:
            past_agents_traj_len[idx] = len(past)
        for i, point in enumerate(past[:4]):
            past_agents_traj[idx, i] = point

        if len(future) < 6:
            future_agents_traj_len[idx] = len(future)
        for i, point in enumerate(future[:6]):
            future_agents_traj[idx, i] = point

        # vel, pose
        if len(future) != 0 and not isinstance(agent_translation[idx], int):
            # print(agent_translation[idx])
            decode_start_vel[idx] = (future[0] - agent_translation[idx]) / time_rate
    # print(np.shape(decode_start_pos))
    return past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, \
           future_agent_masks, decode_start_vel, decode_start_pos

def dataProcessing(traj_path, map_path, traj_list, map_list, idx = 0, virtual=False):

    # map mask & prior mask
    whole_map_path = map_path + map_list[idx][1] + '/' + map_list[idx][0]
    map_image, prior = generateDistanceMaskFromColorMap(whole_map_path, scene_size=(64, 64))

    # agent mask
    whole_path = traj_path + traj_list[idx][1] + '/' + traj_list[idx][0]
    with open(whole_path, 'rb') as f:
        raw_path = pkl.load(f)
    agent_past = raw_path["agent_pasts"]
    agent_future = raw_path["agent_futures"][:,0:]
    agent_translation = raw_path["agent_pasts"][:,-1]
    map_name = map_list[idx][0].split('_',3)
    # scene_id = int(map_name[1]) * 1000000 + int(map_name[3].split('.',1)[0])
    # print(scene_id)
    scene_id = whole_map_path
    # print(np.shape(raw_path["agent_futures"]))
    past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, \
    future_agent_masks, decode_start_vel, decode_start_pos = get_agent_mask(agent_past, agent_future, agent_translation)

    episode = None
    episode = [past_agents_traj, past_agents_traj_len, future_agents_traj,
                future_agents_traj_len, future_agent_masks,
                np.array(decode_start_pos), np.array(decode_start_vel),
                map_image, prior, scene_id]

    return episode


def dataGeneration(traj_path, map_path, traj_list, map_list):
    episodes = []
    N = len(traj_list)

    print("{} number of samples".format(N))
    # count the number of curved agents

    # original data
    for idx in tqdm(range(N), desc='load data'):
        episode = dataProcessing(traj_path, map_path, traj_list, map_list, idx)
        if sum(episode[4]) > 0:
            episodes.append(episode)

    print("--- generation finished ---")

    return episodes


parser = argparse.ArgumentParser(description='load details')
parser.add_argument('--traj_path', type=str, help='path of trajectory', default='./data_pdh/path/')
parser.add_argument('--map_path', type=str, help='path of map', default='./data_pdh/map/')
parser.add_argument('--result_path', type=str, help='path for results', default='./')

args = parser.parse_args()

if __name__ == "__main__":
    TRAJ_PATH = args.traj_path
    MAP_PATH = args.map_path
    result_path = args.result_path

    extension = ".pkl"
    right_list = sorted([(name, 'right') for name in os.listdir(TRAJ_PATH + 'right/') if name.lower().endswith(extension)])
    left_list = sorted([(name, 'left') for name in os.listdir(TRAJ_PATH + 'left/') if name.lower().endswith(extension)])
    fair_length = max(len(right_list), len(left_list)) * 2
    for_list = sorted([(name, 'forward') for name in os.listdir(TRAJ_PATH + 'forward/') if name.lower().endswith(extension)])[:fair_length]
    # for_list = sorted([(name, 'forward') for name in os.listdir(TRAJ_PATH + 'forward/') if name.lower().endswith(extension)])
    # stop_list = sorted([(name, 'stop') for name in os.listdir(TRAJ_PATH + 'stop/') if name.lower().endswith(extension)])[:fair_length]
    traj_list = sorted(for_list + right_list + left_list)
    # traj_list = sorted(for_list)

    extension = ".jpg"
    right_map_list = sorted([(name, 'right') for name in os.listdir(MAP_PATH + 'right/') if name.lower().endswith(extension)])
    left_map_list = sorted([(name, 'left') for name in os.listdir(MAP_PATH + 'left/') if name.lower().endswith(extension)])
    for_map_list = sorted([(name, 'forward') for name in os.listdir(MAP_PATH + 'forward/') if name.lower().endswith(extension)])[:fair_length]
    # for_map_list = sorted([(name, 'forward') for name in os.listdir(MAP_PATH + 'forward/') if name.lower().endswith(extension)])
    # stop_map_list = sorted([(name, 'stop') for name in os.listdir(MAP_PATH + 'stop/') if name.lower().endswith(extension)])[:fair_length]
    map_list = sorted(right_map_list + left_map_list + for_map_list)
    # map_list = sorted(for_map_list)

    # test
    episode = dataProcessing(TRAJ_PATH, MAP_PATH, traj_list, map_list)
    print("test 100: {}".format(episode))
    print("Generation start...")

    # main
    parsed_data = dataGeneration(TRAJ_PATH, MAP_PATH, traj_list, map_list)

    print("Number of Data: {}".format(len(parsed_data)))

    filename = result_path + 'carla_train_2hz_' + str(len(parsed_data))
    valname = result_path + 'carla_val_2hz_' + str(len(parsed_data))
    with open(filename + '.pickle', 'wb') as f:
        pkl.dump(parsed_data[:28000], f, pkl.HIGHEST_PROTOCOL)
        # pkl.dump(parsed_data, f, pkl.HIGHEST_PROTOCOL)
    with open(valname + '.pickle', 'wb') as f:
        pkl.dump(parsed_data[28000:], f, pkl.HIGHEST_PROTOCOL)
    print("--- finished ---")
    print("number of data: {}".format(len(parsed_data)))


