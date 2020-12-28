import pandas as pd
import numpy as np
import pickle as pkl
import carla

# Sampling_Intv = 5

PIXELS_PER_METER = 2

MAP_DEFAULT_SCALE = 0.1
HERO_DEFAULT_SCALE = 1

PIXELS_AHEAD_VEHICLE = 0

CROP = 224

def get_df(scen_name, town_name, world, pasts_and_futures, start_frame, timestamp, hero_transform, map, _world_offset, data_decision, Sampling_Intv):

    frame_passed = timestamp.frame - start_frame

    if frame_passed % Sampling_Intv == (Sampling_Intv-1):

        hero_location_screen = world_to_pixel(location = hero_transform.location, rotation = hero_transform.rotation, hero=True, _world_offset = _world_offset)
        hero_front = hero_transform.get_forward_vector()
        translation_offset = (hero_location_screen[0] - map.size[0] / 2 + hero_front.x * 0,
                                (hero_location_screen[1] - map.size[1] / 2 + hero_front.y * 0))
        crop = map.crop((int(map.size[0]/2)+translation_offset[0]-int(CROP/2), int(map.size[1]/2)+translation_offset[1]-int(CROP/2), 
        int(map.size[0]/2)+translation_offset[0]+int(CROP/2), int(map.size[1]/2)+translation_offset[1]+int(CROP/2)))

        actor_list = world.get_actors().filter('vehicle.*')
        
        for actor in actor_list:
            loc = actor.get_location()
            pasts_and_futures['pasts'][actor.id].append(pasts_and_futures['futures'][actor.id][0])
            pasts_and_futures['futures'][actor.id].append([loc.x, loc.y])
                
        if data_decision == 1:
            lr_label = 'right'
        elif data_decision == -1:
            lr_label = 'left'
        elif data_decision == -2:
            lr_label = 'stop'
        else :
            lr_label = 'forward'
     
        if frame_passed >= (Sampling_Intv * (pasts_and_futures['past_length'] + pasts_and_futures['future_length']) - 1):
            print(lr_label)     

            pasts_array = np.array([val for val in pasts_and_futures['pasts'].values()])
            futures_array = np.array([val for val in pasts_and_futures['futures'].values()])

            data2 = {'agent_pasts' : pasts_array, 'agent_futures' : futures_array}
            pkl.dump(data2, open('./data_pdh/path/'+lr_label+'/%s_%s_frame%07d.pkl' % (scen_name, town_name, frame_passed), 'wb'))
            crop.save('./data_pdh/map/'+lr_label+f'/{scen_name}_{town_name}_{frame_passed:07d}.jpg')

    return 0


def world_to_pixel(location, offset=(0, 0), rotation=carla.Rotation(0,0,0), hero=False, _world_offset=None):
    """Converts the world coordinates to pixel coordinates"""
    # print(rotation)
    # print(offset)
    scale = 1.0
    # _world_offset = (0,0)
    if hero:
        angle = 90+rotation.yaw
        x_ori = scale * PIXELS_PER_METER * (location.x - _world_offset[0]) 
        y_ori = scale * PIXELS_PER_METER * (location.y - _world_offset[1])
        # x = x_ori * math.cos(math.radians(angle)) + y_ori * math.sin(math.radians(angle)) 
        # y = y_ori * math.cos(math.radians(angle)) - x_ori * math.sin(math.radians(angle))
        # print(x, x_ori)
        # print(y, y_ori)
        x = x_ori
        y = y_ori
    else:
        x = scale * PIXELS_PER_METER * (location.x - _world_offset[0])
        y = scale * PIXELS_PER_METER * (location.y - _world_offset[1])
    return [int(x - offset[0]), int(y - offset[1])]

def lr_decision():
    return left_or_right_decision