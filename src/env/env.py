# ## %%
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging; log = logging.getLogger(__name__)

from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import matplotlib as mpl
from functools import reduce
from enum import Enum, auto

from src.env import constants
from src.params import WALK_DIAGRAM_LOGGING_FREQUENCY

from typing import Tuple, List, Dict
import numpy.typing as npt

import wandb


def setup_logging(verbose: bool, experiment_name: str) -> None:
    logs_folder = constants.SAVE_PATH_LOGS
    if not os.path.exists(logs_folder): os.makedirs(logs_folder)

    logs_filename = os.path.join(logs_folder, f"logs_{experiment_name}.log")

    logging.basicConfig(
        filename=logs_filename, filemode="w",
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO
    )


class UserEnum(Enum):
    @classmethod
    def all(cls):
        return list(map(lambda c: c, cls))


class Status(UserEnum):
    VISCEK = auto()
    "Pedestrian under Viscek rules."

    FOLLOWER = auto()
    "Follower of the leader particle (agent)."

    EXITING = auto()
    "Pedestrian in exit zone."

    ESCAPED = auto()
    "Evacuated pedestrian."


class SwitchDistances:
    to_leader: float     = constants.SWITCH_DISTANCE_TO_LEADER
    to_exit: float       = constants.SWITCH_DISTANCE_TO_EXIT
    to_escape: float     = constants.SWITCH_DISTANCE_TO_ESCAPE
    to_pedestrian: float = constants.SWITCH_DISTANCE_TO_OTHER_PEDESTRIAN


def is_distance_low(
    pedestrians_positions: npt.NDArray, 
    destination: npt.NDArray, 
    radius: float
    ) -> npt.NDArray:
    """Get boolean matrix showing pedestrians,
    which are closer to destination than raduis 

    Args:
        pedestrians_positions (npt.NDArray): coordinates of pedestrians 
        (dim: [n x 2])
        
        destination (npt.NDArray): coordinates of destination
        (dim: [2])
        
        radius (float): max distance to destination

    Returns:
        npt.NDArray: boolean matrix
    """
    
    distances = distance_matrix(
        pedestrians_positions, np.expand_dims(destination, axis=0), 2
    )
    return np.where(distances < radius, True, False).squeeze()


def sum_distance(pedestrians_positions, destination):
    """Mean distance between pedestrians and destination"""
    distances = distance_matrix(
        pedestrians_positions, np.expand_dims(destination, axis=0), 2
    )
    return distances.sum() / pedestrians_positions.shape[0]

class Reward:
    def __init__(self, 
        is_new_exiting_reward: bool,
        is_new_followers_reward: bool,
        is_termination_agent_wall_collision: bool,
        init_reward_each_step: float
        ) -> None:
        
        self.init_reward_each_step = init_reward_each_step
        self.is_new_exiting_reward = is_new_exiting_reward
        self.is_new_followers_reward = is_new_followers_reward
        self.is_termination_agent_wall_collision = is_termination_agent_wall_collision

    def estimate_intrinsic_reward(self, pedestrians_positions, exit_position):
        """This is intrinsic reward, which is given to the agent at each step"""
        return (0 - sum_distance(pedestrians_positions, exit_position))

    def estimate_status_reward(self, old_statuses, new_statuses, timesteps, num_pedestrians):
        """This reward is based on how pedestrians update their status
        
        VISCEK or FOLLOWER --> EXITING  :  (15 +  5 * time_factor)
        VISCEK             --> FOLLOWER :  (10 + 10 * time_factor)
        """
        reward = self.init_reward_each_step
        time_factor = 1 - timesteps / (200 * num_pedestrians)

        # Reward for new exiting
        if self.is_new_exiting_reward:
            prev = np.logical_or(old_statuses == Status.VISCEK, 
                                 old_statuses == Status.FOLLOWER)
            curr = new_statuses == Status.EXITING
            n = sum(np.logical_and(prev, curr))
            reward += (15 + 10 * time_factor) * n

        # Reward for new followers
        if self.is_new_followers_reward:
            prev = old_statuses == Status.VISCEK
            curr = new_statuses == Status.FOLLOWER
            n = sum(np.logical_and(prev, curr))
            reward += (10 + 5 * time_factor) * n

        return reward


def update_statuses(statuses, pedestrian_positions, agent_position, exit_position):
    """Measure statuses of all pedestrians based on their position"""
    new_statuses = statuses.copy()

    following = is_distance_low(
        pedestrian_positions, agent_position, SwitchDistances.to_leader)
    new_statuses[following] = Status.FOLLOWER
    
    exiting = is_distance_low(
        pedestrian_positions, exit_position, SwitchDistances.to_exit)
    new_statuses[exiting] = Status.EXITING
    
    escaped = is_distance_low(
        pedestrian_positions, exit_position, SwitchDistances.to_escape)
    new_statuses[escaped] = Status.ESCAPED
    
    viscek = np.logical_not(reduce(np.logical_or, (exiting, following, escaped)))
    new_statuses[viscek] = Status.VISCEK
    
    return new_statuses


def count_new_statuses(old_statuses, new_statuses):
    """Get number of pedestrians, who have updated their status"""
    count = {}
    for status in Status.all():
        count[status] = sum(
            np.logical_and(new_statuses == status, old_statuses != status)
        )
    return count


class Pedestrians:
    num : int                                   # number of pedestrians
    
    positions : np.ndarray                      # [num x 2], r_x and r_y
    directions : np.ndarray                     # [num x 2], v_x and v_y
    statuses : np.ndarray                       # [num], status : Status 
    memory : Dict[str, List[np.ndarray]]        # needed for animation drawing (positions and statuses)

    def __init__(self, num : int):
        self.num = num

    def reset(self, agent_position, exit_position):
        self.positions  = np.random.uniform(-1.0, 1.0, size=(self.num, 2))
        self.directions = np.random.uniform(-1.0, 1.0, size=(self.num, 2))
        self.normirate_directions()
        self.statuses = np.array([Status.VISCEK for _ in range(self.num)])
        self.statuses = update_statuses(
            self.statuses,
            self.positions,
            agent_position,
            exit_position
        )
        self.memory = {'positions' : [], 'statuses' : []}
    
    def normirate_directions(self) -> None:
        x = self.directions
        self.directions = (x.T / np.linalg.norm(x, axis=1)).T

    def save(self):
        self.memory['positions'].append(self.positions.copy())
        self.memory['statuses'].append(self.statuses.copy())

    @property
    def status_stats(self):
        return {
            'escaped'  : sum(self.statuses == Status.ESCAPED),
            'exiting'  : sum(self.statuses == Status.EXITING),
            'following': sum(self.statuses == Status.FOLLOWER),
            'viscek'   : sum(self.statuses == Status.VISCEK),
        }

class Agent:
    start_position : np.ndarray                 # [2], r0_x and r0_y (same each reset)
    start_direction : np.ndarray                # [2], v0_x and v0_x (same each reset)
    position : np.ndarray                       # [2], r_x and r_y
    direction : np.ndarray                      # [2], v_x and v_y
    memory : Dict[str, List[np.ndarray]]        # needed for animation drawing (position)
    enslaving_degree: float                     # 0 < enslaving_degree <= 1

    def __init__(self, enslaving_degree):
        self.start_position = np.zeros(2, dtype=np.float32)
        self.start_direction = np.zeros(2, dtype=np.float32)
        self.enslaving_degree = enslaving_degree
        
    def reset(self):
        self.position = self.start_position.copy()
        self.direction = self.start_position.copy()
        self.memory = {'position' : []}

    def save(self):
        self.memory['position'].append(self.position.copy())


class Exit:
    position : np.ndarray
    def __init__(self):
        self.position = np.array([0, -1], dtype=np.float32)


class Time:
    def __init__(self, 
        max_timesteps : int = constants.MAX_TIMESTEPS,
        n_episodes : int = constants.N_EPISODES,
        n_timesteps : int = constants.N_TIMESTEPS
        ) -> None:
        self.now = 0
        self.max_timesteps = max_timesteps
        self.n_episodes = n_episodes
        self.overall_timesteps = n_timesteps

    def reset(self):
        self.now = 0
        self.n_episodes += 1

    def step(self):
        self.now += 1
        self.overall_timesteps += 1
        return self.truncated()
        
    def truncated(self):
        return self.now >= self.max_timesteps 


def grad_potential_pedestrians(
        agent: Agent, pedestrians: Pedestrians, alpha: float = constants.ALPHA
    ) -> np.ndarray:
    R = agent.position[np.newaxis, :] - pedestrians.positions
    R = R[pedestrians.statuses == Status.VISCEK]

    if len(R) != 0:
        norm = np.linalg.norm(R, axis = 1)[:, np.newaxis] + constants.EPS
        grad = - alpha / norm ** (alpha + 2) * R
        grad = grad.sum(axis = 0)
    else:
        grad = np.zeros(2)
    return grad


def grad_potential_exit(
        agent: Agent, pedestrians: Pedestrians, exit: Exit, alpha: float = constants.ALPHA
    ) -> np.ndarray:
    R = agent.position - exit.position
    norm = np.linalg.norm(R) + constants.EPS
    grad = - alpha / norm ** (alpha + 2) * R
    grad *= sum(pedestrians.statuses == Status.FOLLOWER)
    return grad

def grad_time_derivative_pedestrians(
        agent: Agent, pedestrians: Pedestrians, alpha: float = constants.ALPHA
    ) -> np.ndarray:

    R = agent.position[np.newaxis, :] - pedestrians.positions
    R = R[pedestrians.statuses == Status.VISCEK]

    V = agent.direction[np.newaxis, :] - pedestrians.positions
    V = V[pedestrians.statuses == Status.VISCEK]

    if len(R) != 0:
        norm = np.linalg.norm(R, axis = 1)[:, np.newaxis] + constants.EPS
        grad = - alpha / norm ** (alpha + 4) * (V * norm**2 - (alpha + 2) * np.sum(V * R, axis=1, keepdims=True) * R)
        grad = grad.sum(axis=0)
    else:
        grad = np.zeros(2)
        
    return grad
def grad_time_derivative_exit(
        agent: Agent, pedestrians: Pedestrians, exit: Exit, alpha: float = constants.ALPHA
    ) -> np.ndarray:

    R = agent.position - exit.position

    V = agent.direction
    
    N = sum(pedestrians.statuses == Status.FOLLOWER)

    if N != 0:
        norm = np.linalg.norm(R) + constants.EPS
        grad = - alpha / norm ** (alpha + 4) * (V * norm**2 - (alpha + 2) * np.dot(V, R) * R)
        grad *= N
    else:
        grad = np.zeros(2)
        
    return grad

#from src.utils import Exit, Status, SwitchDistances, update_statuses moved in this file
from numpy.linalg import solve, LinAlgError
import numpy as np

def do_intersect(p1, q1, p2, q2):
    # Convert points to linear equations of the form Ax + By = C
    A1, B1 = q1[1] - p1[1], p1[0] - q1[0]  # A = y2 - y1, B = x1 - x2 for line 1
    C1 = A1 * p1[0] + B1 * p1[1]
    A2, B2 = q2[1] - p2[1], p2[0] - q2[0]  # A = y2 - y1, B = x1 - x2 for line 2
    C2 = A2 * p2[0] + B2 * p2[1]
    
    matrix = np.array([[A1, B1], [A2, B2]])
    constants = np.array([C1, C2])
    
    try:
        # Solve the linear equations system to find the intersection point
        solution = solve(matrix, constants)
        
        # Check if the solution (intersection point) lies within the bounds of both line segments
        if (min(p1[0], q1[0]) <= solution[0] <= max(p1[0], q1[0]) and
            min(p1[1], q1[1]) <= solution[1] <= max(p1[1], q1[1]) and
            min(p2[0], q2[0]) <= solution[0] <= max(p2[0], q2[0]) and
            min(p2[1], q2[1]) <= solution[1] <= max(p2[1], q2[1])):
            return True  # Segments intersect
        else:
            return False  # Segments do not intersect
    except LinAlgError:
        # The lines are parallel or coincident (no single intersection point)
        return False
def calculate_detour(agent_pos, pedestrian_pos, wall):
    # Calculate distances for detours via wall[0] and wall[1]
    detour_via_wall0 = np.linalg.norm(agent_pos - wall[0]) + np.linalg.norm(wall[0] - pedestrian_pos)
    detour_via_wall1 = np.linalg.norm(agent_pos - wall[1]) + np.linalg.norm(wall[1] - pedestrian_pos)
    
    # Choose the shortest path
    if detour_via_wall0 < detour_via_wall1:
        chosen_wall_corner = wall[0]
        detour_distance = detour_via_wall0
    else:
        chosen_wall_corner = wall[1]
        detour_distance = detour_via_wall1
    
    # Construct the detour vector
    # First, find the direction vector from agent to the chosen corner and normalize it
    direction_to_corner = chosen_wall_corner - agent_pos
    direction_to_corner_normalized = direction_to_corner / np.linalg.norm(direction_to_corner)
    
    # Then, create the detour vector with the correct magnitude (total detour distance)
    detour_vector = direction_to_corner_normalized * detour_distance
    
    return detour_vector
#def check_horizonthal_bumping(positions, old_pos, num_openings, opening_positions, doors):
def check_horizonthal_bumping(positions, old_pos, num_openings, opening_positions):
    to_bump_mask = positions[:, 1] * old_pos[:, 1] < 0

    for count, opening_position in enumerate(opening_positions):
        to_bump_mask = np.logical_and(
            to_bump_mask,
            #np.abs(positions[:, 0] - opening_position) > constants.WALL_HOLE_HALF_WIDTH - doors[[count]]
            np.abs(positions[:, 0] - opening_position) > constants.WALL_HOLE_HALF_WIDTH
            #np.abs(positions[:, 0] - opening_position) > constants.WALL_HOLE_HALF_WIDTH * (1 - (action[[2+count]] + 1))/2   
        )

    return to_bump_mask
    
def check_vertical_bumping(positions, old_pos, num_openings, opening_positions):
#def check_vertical_bumping(positions, old_pos, num_openings, opening_positions, doors):
    to_bump_mask = np.logical_and(
    positions[:, 0] * old_pos[:, 0] < 0,
    old_pos[:, 1] > constants.VERTICAL_WALL_POSITION 
    )

    for count, opening_position in enumerate(opening_positions):
        to_bump_mask = np.logical_and(
            to_bump_mask,
            #np.abs(positions[:, 1] - opening_position) > constants.VERTICAL_WALL_HALF_WIDTH - doors[[2]] #remove hardcoded 2 €
            np.abs(positions[:, 1] - opening_position) > constants.VERTICAL_WALL_HALF_WIDTH
            #np.abs(positions[:, 1] - opening_position) > constants.VERTICAL_WALL_HALF_WIDTH* (1 - (action[[-1]] + 1))/2   
        )

    return to_bump_mask
def check_if_same_room(a, b, walls=constants.WALLS):
    m, n = len(a), len(b)
    # Initialize the mask with True values
    mask = np.ones((m, n), dtype=bool)  
    # Update the mask to False where segments intersect with any wall
    for i in range(m):
        for j in range(n):
            segment_start = a[i]
            segment_end = b[j]
            for wall in walls:
                if do_intersect(segment_start, segment_end, wall[0], wall[1]):
                            mask[i, j] = False
                break  # No need to check other walls if one intersection is found            
    return mask  
class Area:
    def __init__(self, 
        reward: Reward,
        width = constants.WIDTH, 
        height = constants.HEIGHT,
        step_size = constants.STEP_SIZE,
        noise_coef = constants.NOISE_COEF,
        ):
        self.reward = reward
        self.width = width
        self.height = height
        self.step_size = step_size
        self.noise_coef = noise_coef
        self.exit = Exit()
    
    def reset(self):
        pass

    def pedestrians_step(self, pedestrians : Pedestrians, agent : Agent, now : int) -> Tuple[Pedestrians, bool, float, float]:

        # Check evacuated pedestrians & record new directions and positions of escaped pedestrians
        escaped = pedestrians.statuses == Status.ESCAPED
        pedestrians.directions[escaped] = 0
        pedestrians.positions[escaped] = self.exit.position
        
        # Check exiting pedestrians & record new directions for exiting pedestrians
        exiting = pedestrians.statuses == Status.EXITING
        if any(exiting):
            vec2exit = self.exit.position - pedestrians.positions[exiting]
            len2exit = np.linalg.norm(vec2exit, axis=1)
            vec_size = np.minimum(len2exit, self.step_size)
            vec2exit = (vec2exit.T / len2exit * vec_size).T
            pedestrians.directions[exiting] = vec2exit

        # Check following pedestrians & record new directions for following pedestrians
        following = pedestrians.statuses == Status.FOLLOWER

        # Check viscek pedestrians
        viscek = pedestrians.statuses == Status.VISCEK
        
        # Use all moving particles (efv -- exiting, following, viscek) to estimate the movement of viscek particles
        efv = reduce(np.logical_or, (exiting, following, viscek))
        efv_directions = pedestrians.directions[efv]
        efv_directions = (efv_directions.T / np.linalg.norm(efv_directions, axis=1)).T 
        
        # Find neighbours between following and viscek (fv) particles and all other moving particles
        fv = reduce(np.logical_or, (following, viscek))
        dm = distance_matrix(pedestrians.positions[fv],
                             pedestrians.positions[efv], 2)
        sr = check_if_same_room(pedestrians.positions[fv],pedestrians.positions[efv]) 
        intersection = np.where(dm < SwitchDistances.to_pedestrian, 1, 0) 
        intersection = np.logical_and(intersection, sr)


        #n_intersections = np.maximum(1, intersection.sum(axis=1))

        '''def estimate_mean_direction_among_neighbours(
                intersection,           # [f+v, f+v+e]  boolean matrix
                efv_directions,         # [f+v+e, 2]    vectors of directions of pedestrians
                n_intersections         # [f+v]         amount of neighbouring pedestrians
            ):
            """Viscek model"""
        
            # Estimate the contibution if each neighbouring particle 
            fv_directions_x = (intersection * efv_directions[:, 0]).sum(axis=1) / n_intersections
            fv_directions_y = (intersection * efv_directions[:, 1]).sum(axis=1) / n_intersections
            fv_theta = np.arctan2(fv_directions_y, fv_directions_x)
                                    
            # Create randomization noise to obtained directions
            # noise = np.random.normal(loc=0., scale=constants.NOISE_COEF, size=len(n_intersections))
            noise = np.random.uniform(low=-self.noise_coef/2, high=self.noise_coef/2, size=len(n_intersections))
            
            # New direction = estimated_direction + noise
            fv_theta = fv_theta + noise
            #print('la stime dentro predestrian step')
            return np.vstack((np.cos(fv_theta), np.sin(fv_theta)))'''
        def estimate_mean_direction_among_neighbours(
                intersection,           # [f+v, f+v+e]  boolean matrix
                efv_directions,        # [f+v+e, 2]    vectors of directions of pedestrians
                # n_intersections         # [f+v]         amount of neighbouring pedestrians
            ):
            """Viscek model"""
        
            n_intersections = np.maximum(1, intersection.sum(axis=1))

            # Estimate the contibution if each neighbouring particle 
            fv_directions_x = (intersection * efv_directions[:, 0]).sum(axis=1) / n_intersections
            fv_directions_y = (intersection * efv_directions[:, 1]).sum(axis=1) / n_intersections
            fv_theta = np.arctan2(fv_directions_y, fv_directions_x)
                                    
            # Create randomization noise to obtained directions
            noise = np.random.uniform(low=-constants.NOISE_COEF/2, high=constants.NOISE_COEF/2, size=len(n_intersections))
            # noise = np.random.normal(loc=0., scale=constants.NOISE_COEF, size=len(n_intersections))
            
            # New direction = estimated_direction + noise
            fv_theta = fv_theta + noise
            #print('la stima fuori pedestrian step')
            return np.vstack((np.cos(fv_theta), np.sin(fv_theta)))
                #€ add mask for walls
        fv_directions = self.estimate_mean_direction_among_neighbours(
            intersection, efv_directions#, n_intersections
        )            

        # Record new directions of following and viscek pedestrians
        pedestrians.directions[fv] = fv_directions.T * self.step_size
        
        # Add enslaving factor of leader's direction to following particles
        f_directions = pedestrians.directions[following]
        #f_positions = pedestrians.positions[following]
        l_directions = agent.direction
        f_directions = agent.enslaving_degree * l_directions + (1. - agent.enslaving_degree) * f_directions
        pedestrians.directions[following] = f_directions
        
        # Record new positions of exiting, following and viscek pedestrians
        old_pos = pedestrians.positions.copy()
        pedestrians.positions[efv] += pedestrians.directions[efv] 
        # Handling of wall collisions
        clipped = np.clip(pedestrians.positions, 
                    [-self.width, -self.height], [self.width, self.height])
        miss = pedestrians.positions - clipped
        pedestrians.positions -= 2 * miss
        pedestrians.directions *= np.where(miss!=0, -1, 1)
        # horizhontal wall bumping HERE€
        # Define the positions of the two openings
        opening_positions = [-1, 1]
        num_openings=len(opening_positions)
        #€to_bump_mask1 = pedestrians.positions[:, 1] * old_pos[:, 1] < 0
        # #to_bump_mask = np.logical_and(to_bump_mask, np.abs(pedestrians.positions[:,0]) > constants.WALL_HOLE_HALF_WIDTH)
        # for opening_position in opening_positions:
            # to_bump_mask = np.logical_and(
                # to_bump_mask,
                # np.abs(pedestrians.positions[:, 0] - opening_position) > constants.WALL_HOLE_HALF_WIDTH
            # )
        # to_bump_mask = np.logical_and(to_bump_mask, efv)
        # Calculate the to_bump_mask using check_bumping
        #to_bump1_mask = check_horizonthal_bumping(pedestrians.positions, old_pos, len(opening_positions), opening_positions, self.doors)
        to_bump1_mask = check_horizonthal_bumping(pedestrians.positions, old_pos, len(opening_positions), opening_positions)

        if any(to_bump1_mask):
            pedestrians.positions[to_bump1_mask] = old_pos[to_bump1_mask]
            pedestrians.directions[to_bump1_mask, 1] = -pedestrians.directions[to_bump1_mask, 1]
        # vertical wall bumping HERE€
        opening_positions = [ 0.5]
        num_openings=len(opening_positions)
        #to_bump0_mask = np.logical_and(pedestrians.positions[:, 0] * old_pos[:, 0] < 0, old_pos[:, 1] > 0)
        #to_bump0_mask = check_vertical_bumping(pedestrians.positions, old_pos, len(opening_positions), opening_positions, self.doors)
        to_bump0_mask = check_vertical_bumping(pedestrians.positions, old_pos, len(opening_positions), opening_positions)

        if any(to_bump0_mask):
            pedestrians.positions[to_bump0_mask] = old_pos[to_bump0_mask]
            pedestrians.directions[to_bump0_mask, 0] = -pedestrians.directions[to_bump0_mask, 0]        
        # Estimate pedestrians statues, reward & update statuses
        old_statuses = pedestrians.statuses.copy()
        new_pedestrians_statuses = update_statuses(
            statuses=pedestrians.statuses,
            pedestrian_positions=pedestrians.positions,
            agent_position=agent.position,
            exit_position=self.exit.position
        )
        reward_pedestrians = self.reward.estimate_status_reward(
            old_statuses=old_statuses,
            new_statuses=new_pedestrians_statuses,
            timesteps=now,
            num_pedestrians=pedestrians.num
        )
        intrinsic_reward = self.reward.estimate_intrinsic_reward(
            pedestrians_positions=pedestrians.positions,
            exit_position=self.exit.position
        )
        pedestrians.statuses = new_pedestrians_statuses
        
        # Termination due to all pedestrians escaped
        if sum(pedestrians.statuses == Status.ESCAPED) == pedestrians.num:
            termination = True
        else: 
            termination = False

        return pedestrians, termination, reward_pedestrians, intrinsic_reward

    def agent_step(self, action : list, agent : Agent) -> Tuple[Agent, bool, float]:
        """
        Perform agent step:
            1. Read & preprocess action
            2. Check wall collision
            3. Return (updated agent, termination, reward)
        """
        action = np.array(action)
        action /= np.linalg.norm(action) + constants.EPS # np.clip(action, -1, 1, out=action)

        agent.direction = self.step_size * action
        
        if not self._if_wall_collision(agent):
            agent.position += agent.direction
            return agent, False, 0.
        else:
            return agent, self.reward.is_termination_agent_wall_collision, -5.

    def estimate_mean_direction_among_neighbours(self,
            intersection,           # [f+v, f+v+e]  boolean matrix
            efv_directions,        # [f+v+e, 2]    vectors of directions of pedestrians
            # n_intersections         # [f+v]         amount of neighbouring pedestrians
        ):
        """Viscek model"""
    
        n_intersections = np.maximum(1, intersection.sum(axis=1))

        # Estimate the contibution if each neighbouring particle 
        fv_directions_x = (intersection * efv_directions[:, 0]).sum(axis=1) / n_intersections
        fv_directions_y = (intersection * efv_directions[:, 1]).sum(axis=1) / n_intersections
        fv_theta = np.arctan2(fv_directions_y, fv_directions_x)
                                
        # Create randomization noise to obtained directions
        noise = np.random.uniform(low=-constants.NOISE_COEF/2, high=constants.NOISE_COEF/2, size=len(n_intersections))
        # noise = np.random.normal(loc=0., scale=constants.NOISE_COEF, size=len(n_intersections))
        
        # New direction = estimated_direction + noise
        fv_theta = fv_theta + noise
        #print('la stima fuori pedestrian step')
        return np.vstack((np.cos(fv_theta), np.sin(fv_theta)))
    def escaped_directions_update(self, pedestrians: Pedestrians, escaped) -> None:
        pedestrians.directions[escaped] = 0
        pedestrians.positions[escaped] = self.exit.position
    def exiting_directions_update(self, pedestrians: Pedestrians, exiting) -> None:
        if any(exiting):
            vec2exit = self.exit.position - pedestrians.positions[exiting]
            len2exit = np.linalg.norm(vec2exit, axis=1)
            vec_size = np.minimum(len2exit, self.step_size)
            vec2exit = (vec2exit.T / len2exit * vec_size).T
            pedestrians.directions[exiting] = vec2exit
    '''def pedestrians_step(self, pedestrians : Pedestrians, agent : Agent, now : int) -> Tuple[Pedestrians, bool, float, float]:
        def check_if_same_room(a, b, walls=constants.WALLS):
            #print('a')
            #print(a)
            #print('b')
            #print(b)

            m, n = len(a), len(b)
            #print('m')
            #print(m)
            #print('n')
            #print(n)

            # Initialize the mask with True values
            mask = np.ones((m, n), dtype=bool)
            
            # Update the mask to False where segments intersect with any wall
            for i in range(m):
                for j in range(n):
                    segment_start = a[i]
                    segment_end = b[j]
                    for wall in walls:
                        if do_intersect(segment_start, segment_end, wall[0], wall[1]):
                            mask[i, j] = False
                            break  # No need to check other walls if one intersection is found
            
            return mask                    
        # print(np.any(pedestrians.statuses == Status.FALLEN))

        # Check evacuated pedestrians & record new directions and positions of escaped pedestrians
        escaped = pedestrians.statuses == Status.ESCAPED
        self.escaped_directions_update(pedestrians, escaped)
        
        # Check exiting pedestrians & record new directions for exiting pedestrians
        exiting = pedestrians.statuses == Status.EXITING
        self.exiting_directions_update(pedestrians, exiting)

        # Check following pedestrians & record new directions for following pedestrians
        following = pedestrians.statuses == Status.FOLLOWER

        # Check viscek pedestrians
        viscek = pedestrians.statuses == Status.VISCEK
        
        # Use all moving particles (efv -- exiting, following, viscek) to estimate the movement of viscek particles
        efv = reduce(np.logical_or, (exiting, following, viscek))
        efv_directions = pedestrians.directions[efv]
        efv_directions = (efv_directions.T / np.linalg.norm(efv_directions, axis=1)).T  # TODO: try using keepdims for simplisity
        
        # Find neighbours between following and viscek (fv) particles and all other moving particles
        fv = reduce(np.logical_or, (following, viscek))
        dm = distance_matrix(pedestrians.positions[fv],
                             pedestrians.positions[efv], 2)
        sr = check_if_same_room(pedestrians.positions[fv],pedestrians.positions[efv]) 

        intersection = np.where(dm < SwitchDistances.to_pedestrian, 1, 0) 
        intersection = np.logical_and(intersection, sr)
        #€ add mask for walls
        print('intersection')
        fv_directions = self.estimate_mean_direction_among_neighbours(intersection, efv_directions)     

        # Record new directions of following and viscek pedestrians
        pedestrians.directions[fv] = fv_directions.T * self.step_size
        
        # Add enslaving factor of leader's direction to following particles
        f_directions = pedestrians.directions[following]
        f_positions = pedestrians.positions[following]
        l_directions = agent.direction
        # l_directions = agent.position.reshape(1, -1) - f_positions
        # l_directions /=  np.linalg.norm(l_directions, axis=1, keepdims=True) / self.step_size
        f_directions = agent.enslaving_degree * l_directions + (1. - agent.enslaving_degree) * f_directions
        pedestrians.directions[following] = f_directions
        
        # to_fall = np.where(dm < SwitchDistances.to_fall, 1, 0).any(axis=0)
        # to_fall = np.flatnonzero(efv)[to_fall]
        # pedestrians.directions[to_fall] *= 0
        # pedestrians.statuses[to_fall] = Status.FALLEN
        # # print(np.any(pedestrians.statuses == Status.FALLEN))
        
        # Record new positions of exiting, following and viscek pedestrians
        old_pos = pedestrians.positions.copy()
        pedestrians.positions[efv] += pedestrians.directions[efv] 

        # Handling of wall collisions
        clipped = np.clip(pedestrians.positions, 
                    [-self.width, -self.height], [self.width, self.height])
        miss = pedestrians.positions - clipped
        pedestrians.positions -= 2 * miss
        pedestrians.directions *= np.where(miss!=0, -1, 1)

        # horizhontal wall bumping HERE€
        # Define the positions of the two openings
        opening_positions = [-1, 1]
        num_openings=len(opening_positions)
        #€to_bump_mask1 = pedestrians.positions[:, 1] * old_pos[:, 1] < 0
        # #to_bump_mask = np.logical_and(to_bump_mask, np.abs(pedestrians.positions[:,0]) > constants.WALL_HOLE_HALF_WIDTH)
        # for opening_position in opening_positions:
            # to_bump_mask = np.logical_and(
                # to_bump_mask,
                # np.abs(pedestrians.positions[:, 0] - opening_position) > constants.WALL_HOLE_HALF_WIDTH
            # )
        # to_bump_mask = np.logical_and(to_bump_mask, efv)
        # Calculate the to_bump_mask using check_bumping
        #to_bump1_mask = check_horizonthal_bumping(pedestrians.positions, old_pos, len(opening_positions), opening_positions, self.doors)
        to_bump1_mask = check_horizonthal_bumping(pedestrians.positions, old_pos, len(opening_positions), opening_positions)

        if any(to_bump1_mask):
            pedestrians.positions[to_bump1_mask] = old_pos[to_bump1_mask]
            pedestrians.directions[to_bump1_mask, 1] = -pedestrians.directions[to_bump1_mask, 1]
        # vertical wall bumping HERE€
        opening_positions = [ 0.5]
        num_openings=len(opening_positions)
        #to_bump0_mask = np.logical_and(pedestrians.positions[:, 0] * old_pos[:, 0] < 0, old_pos[:, 1] > 0)
        #to_bump0_mask = check_vertical_bumping(pedestrians.positions, old_pos, len(opening_positions), opening_positions, self.doors)
        to_bump0_mask = check_vertical_bumping(pedestrians.positions, old_pos, len(opening_positions), opening_positions)

        if any(to_bump0_mask):
            pedestrians.positions[to_bump0_mask] = old_pos[to_bump0_mask]
            pedestrians.directions[to_bump0_mask, 0] = -pedestrians.directions[to_bump0_mask, 0]        
        
        # Estimate pedestrians statues, reward & update statuses
        old_statuses = pedestrians.statuses.copy()
        new_pedestrians_statuses = update_statuses(
            statuses=pedestrians.statuses,
            pedestrian_positions=pedestrians.positions,
            agent_position=agent.position,
            exit_position=self.exit.position
        )
        reward_pedestrians = self.reward.estimate_status_reward(
            old_statuses=old_statuses,
            new_statuses=new_pedestrians_statuses,
            timesteps=now,
            num_pedestrians=pedestrians.num
        )
        intrinsic_reward = self.reward.estimate_intrinsic_reward(
            pedestrians_positions=pedestrians.positions,
            exit_position=self.exit.position
        )
        pedestrians.statuses = new_pedestrians_statuses
        
        # Termination due to all pedestrians escaped
        if sum(pedestrians.statuses == Status.ESCAPED) == pedestrians.num: # TODO: np.all(pedestrians.statuses == Status.ESCAPED) ?
            termination = True
        else: 
            termination = False

        return pedestrians, termination, reward_pedestrians, intrinsic_reward'''

    '''def agent_step(self, action : list, agent : Agent) -> Tuple[Agent, bool, float]:
        """
        Perform agent step:
            1. Read & preprocess action
            2. Check wall collision
            3. Return (updated agent, termination, reward)
        """
        #m_action= action[[0,1]] #€
        #d_action = [action[[2,3,4]]] #shit to remove doors
        #print(type(m_action))
        #print(m_action)
        #d_action = np.array([-1,-1,-1]) #shit to remove doors
        #print(type(d_action))
        #print(d_action)
        #self.doors = (d_action + 1)* constants.WALL_HOLE_HALF_WIDTH / 2 
        #print(self.doors)
        #print()
        #agent.doors = self.doors
        #m_action = np.array(m_action)
        #m_action /= np.linalg.norm(m_action) + constants.EPS # np.clip(action, -1, 1, out=action)
        action = np.array(action)
        #action /= (np.linalg.norm(action) + constants.EPS) # np.clip(action, -1, 1, out=action)
        #agent.direction = self.step_size * m_action #
        agent.direction = self.step_size * action /(np.linalg.norm(action) + constants.EPS)
        h_opening_positions = [-1, 1]
        num_h_openings=len(h_opening_positions)
        v_opening_positions = [ 0.5]
        num_v_openings=len(v_opening_positions)
        def agent_median_wall_bump(pos, dire): #HERE€
            new_pos = pos + dire
            #if ((new_pos[1] * pos[1]) < 0) and (abs(new_pos[0]) > constants.WALL_HOLE_HALF_WIDTH):
                #return True
            #if check_horizonthal_bumping(np.expand_dims(new_pos, axis=0), np.expand_dims(pos, axis=0), num_h_openings, h_opening_positions, self.doors):
            if check_horizonthal_bumping(np.expand_dims(new_pos, axis=0), np.expand_dims(pos, axis=0), num_h_openings, h_opening_positions):
                return True
            #if check_vertical_bumping(np.expand_dims(new_pos, axis=0), np.expand_dims(pos, axis=0), num_v_openings, v_opening_positions, self.doors):
            if check_vertical_bumping(np.expand_dims(new_pos, axis=0), np.expand_dims(pos, axis=0), num_v_openings, v_opening_positions):
                print('vertical bumping')
                return True
        if agent_median_wall_bump(agent.position, agent.direction):
            #agent.position -= agent.direction
            return agent, self.reward.is_termination_agent_wall_collision, -500.
        
        if not self._if_wall_collision(agent):
            agent.position += agent.direction
            return agent, False, 0.
        else:
            #agent.position -= agent.direction
            return agent, self.reward.is_termination_agent_wall_collision, -500.'''

    def _if_wall_collision(self, agent : Agent): #change entirely this stuff €€€
        pt = agent.position + agent.direction

        left  = pt[0] < -self.width
        right = pt[0] > self.width
        down  = pt[1] < -self.height  
        up    = pt[1] > self.height
        
        if left or right or down or up:
            return True
        return False



class EvacuationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    """
    Evacuation Game Enviroment for Gym
    Continious Action and Observation Space.
    """
    def __init__(self, 
        experiment_name='test',
        number_of_pedestrians=constants.NUM_PEDESTRIANS,
        
        # leader params
        enslaving_degree=constants.ENSLAVING_DEGREE,
        
        # area params
        width=constants.WIDTH,
        height=constants.HEIGHT,
        step_size=constants.STEP_SIZE,
        noise_coef=constants.NOISE_COEF,
        
        # reward params
        is_termination_agent_wall_collision=constants.TERMINATION_AGENT_WALL_COLLISION,
        is_new_exiting_reward=constants.IS_NEW_EXITING_REWARD,
        is_new_followers_reward=constants.IS_NEW_FOLLOWERS_REWARD,
        intrinsic_reward_coef=constants.INTRINSIC_REWARD_COEF,
        init_reward_each_step=constants.INIT_REWARD_EACH_STEP, 
        
        # time params
        max_timesteps=constants.MAX_TIMESTEPS,
        n_episodes=constants.N_EPISODES,
        n_timesteps=constants.N_TIMESTEPS,
        
        # gravity embedding params
        enabled_gravity_embedding=constants.ENABLED_GRAVITY_EMBEDDING,
        enabled_gravity_and_speed_embedding = constants.ENABLED_GRAVITY_AND_SPEED_EMBEDDING,
        alpha=constants.ALPHA,
        
        # logging params
        verbose=False,
        render_mode=None,
        draw=False
        
        ) -> None:
        super(EvacuationEnv, self).__init__()
        
        # setup env
        self.pedestrians = Pedestrians(num=number_of_pedestrians)
        
        reward = Reward(
            is_new_exiting_reward=is_new_exiting_reward,
            is_new_followers_reward=is_new_followers_reward,
            is_termination_agent_wall_collision=is_termination_agent_wall_collision,
            init_reward_each_step=init_reward_each_step)        
        
        self.area = Area(
            reward=reward, 
            width=width, height=height, 
            step_size=step_size, noise_coef=noise_coef)
        
        self.time = Time(
            max_timesteps=max_timesteps, 
            n_episodes=n_episodes, n_timesteps=n_timesteps)
        
        # setup agent
        self.agent = Agent(enslaving_degree=enslaving_degree)
        
        self.intrinsic_reward_coef = intrinsic_reward_coef
        self.episode_reward = 0
        self.episode_intrinsic_reward = 0
        self.episode_status_reward = 0        
        self.enabled_gravity_embedding = enabled_gravity_embedding
        self.enabled_gravity_and_speed_embedding = enabled_gravity_and_speed_embedding
        self.alpha = alpha

        self.action_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.observation_space = self._get_observation_space()
        
        # logging
        self.render_mode = render_mode
        self.experiment_name = experiment_name
        setup_logging(verbose, experiment_name)

        # drawing
        self.draw = draw
        self.save_next_episode_anim = False
        log.info(f'Env {self.experiment_name} is initialized.')        
        
    def _get_observation_space(self):        
        observation_space = {
            'agent_position' : spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
        }
        if self.enabled_gravity_and_speed_embedding:
            observation_space['grad_potential_pedestrians'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
            observation_space['grad_potential_exit'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)  
            observation_space['grad_time_derivative_exit'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
            observation_space['grad_time_derivative_pedestrians'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)         
        elif self.enabled_gravity_embedding:
            observation_space['grad_potential_pedestrians'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
            observation_space['grad_potential_exit'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
    
        else:
            observation_space['pedestrians_positions'] = \
                spaces.Box(low=-1, high=1, shape=(self.pedestrians.num, 2), dtype=np.float32)
            observation_space['exit_position'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)

        return spaces.Dict(observation_space)

    def _get_observation(self):
        observation = {}
        observation['agent_position'] = self.agent.position
        if self.enabled_gravity_and_speed_embedding:
            observation['grad_potential_pedestrians'] = grad_potential_pedestrians(
                agent=self.agent, 
                pedestrians=self.pedestrians, 
                alpha=self.alpha
            )
            observation['grad_potential_exit'] = grad_potential_exit(
                agent=self.agent,
                pedestrians=self.pedestrians,
                exit=self.area.exit,
                alpha=self.alpha
            )
            observation['grad_time_derivative_exit'] = grad_time_derivative_exit(
                agent=self.agent,
                pedestrians=self.pedestrians,
                exit=self.area.exit,
                alpha=self.alpha
            )
            observation['grad_time_derivative_pedestrians'] = grad_time_derivative_pedestrians(
                agent=self.agent,
                pedestrians=self.pedestrians,
                alpha=self.alpha
            )        
        elif self.enabled_gravity_embedding:
            observation['grad_potential_pedestrians'] = grad_potential_pedestrians(
                agent=self.agent, 
                pedestrians=self.pedestrians, 
                alpha=self.alpha
            )
            observation['grad_potential_exit'] = grad_potential_exit(
                agent=self.agent,
                pedestrians=self.pedestrians,
                exit=self.area.exit,
                alpha=self.alpha
            )
        else:
            observation['pedestrians_positions'] = self.pedestrians.positions
            observation['exit_position'] = self.area.exit.position

        return observation

    def reset(self, seed=None, options=None):
        if self.save_next_episode_anim or (self.time.n_episodes + 1) % WALK_DIAGRAM_LOGGING_FREQUENCY == 0:
            self.draw = True
            self.save_next_episode_anim = True

        if self.time.n_episodes > 0:
            logging_dict = {
                "episode_intrinsic_reward" : self.episode_intrinsic_reward,
                "episode_status_reward" : self.episode_status_reward,
                "episode_reward" : self.episode_reward,
                "episode_length" : self.time.now,
                "escaped_pedestrians" : sum(self.pedestrians.statuses == Status.ESCAPED),
                "exiting_pedestrians" : sum(self.pedestrians.statuses == Status.EXITING),
                "following_pedestrians" : sum(self.pedestrians.statuses == Status.FOLLOWER),
                "viscek_pedestrians" : sum(self.pedestrians.statuses == Status.VISCEK),
                "overall_timesteps" : self.time.overall_timesteps
            }
            log.info('\t'.join([f'{key}={value}' for key, value in logging_dict.items()]))
            wandb.log(logging_dict)
        
        self.episode_reward = 0
        self.episode_intrinsic_reward = 0
        self.episode_status_reward = 0 
        self.time.reset()
        self.area.reset()
        self.agent.reset()
        self.pedestrians.reset(agent_position=self.agent.position,
                               exit_position=self.area.exit.position)
        self.pedestrians.save()
        log.info('Env is reseted.')
        return self._get_observation(), {}

    def step(self, action: list):
        # Increment time
        truncated = self.time.step()

        # Agent step
        self.agent, terminated_agent, reward_agent = self.area.agent_step(action, self.agent)
        
        # Pedestrians step
        self.pedestrians, terminated_pedestrians, reward_pedestrians, intrinsic_reward = \
            self.area.pedestrians_step(self.pedestrians, self.agent, self.time.now)
        
        # Save positions for rendering an animation
        if self.draw:
            self.pedestrians.save()
            self.agent.save()

        # Collect rewards
        reward = reward_agent + reward_pedestrians + self.intrinsic_reward_coef * intrinsic_reward
        
        # Record observation
        observation = self._get_observation()
        
        # Draw animation
        if (terminated_agent or terminated_pedestrians or truncated) and self.draw:
            self.save_animation()

        # Log reward
        self.episode_reward += reward
        self.episode_intrinsic_reward += intrinsic_reward
        self.episode_status_reward += reward_agent + reward_pedestrians
        return observation, reward, terminated_agent or terminated_pedestrians, truncated, {}

    def render(self):
        fig, ax = plt.subplots(figsize=(5, 5))

        exit_coordinates = (self.area.exit.position[0], self.area.exit.position[1])
        agent_coordinates = (self.agent.position[0], self.agent.position[1])

        # Draw exiting zone
        exiting_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_exit, 
            0, 180, alpha=0.2, color='green'
        )
        ax.add_patch(exiting_zone)

        # Draw escaping zone
        escaping_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_escape, 
            0, 180, color='white'
        )
        ax.add_patch(escaping_zone)
        
        # Draw exit
        ax.plot(exit_coordinates[0], exit_coordinates[1], marker='X', color='green')

        # Draw following zone
        following_zone = mpatches.Wedge(
            agent_coordinates, 
            SwitchDistances.to_leader, 
            0, 360, alpha=0.1, color='blue'
        )
        ax.add_patch(following_zone)

        from itertools import cycle
        colors = cycle([item['color'] for item in ax._get_lines._cycler_items])
        
        # Draw pedestrians
        for status in Status.all():
            selected_pedestrians = self.pedestrians.statuses == status
            color = next(colors)
            # color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(self.pedestrians.positions[selected_pedestrians, 0], 
                    self.pedestrians.positions[selected_pedestrians, 1],
                lw=0, marker='.', color=color)
            # for i in range(self.pedestrians.directions.shape[0]):
            #     ax.plot(self.pedestrians.positions[i],
            #     self.pedestrians.positions[i] + self.pedestrians.directions[i])

        # Draw agent
        ax.plot(agent_coordinates[0], agent_coordinates[1], marker='+', color='red')

        plt.xlim([ -1.1 * self.area.width, 1.1 * self.area.width])
        plt.ylim([ -1.1 * self.area.height, 1.1 * self.area.height])
        plt.xticks([]); plt.yticks([])
        plt.hlines([self.area.height, -self.area.height], 
            -self.area.width, self.area.width, linestyle='--', color='grey')
        plt.vlines([self.area.width, -self.area.width], 
            -self.area.height, self.area.height, linestyle='--', color='grey')

        plt.title(f"{self.experiment_name}. Timesteps: {self.time.now}")

        plt.tight_layout()
        if not os.path.exists(constants.SAVE_PATH_PNG): os.makedirs(constants.SAVE_PATH_PNG)
        filename = os.path.join(constants.SAVE_PATH_PNG, f'{self.experiment_name}_{self.time.now}.png')
        plt.savefig(filename)
        plt.show()
        log.info(f"Env is rendered and pnd image is saved to {filename}")

    def save_animation(self):
        
        fig, ax = plt.subplots(figsize=(5, 5))

        plt.title(f"{self.experiment_name}\nn_episodes = {self.time.n_episodes}")
        plt.hlines([self.area.height, -self.area.height], 
            -self.area.width, self.area.width, linestyle='--', color='grey')
        plt.vlines([self.area.width, -self.area.width], 
            -self.area.height, self.area.height, linestyle='--',  color='grey')
        plt.xlim([ -1.1 * self.area.width, 1.1 * self.area.width])
        plt.ylim([ -1.1 * self.area.height, 1.1 * self.area.height])
        plt.xticks([]); plt.yticks([])

        exit_coordinates = (self.area.exit.position[0], self.area.exit.position[1])
        agent_coordinates = (self.agent.memory['position'][0][0], self.agent.memory['position'][0][1])
        #plot Middle Wall
        plt.hlines([0], -1 + constants.WALL_HOLE_HALF_WIDTH, 1 - constants.WALL_HOLE_HALF_WIDTH, linestyle='--', color='grey')
        # Draw exiting zone
        exiting_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_exit, 
            0, 180, alpha=0.2, color='green'
        )
        ax.add_patch(exiting_zone)

        # Draw following zone
        following_zone = mpatches.Wedge(
            agent_coordinates, 
            SwitchDistances.to_leader, 
            0, 360, alpha=0.1, color='blue'
        )
        following_zone_plots = ax.add_patch(following_zone)

        # Draw escaping zone
        escaping_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_escape, 
            0, 180, color='white'
        )
        ax.add_patch(escaping_zone)
        
        # Draw exit
        ax.plot(exit_coordinates[0], exit_coordinates[1], marker='X', color='green')
        
        from itertools import cycle
        colors = cycle([item['color'] for item in ax._get_lines._cycler_items])
        
        # Draw pedestrians
        pedestrian_position_plots = {}
        for status in Status.all():
            selected_pedestrians = self.pedestrians.memory['statuses'][0] == status
            color = next(colors)
            # color = next(ax._get_lines.prop_cycler)['color']
            pedestrian_position_plots[status] = \
                ax.plot(self.pedestrians.memory['positions'][0][selected_pedestrians, 0], 
                self.pedestrians.memory['positions'][0][selected_pedestrians, 1],
                lw=0, marker='.', color=color)[0]

        # Draw agent
        agent_position_plot = ax.plot(agent_coordinates[0], agent_coordinates[1], marker='+', color='red')[0]

        def update(i):

            agent_coordinates = (self.agent.memory['position'][i][0], self.agent.memory['position'][i][1])
            following_zone_plots.set_center(agent_coordinates)

            for status in Status.all():
                selected_pedestrians = self.pedestrians.memory['statuses'][i] == status
                pedestrian_position_plots[status].set_xdata(self.pedestrians.memory['positions'][i][selected_pedestrians, 0])
                pedestrian_position_plots[status].set_ydata(self.pedestrians.memory['positions'][i][selected_pedestrians, 1])
                 
            # agent_position_plot.set_xdata(agent_coordinates[0])
            # agent_position_plot.set_ydata(agent_coordinates[1])
            agent_position_plot.set_data(agent_coordinates)

        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.time.now, interval=20)
        
        if not os.path.exists(constants.SAVE_PATH_GIFF): os.makedirs(constants.SAVE_PATH_GIFF)
        filename = os.path.join(constants.SAVE_PATH_GIFF, f'{self.experiment_name}_ep-{self.time.n_episodes}.gif')
        ani.save(filename=filename, writer='pillow')
        log.info(f"Env is rendered and gif animation is saved to {filename}")

        if self.save_next_episode_anim:
            self.save_next_episode_anim = False
            self.draw = False

    def close(self):
        pass
    
    def seed(self, seed=None):
        from gym.utils.seeding import np_random
        return np_random(seed)
# # %%
# e = EvacuationEnv(number_of_pedestrians=100)

# e.reset()
# e.step([1, 0])

# for i in range(300):
#     e.step([np.sin(i*0.1), np.cos(i*0.1)])
# e.save_animation()
# e.render()
# # %%
