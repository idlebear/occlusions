import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

BLOCK_SIZE = 32
Y_BLOCK_SIZE = 32
MAX_BLOCKS = 32

mod = SourceModule(
    """
    #include <cmath>
    #include <cfloat>

    const float SCAN_RANGE = 30.0;
    const float VEHICLE_LENGTH = 3.0;
    const float DISCOUNT_FACTOR = 1.0;

    typedef enum _VisibilityMethod {
        OURS,
        HIGGINS,
        ANDERSEN,
    } VisibilityMethod;


    typedef struct _Object {
        float x;
        float y;
        float radius;
    } Object;

    typedef struct _Obstacle {
        Object loc;
        float min_x;
        float min_y;
        float distance;
    } Obstacle;

    typedef struct _Costmap {
        float *map_data;
        float ox;
        float oy;
        float resolution;
        int height;
        int width;
    } Costmap;

    typedef struct _State {
        float x;
        float y;
        float v;
        float theta;
        float delta;
    } State;

    typedef struct _Control {
        float a;
        float delta;
    } Control;


    __device__
    float obstacle_cost( const float *obstacle_data, int num_obstacles, float px, float py, float radius ) {
        // Obstacles are stored as a flat array of floats, with each obstacle being a triplet of floats (x, y, radius)

        for (int i = 0; i < num_obstacles; i++) {
            auto obstacle = reinterpret_cast<const Obstacle *>(&obstacle_data[i]);
            float dx = obstacle->loc.x - px;
            float dy = obstacle->loc.y - py;
            float d_2 = dx * dx + dy * dy;
            float min_dist = obstacle->loc.radius + radius;

            if (d_2 < min_dist * min_dist) {
                return 100000.0;
            }
        }
        return 0.0;
    }


    __device__
    float higgins_cost( const float M, const float *obstacle_data, int num_obstacles, float px, float py ) {

        float cost = 0.0;

        float r_fov = SCAN_RANGE;
        float r_fov_2 = r_fov**2;

        for (int i = 0; i < num_obstacles; i++) {
            auto obstacle = reinterpret_cast<const Obstacle *>(&obstacle_data[i*sizeof(Obstacle)]);
            float dx = obstacle->loc.x - px;
            float dy = obstacle->loc.y - py;
            float d_2 = dx * dx + dy * dy;
            float d = sqrt(d_2);

            float inner = obstacle->loc.radius / d * (r_fov_2 - d_2);
            auto inner_exp = exp(inner);
            float score;
            if( inner_exp == INFINITY ) {
                score = M * inner;
            } else {
                score = M * log(1 + inner_exp);
            }
            cost += score;
        }

        return cost;
    }

    __device__
    float andersen_cost( const float M, const float *obstacle_data, int num_obstacles, float px, float py, float vx, float vy ) {
        float cost = 0.0;
        float v = sqrt(vx * vx + vy * vy);

        for (int i = 0; i < num_obstacles; i++) {
            auto obstacle = reinterpret_cast<const Obstacle *>(&obstacle_data[i]);
            float dx = obstacle->loc.x - px;
            float dy = obstacle->loc.y - py;

            // check if the obstacle is in front of the vehicle
            auto dot = dx * vx + dy * vy;
            if( dot > 0 ) {
                float d = sqrt(dx * dx + dy * dy);
                cost += M * acos(dot / (d * v));
            }
        }

        return cost;
    }


    __device__
    float our_cost( const float M, const Costmap *costmap, const float px, const float py, const float discount_factor, const int step) {
        float cost = 0.0;

        int map_x = int((px - costmap->ox) / costmap->resolution);
        int map_y = int((py - costmap->oy) / costmap->resolution);

        if (map_x < 0 || map_x >= costmap->width || map_y < 0 || map_y >= costmap->height) {
            return 100000.0;
        }

        cost = -M * (costmap->map_data[map_y * costmap->width + map_x]) * pow( discount_factor, step);

        return cost;
    }


    // Basic step function -- apply the control to advance one step
    __device__
    void euler( const State *state, const Control *control, State *result) {
        result->x = state->v * cos(state->theta);
        result->y = state->v * sin(state->theta);
        result->v = control->a;
        result->theta = state->v * tan(state->delta) / VEHICLE_LENGTH;
        result->delta = control->delta;
    }

    inline __device__
    void update_state( const State *state, const State *update, float dt, State *result) {
        result->x = state->x + update->x * dt;
        result->y = state->y + update->y * dt;
        result->v = state->v + update->v * dt;
        result->theta = state->theta + update->theta * dt;
        result->delta = state->delta + update->delta * dt;
    }

    //
    // Also define the Runge-Kutta variant as it is (apparently) a much
    // better approximation of the first order derivative
    //  https://en.wikipedia.org/wiki/Runge-Kutta_methods
    __device__
    void runge_kutta_step( const State *state, const Control *control, float dt, State *result) {
        State k1, k2, k3, k4;
        State tmp_state;

        euler( state, control, &k1 );
        update_state( state, &k1, dt / 2, &tmp_state );
        euler( &tmp_state, control, &k2 );
        update_state( state, &k2, dt / 2, &tmp_state );
        euler( &tmp_state, control, &k3 );
        update_state( state, &k3, dt, &tmp_state );
        euler( &tmp_state, control, &k4 );

        result->x = (k1.x + 2 * (k2.x + k3.x) + k4.x) / 6.0;
        result->y = (k1.y + 2 * (k2.y + k3.y) + k4.y) / 6.0;
        result->v = (k1.v + 2 * (k2.v + k3.v) + k4.v) / 6.0;
        result->theta = (k1.theta + 2 * (k2.theta + k3.theta) + k4.theta) / 6.0;
        result->delta = (k1.delta + 2 * (k2.delta + k3.delta) + k4.delta) / 6.0;
    }



    __global__
    void setup_kernel (curandState* state, unsigned long seed ) {
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init ( seed, id, 0, &state[id] );
    }

    __global__ void generate_kernel(curandState* globalState, float* random_nums) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        random_nums[idx] = generate_random_number(globalState, idx);
    }

    __device__
    float generate_random_number(curandState* globalState, int ind) {
        curandState localState = globalState[ind];
        float RANDOM = curand_uniform(&localState);
        globalState[ind] = localState;
        return RANDOM;
    }

    __device__
    void generate_controls(
            curandState *globalState,
            int index,
            const Control *u_nom,
            const int num_controls,
            float *u_limits,
            Control *u_dist
    ) {
        curandState localState = globalState[ind];
        for( int i = 0; i < num_controls; i++) {
            auto a_dist = curand_uniform(&localState) * u_limits[0] * 2 - u_limits[0];
            auto delta_dist = curand_uniform(&localState) * u_limits[1] * 2 - u_limits[1];
            u_dist[i].a = a_dist;
            u_dist[i].delta = delta_dist;
        }
        globalState[ind] = localState;
    }


    __device__
    float rollout(
            curandState *globalState,
            int index,

            const Costmap *costmap,
            const State *x_nom,
            const Control *u_nom,
            int num_controls,

            const float *obstacle_data,
            int num_obstacles,

            const float M,
            const float dt,

            float *u_limits,

            float *Q,
            float *R,

            Control *u_dist,
            float *u_weight,
            VisibilityMethod method
    ){
        generate_controls( globalState, index, u_nom, num_controls, u_limits, u_dist);

        State current_state;
        State state_step;
        update_state( &x_nom[0], &state_step, 0, &current_state );

        float score = 0.0;
        for (int i = 1; i <= num_controls; i++) {
            // generate the next state
            Control c = { u_nom[i-1].a + u_dist[i-1].a, u_nom[i-1].delta + u_dist[i-1].delta };
            runge_kutta_step( &current_state, &c, dt, &state_step );
            update_state( &current_state, &state_step, dt, &current_state );

            // penalize error in trajectory
            auto state_err = (x_nom[i].x - current_state.x) * Q[0] * (x_nom[i].x - current_state.x) +
                             (x_nom[i].y - current_state.y) * Q[1] * (x_nom[i].y - current_state.y) +
                             (x_nom[i].v - current_state.v) * Q[2] * (x_nom[i].v - current_state.v) +
                             (x_nom[i].theta - current_state.theta) * Q[3] * (x_nom[i].theta - current_state.theta) +
                             (x_nom[i].delta - current_state.delta) * Q[4] * (x_nom[i].delta - current_state.delta);

            // penalize control action
            auto control_err = (c.a - u_nom[i-1].a) * R[0] * (c.a - u_nom[i-1].a) +
                               (c.delta - u_nom[i-1].delta) * R[1] * (c.delta - u_nom[i-1].delta);

            // penalize obstacles
            auto obstacle_err = obstacle_cost(obstacle_data, num_obstacles, current_state.x, current_state.y, VEHICLE_LENGTH);

            // penalize visibility
            float visibility_err = 0.0;
            if (method == OURS) {
                visibility_err = our_cost(M, costmap, current_state.x, current_state.y, DISCOUNT_FACTOR, i);
            } else if (method == HIGGINS) {
                visibility_err = higgins_cost(M, obstacle_data, num_obstacles, current_state.x, current_state.y);
            } else if (method == ANDERSEN) {
                visibility_err = andersen_cost(M, obstacle_data, num_obstacles, current_state.x, current_state.y, state_step.x, state_step.y);
            }
            score += state_err + control_err + obstacle_err + visibility_err;

            u_weight[i-1] = score;
        }
    }



const float SCAN_RANGE = 30.0;
const float VEHICLE_LENGTH = 3.0;
const float DISCOUNT_FACTOR = 1.0;

typedef enum _VisibilityMethod {
    OURS,
    HIGGINS,
    ANDERSEN,
} VisibilityMethod;


typedef struct _Object {
    float x;
    float y;
    float radius;
} Object;

typedef struct _Obstacle {
    Object loc;
    float min_x;
    float min_y;
    float distance;
} Obstacle;

typedef struct _Costmap {
    float *map_data;
    float ox;
    float oy;
    float resolution;
    int height;
    int width;
} Costmap;

typedef struct _State {
    float x;
    float y;
    float v;
    float theta;
    float delta;
} State;

typedef struct _Control {
    float a;
    float delta;
} Control;


__device__
float obstacle_cost( const float *obstacle_data, int num_obstacles, float px, float py, float radius ) {
    // Obstacles are stored as a flat array of floats, with each obstacle being a triplet of floats (x, y, radius)

    for (int i = 0; i < num_obstacles; i++) {
        auto obstacle = reinterpret_cast<const Obstacle *>(&obstacle_data[i]);
        float dx = obstacle->loc.x - px;
        float dy = obstacle->loc.y - py;
        float d_2 = dx * dx + dy * dy;
        float min_dist = obstacle->loc.radius + radius;

        if (d_2 < min_dist * min_dist) {
            return 100000.0;
        }
    }
    return 0.0;
}


__device__
float higgins_cost( const float M, const float *obstacle_data, int num_obstacles, float px, float py ) {

    float cost = 0.0;

    float r_fov = SCAN_RANGE;
    float r_fov_2 = r_fov**2;

    for (int i = 0; i < num_obstacles; i++) {
        auto obstacle = reinterpret_cast<const Obstacle *>(&obstacle_data[i*sizeof(Obstacle)]);
        float dx = obstacle->loc.x - px;
        float dy = obstacle->loc.y - py;
        float d_2 = dx * dx + dy * dy;
        float d = sqrt(d_2);

        float inner = obstacle->loc.radius / d * (r_fov_2 - d_2);
        auto inner_exp = exp(inner);
        float score;
        if( inner_exp == INFINITY ) {
            score = M * inner;
        } else {
            score = M * log(1 + inner_exp);
        }
        cost += score;
    }

    return cost;
}

__device__
float andersen_cost( const float M, const float *obstacle_data, int num_obstacles, float px, float py, float vx, float vy ) {
    float cost = 0.0;
    float v = sqrt(vx * vx + vy * vy);

    for (int i = 0; i < num_obstacles; i++) {
        auto obstacle = reinterpret_cast<const Obstacle *>(&obstacle_data[i]);
        float dx = obstacle->loc.x - px;
        float dy = obstacle->loc.y - py;

        // check if the obstacle is in front of the vehicle
        auto dot = dx * vx + dy * vy;
        if( dot > 0 ) {
            float d = sqrt(dx * dx + dy * dy);
            cost += M * acos(dot / (d * v));
        }
    }

    return cost;
}


__device__
float our_cost( const float M, const Costmap *costmap, const float px, const float py, const float discount_factor, const int step) {
    float cost = 0.0;

    int map_x = int((px - costmap->ox) / costmap->resolution);
    int map_y = int((py - costmap->oy) / costmap->resolution);

    if (map_x < 0 || map_x >= costmap->width || map_y < 0 || map_y >= costmap->height) {
        return 100000.0;
    }

    cost = -M * (costmap->map_data[map_y * costmap->width + map_x]) * pow( discount_factor, step);

    return cost;
}


// Basic step function -- apply the control to advance one step
__device__
void euler( const State *state, const Control *control, State *result) {
    result->x = state->v * cos(state->theta);
    result->y = state->v * sin(state->theta);
    result->v = control->a;
    result->theta = state->v * tan(state->delta) / VEHICLE_LENGTH;
    result->delta = control->delta;
}

inline __device__
void update_state( const State *state, const State *update, float dt, State *result) {
    result->x = state->x + update->x * dt;
    result->y = state->y + update->y * dt;
    result->v = state->v + update->v * dt;
    result->theta = state->theta + update->theta * dt;
    result->delta = state->delta + update->delta * dt;
}

//
// Also define the Runge-Kutta variant as it is (apparently) a much
// better approximation of the first order derivative
//  https://en.wikipedia.org/wiki/Runge-Kutta_methods
__device__
void runge_kutta_step( const State *state, const Control *control, float dt, State *result) {
    State k1, k2, k3, k4;
    State tmp_state;

    euler( state, control, &k1 );
    update_state( state, &k1, dt / 2, &tmp_state );
    euler( &tmp_state, control, &k2 );
    update_state( state, &k2, dt / 2, &tmp_state );
    euler( &tmp_state, control, &k3 );
    update_state( state, &k3, dt, &tmp_state );
    euler( &tmp_state, control, &k4 );

    result->x = (k1.x + 2 * (k2.x + k3.x) + k4.x) / 6.0;
    result->y = (k1.y + 2 * (k2.y + k3.y) + k4.y) / 6.0;
    result->v = (k1.v + 2 * (k2.v + k3.v) + k4.v) / 6.0;
    result->theta = (k1.theta + 2 * (k2.theta + k3.theta) + k4.theta) / 6.0;
    result->delta = (k1.delta + 2 * (k2.delta + k3.delta) + k4.delta) / 6.0;
}



__global__
void setup_kernel (curandState* state, unsigned long seed ) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init ( seed, id, 0, &state[id] );
}

__global__ void generate_kernel(curandState* globalState, float* random_nums) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    random_nums[idx] = generate_random_number(globalState, idx);
}

__device__
float generate_random_number(curandState* globalState, int ind) {
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform(&localState);
    globalState[ind] = localState;
    return RANDOM;
}

__device__
void generate_controls(
        curandState *globalState,
        int index,
        const Control *u_nom,
        const int num_controls,
        float *u_limits,
        Control *u_dist
) {
    curandState localState = globalState[ind];
    for( int i = 0; i < num_controls; i++) {
        auto a_dist = curand_uniform(&localState) * u_limits[0] * 2 - u_limits[0];
        auto delta_dist = curand_uniform(&localState) * u_limits[1] * 2 - u_limits[1];
        u_dist[i].a = a_dist;
        u_dist[i].delta = delta_dist;
    }
    globalState[ind] = localState;
}


__device__
float rollout(
        curandState *globalState,
        int index,

        const Costmap *costmap,
        const State *x_nom,
        const Control *u_nom,
        int num_controls,

        const float *obstacle_data,
        int num_obstacles,

        const float M,
        const float dt,

        float *u_limits,

        float *Q,
        float *R,

        Control *u_dist,
        float *u_weight,
        VisibilityMethod method
){
    generate_controls( globalState, index, u_nom, num_controls, u_limits, u_dist);

    State current_state;
    State state_step;
    update_state( &x_nom[0], &state_step, 0, &current_state );

    float score = 0.0;
    for (int i = 1; i <= num_controls; i++) {
        // generate the next state
        Control c = { u_nom[i-1].a + u_dist[i-1].a, u_nom[i-1].delta + u_dist[i-1].delta };
        runge_kutta_step( &current_state, &c, dt, &state_step );
        update_state( &current_state, &state_step, dt, &current_state );

        // penalize error in trajectory
        auto state_err = (x_nom[i].x - current_state.x) * Q[0] * (x_nom[i].x - current_state.x) +
                         (x_nom[i].y - current_state.y) * Q[1] * (x_nom[i].y - current_state.y) +
                         (x_nom[i].v - current_state.v) * Q[2] * (x_nom[i].v - current_state.v) +
                         (x_nom[i].theta - current_state.theta) * Q[3] * (x_nom[i].theta - current_state.theta) +
                         (x_nom[i].delta - current_state.delta) * Q[4] * (x_nom[i].delta - current_state.delta);

        // penalize control action
        auto control_err = (c.a - u_nom[i-1].a) * R[0] * (c.a - u_nom[i-1].a) +
                           (c.delta - u_nom[i-1].delta) * R[1] * (c.delta - u_nom[i-1].delta);

        // penalize obstacles
        auto obstacle_err = obstacle_cost(obstacle_data, num_obstacles, current_state.x, current_state.y, VEHICLE_LENGTH);

        // penalize visibility
        float visibility_err = 0.0;
        if (method == OURS) {
            visibility_err = our_cost(M, costmap, current_state.x, current_state.y, DISCOUNT_FACTOR, i);
        } else if (method == HIGGINS) {
            visibility_err = higgins_cost(M, obstacle_data, num_obstacles, current_state.x, current_state.y);
        } else if (method == ANDERSEN) {
            visibility_err = andersen_cost(M, obstacle_data, num_obstacles, current_state.x, current_state.y, state_step.x, state_step.y);
        }
        score += state_err + control_err + obstacle_err + visibility_err;
    }
    *u_weight = score;
}

__global__ void perform_rollout(
        curandState *globalState,
        const Costmap* costmap,
        const float* x_nom,   // nominal states, num_controls + 1 x state_size
        int state_size,
        const float* u_nom,   // nominal controls, num_controls x control_size
        int num_controls,
        int control_size,
        const float* obstacle_data,
        int num_obstacles,
        int obstacle_size,
        int samples,
        float M,
        float dt,
        const float* u_limits,
        Control* u_dist,
        float* u_weight,
        float c_lambda,
        const float* Q,
        const float* R,
        VisibilityMethod method
) {
    int start_sample_index = blockIdx.y * blockDim.y + threadIdx.y;

    for (int j = start_sample_index; j < samples; j += blockDim.x * gridDim.x) {
        // rollout the trajectory -- assume we are placing the result in the larger u_dist/u_weight arrays
        rollout(globalState, start_sample_index, j, costmap, x_nom, u_nom, num_controls, obstacle_data, num_obstacles,
                M, dt,
                &u_dist[j * num_controls * control_size], &u_weight[j * num_controls], c_lambda);
    }


    const float M,
    const float dt,

    float *u_limits,

    float *Q,
    float *R,

    Control *u_dist,
    float *u_weight,
    VisibilityMethod method

}

__global__
void calculate_weights(
    int samples,
    float* u_weight,
    float* u_weight_totals,
    float c_lambda
) {
    int start_sample_index = blockIdx.y * blockDim.y + threadIdx.y;

    for (int j = start_sample_index; j < samples; j += blockDim.x * gridDim.x) {
        u_weight[j] = expf(-1.0f / c_lambda * u_weight[j]);
        atomicAdd(&(u_weight_totals[j]), u_weight[j]);
    }
}


__global__
void calculate_mppi_control(
        int samples,
        const Control* u_dist,
        const float* u_weight,
        const float* u_weight_totals,
        int num_controls,
        Control* u_mppi
) {
    int start_sample_index = blockIdx.y * blockDim.y + threadIdx.y;
    int start_control_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int j = start_sample_index; j < samples; j += blockDim.x * gridDim.x) {
        for (int i = start_control_index; i < num_controls; i += blockDim.y * gridDim.y) {
            atomicAdd( &(u_mppi[i*sizeof(Control)].a), u_dist[(j*num_controls + i)*sizeof(Control)].a * u_weight[j] / u_weight_totals[j] );
            atomicAdd( &(u_mppi[i*sizeof(Control)].delta), u_dist[(j*num_controls + i)*sizeof(Control)].delta * u_weight[j] / u_weight_totals[j] );
        }
    }
}




"""
)


def mppi(
    costmap,
    origin,
    resolution,
    x_nom,
    u_nom,
    actors,
    samples,
    dt,
    Q,
    R,
    c_lambda,
):
    costmap = costmap.astype(np.float32)
    height, width = costmap.shape
    data_gpu = cuda.mem_alloc(costmap.nbytes)
    cuda.memcpy_htod(data_gpu, costmap)

    actors = np.array(actors, dtype=np.float32)
    num_actors = len(actors)
    actors_gpu = cuda.mem_alloc(actors.nbytes)
    cuda.memcpy_htod(actors_gpu, actors)

    u_nom = np.array(u_nom, dtype=np.float32)
    num_controls = len(u_nom)
    u_nom_gpu = cuda.mem_alloc(u_nom.nbytes)
    cuda.memcpy_htod(u_nom_gpu, u_nom)

    x_nom = np.array(x_nom, dtype=np.float32)
    x_nom_gpu = cuda.mem_alloc(x_nom.nbytes)
    cuda.memcpy_htod(x_nom_gpu, x_nom)

    controls_size = u_nom.nbytes
    u_mppi_gpu = cuda.mem_alloc(controls_size)
    u_dist_gpu = cuda.mem_alloc(controls_size * samples)
    u_weight_gpu = cuda.mem_alloc(controls_size * samples)

    # block_size = 256
    # num_blocks = max(128, int((num_points + block_size - 1) / block_size))
    block = (BLOCK_SIZE, MAX_BLOCKS, 1)
    grid = (int((samples + block[0] - 1) / block[0]), int((num_controls + block[1] - 1) / block[1]))

    # perform the rollouts
    func = mod.get_function("perform_rollout")
    func(
        data_gpu,
        np.int32(height),
        np.int32(width),
        np.float32(origin[0]),
        np.float32(origin[1]),
        np.float32(resolution),
        actors_gpu,
        np.int32(num_actors),
        x_nom_gpu,
        u_nom_gpu,
        np.int32(num_controls),
        u_dist_gpu,
        u_weight_gpu,
        np.float32(c_lambda),
        np.int32(samples),
        np.float32(dt),
        np.float32(Q),
        np.float32(R),
        block=block,
        grid=grid,
    )

    # collect the rollouts into a single control
    func = mod.get_function("collect_rollouts")
    func(
        np.int32(samples),
        u_dist_gpu,
        u_weight_gpu,
        np.int32(num_controls),
        u_mppi_gpu,
        block=block,
        grid=grid,
    )

    # copy the results back
    u_mppi = np.zeros(controls_size, dtype=np.float32)
    cuda.memcpy_dtoh(u_mppi, u_mppi_gpu)

    return u_mppi
