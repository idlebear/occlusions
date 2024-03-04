import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import characterize

import numpy as np

from matplotlib import pyplot as plt

BLOCK_SIZE = 32


class MPPI:

    # TODO: this is a bit of a hack to make the visibility methods in the GPU code
    #       match the ones in the python code.
    visibility_methods = {
        "Ours": 0,
        "Higgins": 1,
        "Andersen": 2,
        "None": 3,
    }

    mod = SourceModule(
        """
    #include <cuda_runtime.h>
    #include <curand.h>
    #include <curand_kernel.h>
    #include <cmath>
    #include <cfloat>

    enum VisibilityMethod {
        OURS = 0,
        HIGGINS = 1,
        ANDERSEN = 2,
        NONE = 3
    };

    struct Costmap_Params {
        int height;
        int width;
        float origin_x;
        float origin_y;
        float resolution;
    };

    struct Optimization_Params {
        int samples;
        float M;
        float dt;
        int num_controls;
        int num_obstacles;
        float x_init[4];
        float u_limits[2];
        float Q[4];
        float R[2];
        long method;
        float c_lambda;
    } Optimization_Params;


    struct Object {
        float x;
        float y;
        float radius;
    } Object;

    struct Obstacle {
        Object loc;
        float min_x;
        float min_y;
        float distance;
    };

    struct State {
        float x;
        float y;
        float v;
        float theta;
    };

    struct Control {
        float a;
        float delta;
    };


    __device__
    float obstacle_cost(const float *obstacle_data, int num_obstacles, float px, float py, float radius) {
      for (int i = 0; i < num_obstacles; i++) {
        auto obstacle = reinterpret_cast<const Obstacle *>(&obstacle_data[i * (sizeof(Obstacle)/sizeof(float))]);
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
    float higgins_cost(const float M, const float *obstacle_data, int num_obstacles, float px, float py) {

      float cost = 0.0;

      float r_fov = SCAN_RANGE;
      float r_fov_2 = r_fov*r_fov;

      for (int i = 0; i < num_obstacles; i++) {
        auto obstacle = reinterpret_cast<const Obstacle *>(&obstacle_data[i * (sizeof(Obstacle)/sizeof(float))]);
        float dx = obstacle->loc.x - px;
        float dy = obstacle->loc.y - py;
        float d_2 = dx * dx + dy * dy;
        float d = sqrt(d_2);

        float inner = obstacle->loc.radius / d * (r_fov_2 - d_2);
        auto inner_exp = exp(inner);
        float score;
        if (inner_exp == INFINITY) {
          score = inner;
        } else {
          score = log(1 + inner_exp);
        }
        cost += M * score * score;
      }

      return cost;
    }

    __device__
    float
    andersen_cost(const float M, const float *obstacle_data, int num_obstacles, float px, float py, float vx, float vy) {
      float cost = 0.0;
      float v = sqrt(vx * vx + vy * vy);

      for (int i = 0; i < num_obstacles; i++) {
        auto obstacle = reinterpret_cast<const Obstacle *>(&obstacle_data[i * (sizeof(Obstacle)/sizeof(float))]);
        float dx = obstacle->min_x - px;
        float dy = obstacle->min_y - py;

        // check if the obstacle is in front of the vehicle
        auto dot = dx * vx + dy * vy;
        if (dot > 0) {
          float d = sqrt(dx * dx + dy * dy);
          cost += M * acos(dot / (d * v));
        }
      }

      return cost;
    }


    __device__
    float our_cost(const float M, const float *costmap, long height, long width, float origin_x, float origin_y, float resolution, const float px, const float py, const float discount_factor,
                   const int step) {
      float cost = 0.0;

      printf("px: %f, py: %f\\n", px, py);
      return 0.0;

      long map_x = long((px - origin_x) / resolution);
      long map_y = long((py - origin_y) / resolution);

      if (map_x < 0 || map_x >= width || map_y < 0 || map_y >= height) {
        return 100000.0;
      }

      cost = -M * (costmap[map_y * width + map_x]) * pow(discount_factor, step);

      return cost;
    }


    // Basic step function -- apply the control to advance one step
    __device__
    void euler(const State *state, const Control *control, State *result) {
      result->x = state->v * cos(state->theta);
      result->y = state->v * sin(state->theta);
      result->v = control->a;
      result->theta = state->v * tan(control->delta) / VEHICLE_LENGTH;
    }

    inline __device__
    void update_state(const State *state, const State *update, float dt, State *result) {
      result->x = state->x + update->x * dt;
      result->y = state->y + update->y * dt;
      result->v = state->v + update->v * dt;
      result->theta = state->theta + update->theta * dt;
    }

    //
    // Also define the Runge-Kutta variant as it is (apparently) a much
    // better approximation of the first order derivative
    //  https://en.wikipedia.org/wiki/Runge-Kutta_methods
    __device__
    void runge_kutta_step(const State *state, const Control *control, float dt, State *result) {
      State k1, k2, k3, k4;
      State tmp_state;

      euler(state, control, &k1);
      update_state(state, &k1, dt / 2, &tmp_state);
      euler(&tmp_state, control, &k2);
      update_state(state, &k2, dt / 2, &tmp_state);
      euler(&tmp_state, control, &k3);
      update_state(state, &k3, dt, &tmp_state);
      euler(&tmp_state, control, &k4);

      result->x = (k1.x + 2 * (k2.x + k3.x) + k4.x) / 6.0;
      result->y = (k1.y + 2 * (k2.y + k3.y) + k4.y) / 6.0;
      result->v = (k1.v + 2 * (k2.v + k3.v) + k4.v) / 6.0;
      result->theta = (k1.theta + 2 * (k2.theta + k3.theta) + k4.theta) / 6.0;
    }


    __device__
    void generate_controls(
            curandState *globalState,
            int index,
            const Control *u_nom,
            const int num_controls,
            const float *u_limits,
            Control *u_dist
    ) {
      curandState localState = globalState[index];
      for (int i = 0; i < num_controls; i++) {
        auto a_dist = curand_uniform(&localState) * u_limits[0] * 2 - u_limits[0];
        auto delta_dist = curand_uniform(&localState) * u_limits[1] * 2 - u_limits[1];
        u_dist[i].a = a_dist;
        u_dist[i].delta = delta_dist;
      }
      globalState[index] = localState;
    }


    // External functions -- each is wrapped with extern "C" to prevent name mangling
    // because pycuda doesn't support C++ name mangling
    extern "C" __global__
    void setup_kernel(curandState *state, unsigned long seed) {
      int id = threadIdx.x + blockIdx.x * blockDim.x;
      curand_init(seed, id, 0, &state[id]);
    }


    extern "C" __global__
    void perform_rollout(
            curandState *globalState,
            const float *costmap,
            const Costmap_Params *costmap_args,
            const float *x_nom,   // nominal states, num_controls + 1 x state_size
            const float *u_nom,   // nominal controls, num_controls x control_size
            const float *obstacle_data,
            const Optimization_Params *optimization_args,
            float *u_dists,
            float *u_weights
    ) {
        int start_sample_index = blockIdx.x * blockDim.x + threadIdx.x;
        int samples = optimization_args->samples;

        for (int sample_index = start_sample_index; sample_index < samples; sample_index += blockDim.x * gridDim.x) {

            int height = costmap_args->height;
            int width = costmap_args->width;
            float origin_x = costmap_args->origin_x;
            float origin_y = costmap_args->origin_y;
            float resolution = costmap_args->resolution;

            int num_controls = optimization_args->num_controls;
            int num_obstacles = optimization_args->num_obstacles;
            float M = optimization_args->M;
            float dt = optimization_args->dt;
            const float *x_init = optimization_args->x_init;
            const float *u_limits = optimization_args->u_limits;
            const float *Q = optimization_args->Q;
            const float *R = optimization_args->R;
            VisibilityMethod method = (VisibilityMethod) optimization_args->method;

            // printf("samples: %d\\n", samples);
            // printf("num_controls: %d\\n", num_controls);
            // printf("num_obstacles: %d\\n", num_obstacles);
            // printf("M: %f\\n", M);
            // printf("dt: %f\\n", dt);
            // printf("u_limits: %f, %f\\n", u_limits[0], u_limits[1]);
            // printf("Q: %f, %f, %f, %f, %f\\n", Q[0], Q[1], Q[2], Q[3], Q[4]);
            // printf("R: %f, %f\\n", R[0], R[1]);
            // printf("method: %d\\n", method);
            // printf("***************\\n");

            float score = 0.0;

            // rollout the trajectory -- assume we are placing the result in the larger u_dist/u_weight arrays
            const State *x_nom_states = reinterpret_cast<const State *>(x_nom);
            const State *x_init_state = reinterpret_cast<const State *>(x_init);
            const Control *u_nom_controls = reinterpret_cast<const Control *>(u_nom);
            Control *u_dist_controls = reinterpret_cast<Control *>(&u_dists[sample_index * num_controls * 2]);

            generate_controls(globalState, start_sample_index, u_nom_controls, num_controls, u_limits, u_dist_controls);

            State current_state;
            State state_step;
            update_state(x_init_state, &state_step, 0, &current_state);

            for (int i = 1; i <= num_controls; i++) {
                // generate the next state
                Control c = {u_nom_controls[i - 1].a + u_dist_controls[i - 1].a, u_nom_controls[i - 1].delta + u_dist_controls[i - 1].delta};
                runge_kutta_step(&current_state, &c, dt, &state_step);
                update_state(&current_state, &state_step, dt, &current_state);


                // penalize error in trajectory
                auto state_err = (x_nom_states[i].x - current_state.x) * Q[0] * (x_nom_states[i].x - current_state.x) +
                                 (x_nom_states[i].y - current_state.y) * Q[1] * (x_nom_states[i].y - current_state.y) +
                                 (x_nom_states[i].v - current_state.v) * Q[2] * (x_nom_states[i].v - current_state.v) +
                                (x_nom_states[i].theta - current_state.theta) * Q[3] * (x_nom_states[i].theta - current_state.theta);

                // penalize control action
                float control_err = (c.a - u_nom_controls[i - 1].a) * R[0] * (c.a - u_nom_controls[i - 1].a) +
                                   (c.delta - u_nom_controls[i - 1].delta) * R[1] * (c.delta - u_nom_controls[i - 1].delta);

                // penalize obstacles
                float obstacle_err = obstacle_cost(obstacle_data, num_obstacles, current_state.x, current_state.y, VEHICLE_LENGTH);

                // penalize visibility
                float visibility_err = 0;

                // printf( "origin_x: %f\\n", origin_x );
                // printf( "origin_y: %f\\n", origin_y );
                // printf( "resolution: %f\\n", resolution );
                // printf( "current_state.x: %f\\n", current_state.x );
                // printf( "current_state.y: %f\\n", current_state.y );
                // printf( "DISCOUNT_FACTOR: %f\\n", DISCOUNT_FACTOR );
                // printf( "height: %d\\n", height );
                // printf( "width: %d\\n", width );

                if (method == OURS) {
                    visibility_err = our_cost(M, costmap, height, width, origin_x, origin_y, resolution, current_state.x, current_state.y, DISCOUNT_FACTOR, i);
                } else if (method == HIGGINS) {
                    visibility_err = higgins_cost(M, obstacle_data, num_obstacles, current_state.x, current_state.y);
                } else if (method == ANDERSEN) {
                    visibility_err = andersen_cost(M, obstacle_data, num_obstacles, current_state.x, current_state.y, state_step.x, state_step.y);
                } else {
                    visibility_err = 0.0;
                }
                score += state_err + control_err + obstacle_err + visibility_err;
            }
            u_weights[sample_index] = score;
        }
    }

    extern "C" __global__
    void calculate_weights(
            int samples,
            float *u_weights,
            float c_lambda,
            float *u_weight_total
    ) {
      int start_sample_index = blockIdx.x * blockDim.x + threadIdx.x;

      for (int sample_index = start_sample_index; sample_index < samples; sample_index += blockDim.x * gridDim.x) {
        u_weights[sample_index] = expf(-1.0f / c_lambda * 0.5); // u_weights[sample_index]);
        atomicAdd(u_weight_total, u_weights[sample_index]);
      }
    }



    extern "C" __global__
    void calculate_mppi_control(
            int samples,
            const Control *u_dist,
            int num_controls,
            const float *u_weights,
            const float *u_weight_total,
            Control *u_mppi
    ) {
      int start_sample_index = blockIdx.x * blockDim.x + threadIdx.x;
      int start_control_index = blockIdx.y * blockDim.y + threadIdx.y;

      for (int sample_index = start_sample_index; sample_index < samples; sample_index += blockDim.x * gridDim.x) {
        for (int control_index = start_control_index; control_index < num_controls; control_index += blockDim.y * gridDim.y) {
          auto dist_index = sample_index * num_controls + control_index;
          atomicAdd(&(u_mppi[control_index].a),u_dist[dist_index].a * u_weights[sample_index] / u_weight_total[0]);
          atomicAdd(&(u_mppi[control_index].delta), u_dist[dist_index].delta * u_weights[sample_index] / u_weight_total[0]);
        }
      }
    }

    """,
        no_extern_c=True,
    )

    def __init__(self, samples, seed, u_limits, M, Q, R, method, c_lambda):

        self.samples = np.int32(samples)
        self.c_lambda = np.float32(c_lambda)
        self.method = np.int32(MPPI.visibility_methods[method])

        block = (BLOCK_SIZE, 1, 1)
        grid = (int((self.samples + block[0] - 1) / block[0]), 1)

        # setup the random number generator
        self.globalState_gpu = cuda.mem_alloc(
            block[0] * grid[0] * characterize.sizeof("curandState", "#include <curand_kernel.h>")
        )
        setup_kernel = MPPI.mod.get_function("setup_kernel")
        setup_kernel(
            self.globalState_gpu,
            np.uint32(seed),
            block=block,
            grid=grid,
        )

        self.optimization_dtype = np.dtype(
            [
                ("samples", np.int32),
                ("M", np.float32),
                ("dt", np.float32),
                ("num_controls", np.int32),
                ("num_obstacles", np.int32),
                ("x_init", np.float32, 4),
                ("u_limits", np.float32, 2),
                ("Q", np.float32, 4),
                ("R", np.float32, 2),
                ("method", np.int32),
                ("c_lambda", np.float32),
            ]
        )

        self.optimization_args = np.zeros(1, dtype=self.optimization_dtype)
        self.optimization_args["samples"] = np.int32(samples)
        self.optimization_args["M"] = np.float32(M)
        self.optimization_args["u_limits"] = np.array(u_limits, dtype=np.float32)
        self.optimization_args["Q"] = np.array(Q, dtype=np.float32)
        self.optimization_args["R"] = np.array(R, dtype=np.float32)
        self.optimization_args["method"] = np.int32(MPPI.visibility_methods[method])
        self.optimization_args["c_lambda"] = np.float32(c_lambda)
        self.optimization_args_gpu = cuda.mem_alloc(self.optimization_args.nbytes)

        self.costmap_dtype = np.dtype(
            [
                ("height", np.int32),
                ("width", np.int32),
                ("origin_x", np.float32),
                ("origin_y", np.float32),
                ("resolution", np.float32),
            ]
        )
        self.costmap_args = np.zeros(1, dtype=self.costmap_dtype)
        self.costmap_args_gpu = cuda.mem_alloc(self.costmap_args.nbytes)

    def find_control(self, costmap, origin, resolution, x_init, x_nom, u_nom, actors, dt):
        costmap = costmap.astype(np.float32)
        height, width = costmap.shape
        costmap_gpu = cuda.mem_alloc(costmap.nbytes)
        cuda.memcpy_htod(costmap_gpu, costmap)

        self.costmap_args["height"] = height
        self.costmap_args["width"] = width
        self.costmap_args["origin_x"] = origin[0]
        self.costmap_args["origin_y"] = origin[1]
        self.costmap_args["resolution"] = resolution
        cuda.memcpy_htod(self.costmap_args_gpu, self.costmap_args)

        if len(actors):
            actors = np.array(actors, dtype=np.float32)
            num_actors = len(actors)
            actors_gpu = cuda.mem_alloc(actors.nbytes)
            cuda.memcpy_htod(actors_gpu, actors)
        else:
            num_actors = 0
            actors_gpu = np.intp(0)

        u_nom = np.array(u_nom, dtype=np.float32).T
        controls_size = u_nom.nbytes
        num_controls, num_control_elements = u_nom.shape
        u_nom_gpu = cuda.mem_alloc(u_nom.nbytes)
        cuda.memcpy_htod(u_nom_gpu, u_nom)

        x_nom = np.array(x_nom, dtype=np.float32).T
        x_nom_gpu = cuda.mem_alloc(x_nom.nbytes)
        cuda.memcpy_htod(x_nom_gpu, x_nom)

        # allocate space for the outputs
        u_mppi_gpu = cuda.mem_alloc(controls_size)
        u_weight_gpu = cuda.mem_alloc(int(self.samples * np.float32(1).nbytes))
        u_dist_gpu = cuda.mem_alloc(int(controls_size * self.samples))

        # 1D blocks -- 1 thread per sample
        block = (BLOCK_SIZE, 1, 1)
        grid = (int((self.samples + block[0] - 1) / block[0]), 1)

        # update the optimization parameters
        self.optimization_args["dt"] = np.float32(dt)
        self.optimization_args["num_controls"] = np.int32(num_controls)
        self.optimization_args["num_obstacles"] = np.int32(num_actors)
        self.optimization_args["x_init"] = x_init
        cuda.memcpy_htod(self.optimization_args_gpu, self.optimization_args)

        # # Synchronize the device
        # cuda.Context.synchronize()

        # # perform the rollouts
        func = MPPI.mod.get_function("perform_rollout")
        func(
            self.globalState_gpu,
            costmap_gpu,
            self.costmap_args_gpu,
            x_nom_gpu,
            u_nom_gpu,
            actors_gpu,
            self.optimization_args_gpu,
            u_dist_gpu,
            u_weight_gpu,
            block=block,
            grid=grid,
        )

        # # Synchronize the device
        # cuda.Context.synchronize()

        # calculate the weights -- the block size remains the same
        u_weight_total = cuda.mem_alloc(np.float32(1).nbytes)
        func = MPPI.mod.get_function("calculate_weights")
        func(
            self.samples,
            u_weight_gpu,
            self.c_lambda,
            u_weight_total,
            block=block,
            grid=grid,
        )

        # # Synchronize the device
        # cuda.Context.synchronize()

        # final evaluation of the control -- 2D blocks, 1 thread per control and sample
        block = (BLOCK_SIZE, BLOCK_SIZE, 1)
        grid = (int((self.samples + block[0] - 1) / block[0]), int((num_controls + block[1] - 1) / block[1]))

        # collect the rollouts into a single control
        func = MPPI.mod.get_function("calculate_mppi_control")
        func(
            self.samples,
            u_dist_gpu,
            np.int32(num_controls),
            u_weight_gpu,
            u_weight_total,
            u_mppi_gpu,
            block=block,
            grid=grid,
        )

        # # Synchronize the device
        # cuda.Context.synchronize()

        # copy the results back
        u_dist = np.zeros((self.samples * num_controls * num_control_elements), dtype=np.float32)
        cuda.memcpy_dtoh(u_dist, u_dist_gpu)

        u_mppi = np.zeros_like(u_nom)
        cuda.memcpy_dtoh(u_mppi, u_mppi_gpu)

        return u_mppi.T, u_dist.reshape((self.samples, num_control_elements, -1))


import ModelParameters.Ackermann as Ackermann
import ModelParameters.GenericCar as GenericCar

global fig, ax, plot_lines, weighed_line, plot_backgrounds
fig, ax = None, None


# Basic step function -- apply the control to advance one step
def euler(vehicle, state, control):
    return vehicle.ode(state, control)


#
# Also define the Runge-Kutta variant as it is (apparently) a much
# better approximation of the first order derivative
#
# https://en.wikipedia.org/wiki/Runge-Kutta_methods
def runge_kutta_step(vehicle, state, control, dt):
    k1 = vehicle.ode(state, control)
    k2 = vehicle.ode(state + k1 * (dt / 2), control)
    k3 = vehicle.ode(state + k2 * (dt / 2), control)
    k4 = vehicle.ode(state + k3 * dt, control)

    return (k1 + 2 * (k2 + k3) + k4) / 6.0


# wrapper to allow easy switch between methods. (simplifies validation)
def step_fn(vehicle, state, control, dt=None):
    # return euler(vehicle=vehicle, state=state, control=control)
    return runge_kutta_step(vehicle=vehicle, state=state, control=control, dt=dt)


def rollout_trajectory(vehicle, state, controls, dt):
    states = [state]
    state = np.array(state)
    for control in controls:
        step = step_fn(vehicle=vehicle, state=state, control=control, dt=dt)
        state = state + step * dt
        states.append(state)
    return states


class Vehicle:
    control_len = 2
    state_len = 3

    # set limits on velocity and turning
    min_v = GenericCar.MIN_V
    max_v = GenericCar.MAX_V
    min_w = GenericCar.MIN_W
    max_w = GenericCar.MAX_W
    max_delta = GenericCar.MAX_DELTA
    min_delta = GenericCar.MIN_DELTA

    def __init__(self) -> None:
        pass

    #   Step Functions
    #   --------------
    #        x(k+1) = x(k) + v cos(theta(k)),
    #        y(k+1) = y(k) + v sin(theta(k)),
    #        theta(k+1) = theta(k) + w,
    #  next_state = [v*cos(theta)*dt, v*sin(theta)*dt
    @staticmethod
    def ode(state, control):
        ds0 = control[0] * np.cos(state[2])
        ds1 = control[0] * np.sin(state[2])
        ds2 = control[1]

        return np.vertcat(ds0, ds1, ds2)


class Ackermann3:
    CONTROL_LEN = 2  # v, omega
    STATE_LEN = 5  # x, y, theta

    def __init__(self, length=None) -> None:
        if length is None:
            self.L = Ackermann.L
        else:
            self.L = length

        # set defaults limit on velocity and turning
        self.min_v = Ackermann.MIN_V
        self.max_v = Ackermann.MAX_V
        self.min_a = Ackermann.MIN_A
        self.max_a = Ackermann.MAX_A
        self.max_delta = Ackermann.MAX_DELTA
        self.min_delta = Ackermann.MIN_DELTA
        self.min_w = Ackermann.MIN_W
        self.max_w = Ackermann.MAX_W

    #   Step Function
    def ode(self, state, control):
        ds0 = control[0] * np.cos(state[2])
        ds1 = control[0] * np.sin(state[2])
        ds2 = control[0] * np.tan(control[1]) / self.L

        return np.vertcat(ds0, ds1, ds2)


class Ackermann4:
    CONTROL_LEN = 2  # a, delta
    STATE_LEN = 4  # x, y, v, theta

    def __init__(self, length=None) -> None:
        if length is None:
            self.L = Ackermann.L
        else:
            self.L = length

        # set defaults limit on velocity and turning
        self.min_v = Ackermann.MIN_V
        self.max_v = Ackermann.MAX_V
        self.min_a = Ackermann.MIN_A
        self.max_a = Ackermann.MAX_A
        self.max_delta = Ackermann.MAX_DELTA
        self.min_delta = Ackermann.MIN_DELTA
        self.min_w = Ackermann.MIN_W
        self.max_w = Ackermann.MAX_W

    #   Step Function
    def ode(self, state, control):
        dx = state[2] * np.cos(state[3])
        dy = state[2] * np.sin(state[3])
        dv = control[0]
        dtheta = state[2] * np.tan(control[1]) / self.L

        # return np.vertcat(dx, dy, dv, dtheta)
        return np.array([dx, dy, dv, dtheta])


class Ackermann5:
    CONTROL_LEN = 2  # a, omega
    STATE_LEN = 5  # x, y, v, theta, delta

    def __init__(self, length=None) -> None:
        if length is None:
            self.L = Ackermann.L
        else:
            self.L = length

        # set defaults limit on velocity and turning
        self.min_v = Ackermann.MIN_V
        self.max_v = Ackermann.MAX_V
        self.min_a = Ackermann.MIN_A
        self.max_a = Ackermann.MAX_A
        self.max_delta = Ackermann.MAX_DELTA
        self.min_delta = Ackermann.MIN_DELTA
        self.min_w = Ackermann.MIN_W
        self.max_w = Ackermann.MAX_W

    #   Step Function
    def ode(self, state, control):
        dx = state[2] * np.cos(state[3])
        dy = state[2] * np.sin(state[3])
        dv = control[0]
        dtheta = state[2] * np.tan(state[4]) / self.L
        ddelta = control[1]

        # return np.vertcat(dx, dy, dv, dtheta, ddelta)
        return np.array([dx, dy, dv, dtheta, ddelta])


def run_trajectory(vehicle, initial_state, controls, dt):
    N, M = controls.shape

    traj = np.zeros((len(initial_state), M + 1))
    traj[:, 0] = initial_state

    state = np.array(initial_state)
    for m in range(M):
        u = controls[:, m]
        step = step_fn(vehicle=vehicle, state=state, control=u, dt=dt)
        state += step * dt
        traj[:, m + 1] = state

    return traj


def visualize_variations(vehicle, initial_state, u_nom, u_variations, u_weighted, dt):
    # visualizing!

    global fig, ax, plot_lines, weighted_line, plot_backgrounds
    if fig is None:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        plot_lines = None
        weighted_line = None
        plot_backgrounds = []

    n_samples, n_controls, n_steps = u_variations.shape
    new_traj_pts = []
    for i in range(n_samples):
        u_var = np.array(u_nom)
        u_var = u_var + u_variations[i, ...]

        traj = run_trajectory(vehicle=vehicle, initial_state=initial_state, controls=u_var, dt=dt)
        new_traj_pts.append(np.expand_dims(traj, axis=0))

    new_traj_pts = np.vstack(new_traj_pts)

    traj = run_trajectory(vehicle=vehicle, initial_state=initial_state, controls=u_weighted, dt=dt)

    if plot_lines is None:
        plot_lines = ax[0].plot(new_traj_pts[:, 0, :].T, new_traj_pts[:, 1, :].T)
        plot_initialized = True
        weighted_line = ax[1].plot(traj[0, :], traj[1, :])

        ax[0].axis("equal")
        ax[1].axis("equal")
        plt.show(block=False)
    else:
        for line, data in zip(plot_lines, new_traj_pts):
            line.set_data(data[0, :], data[1, :])

        weighted_line[0].set_data(traj[0, :], traj[1, :])

        ax[0].relim()
        ax[0].autoscale_view()
        ax[1].relim()
        ax[1].autoscale_view()

        plt.pause(0.01)


if __name__ == "__main__":

    from time import time

    samples = 1000
    seed = 123
    u_limits = [3.4, np.pi / 24]
    M = -0.4
    Q = [1, 1, 1, 1]
    R = [1, 1]
    method = "None"
    c_lambda = 1

    tic = time()
    mppi = MPPI(samples, seed, u_limits, M, Q, R, method, c_lambda)

    costmap = np.zeros((100, 100))
    origin = (0, 0)
    resolution = 1
    x_nom = np.zeros((4, 10))
    u_nom = np.ones((2, 9))
    x_init = np.array([0.0, 0.0, 1.0, 0.0])
    actors = []
    dt = 0.1
    u_mppi, u_dist = mppi.find_control(costmap, origin, resolution, x_init, x_nom, u_nom, actors, dt)

    toc = time()
    print(f"Time: {toc - tic}, per sample: {(toc - tic) / samples}")

    visualize_variations(
        vehicle=Ackermann4(),
        initial_state=x_init,
        u_nom=u_nom,
        u_variations=u_dist,
        u_weighted=u_mppi,
        dt=dt,
    )

    print(u_mppi)
    print(u_dist)
