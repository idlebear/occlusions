import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoprimaryctx  # import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import characterize

import numpy as np

from matplotlib import pyplot as plt

BLOCK_SIZE = 32


class ControlVariations:

    vehicle_model_ids = {
        "Ackermann4": 0,
        "Ackermann5": 1,
        "Unicycle": 2,
        "SkidSteer": 3,
    }

    mod = SourceModule(
        """
    #include <cuda_runtime.h>
    #include <curand.h>
    #include <curand_kernel.h>
    #include <cmath>
    #include <cfloat>

    enum VehicleModel {
        ACKERMANN4 = 0,
        ACKERMANN5 = 1,
        UNICYCLE = 2,
        SKIDSTEER = 3
    };

    struct Optimization_Params {
        int samples;
        float dt;
        int num_controls;
        float x_init[4];
        float u_limits[2];
        float u_dist_limits[2];
        float vehicle_length;
        int vehicle_model_id;
    };

    struct Object {
        float x;
        float y;
        float radius;
    };

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
        float first;
        float second;
    };



    // Basic step function -- apply the control to advance one step
    __device__
    void A4_euler(const State *state, const Control *control, float vehicle_length, State *result) {
        result->x     = state->v * cosf(state->theta);
        result->y     = state->v * sinf(state->theta);
        result->theta = state->v * tanf(control->second) / vehicle_length;
        result->v     = control->first;
    }

    __device__
    void UNI_euler(const State *state, const Control *control, State *result) {
        result->x     = control->first * cosf(state->theta);
        result->y     = control->first * sinf(state->theta);
        result->theta = control->second;
        result->v     = control->first;   // not used as this is constant velocity
    }

    inline __device__
    void update_state(const State *state, const State *update, float dt, State *result) {
      result->x     = state->x     + update->x * dt;
      result->y     = state->y     + update->y * dt;
      result->v     = state->v     + update->v * dt;
      result->theta = state->theta + update->theta * dt;
    }

    //
    // Also define the Runge-Kutta variant as it is (apparently) a much
    // better approximation of the first order derivative
    //  https://en.wikipedia.org/wiki/Runge-Kutta_methods
    __device__
    void A4_runge_kutta_step(const State *state, const Control *control, float vehicle_length, float dt, State *result) {
      State k1, k2, k3, k4;
      State tmp_state;

      A4_euler(state, control, vehicle_length, &k1);
      update_state(state, &k1, dt / 2, &tmp_state);
      A4_euler(&tmp_state, control, vehicle_length, &k2);
      update_state(state, &k2, dt / 2, &tmp_state);
      A4_euler(&tmp_state, control, vehicle_length, &k3);
      update_state(state, &k3, dt, &tmp_state);
      A4_euler(&tmp_state, control, vehicle_length, &k4);

      result->x = (k1.x + 2 * (k2.x + k3.x) + k4.x) / 6.0;
      result->y = (k1.y + 2 * (k2.y + k3.y) + k4.y) / 6.0;
      result->v = (k1.v + 2 * (k2.v + k3.v) + k4.v) / 6.0;
      result->theta = (k1.theta + 2 * (k2.theta + k3.theta) + k4.theta) / 6.0;
    }

    //
    // Also define the Runge-Kutta variant as it is (apparently) a much
    // better approximation of the first order derivative
    //  https://en.wikipedia.org/wiki/Runge-Kutta_methods
    __device__
    void UNI_runge_kutta_step(const State *state, const Control *control, float dt, State *result) {
      State k1, k2, k3, k4;
      State tmp_state;

      UNI_euler(state, control, &k1);
      update_state(state, &k1, dt / 2, &tmp_state);
      UNI_euler(&tmp_state, control, &k2);
      update_state(state, &k2, dt / 2, &tmp_state);
      UNI_euler(&tmp_state, control, &k3);
      update_state(state, &k3, dt, &tmp_state);
      UNI_euler(&tmp_state, control, &k4);

      result->x = (k1.x + 2 * (k2.x + k3.x) + k4.x) / 6.0;
      result->y = (k1.y + 2 * (k2.y + k3.y) + k4.y) / 6.0;
      result->v = (k1.v + 2 * (k2.v + k3.v) + k4.v) / 6.0;
      result->theta = (k1.theta + 2 * (k2.theta + k3.theta) + k4.theta) / 6.0;
    }



    __device__
    void generate_control(
            curandState *globalState,
            int index,
            const Control *u_nom,
            const float *u_limits,
            const float *u_dist_limits,
            Control *u
    ) {
        curandState localState = globalState[index];

        float first_dist;
        float second_dist;
        int count = 0;
        // sample until we get a valid control
        do {
            count++;
            if (count > 1000) {
                printf( "%d: No first control found! u_nom(0): %f, first: %f, limit: %f", index, u_nom->first, first_dist, u_limits[0]  );
                first_dist = 0;
                break;
            }
            first_dist = curand_uniform(&localState) * u_dist_limits[0] * 2.0 - u_dist_limits[0];
        } while ((u_nom->first + first_dist > u_limits[0]) || (u_nom->first + first_dist < -u_limits[0]) );
        count = 0;
        do {
            count++;
            if (count > 1000) {
                printf( "%d: No second control found! u_nom(1): %f, second: %f, limit: %f", index, u_nom->second, second_dist, u_limits[1] );
                second_dist = 0;
                break;
            }
            second_dist = curand_uniform(&localState) * u_dist_limits[1] * 2.0 - u_dist_limits[1];
        } while ( (u_nom->second + second_dist > u_limits[1] ) || ( u_nom->second + second_dist < -u_limits[1] ) );

        u->first = u_nom->first + first_dist;
        u->second = u_nom->second + second_dist;
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
            const Control *u_nom,   // nominal controls, num_controls x control_size
            const Optimization_Params *optimization_args,
            State *x_dists
    ) {
        int start_sample_index = blockIdx.x * blockDim.x + threadIdx.x;
        int samples = optimization_args->samples;

        for (int sample_index = start_sample_index; sample_index < samples; sample_index += blockDim.x * gridDim.x) {

            int num_controls = optimization_args->num_controls;
            int num_states = num_controls + 1;

            float dt = optimization_args->dt;
            const float *u_limits = optimization_args->u_limits;
            const float *u_dist_limits = optimization_args->u_dist_limits;
            VehicleModel vehicle_model_id = (VehicleModel) optimization_args->vehicle_model_id;

            State* x_dist = x_dists + sample_index * num_states;

            /*
            printf("samples: %d\\n", samples);
            printf("num_controls: %d\\n", num_controls);
            printf("dt: %f\\n", dt);
            printf("u_limits: %f, %f\\n", u_limits[0], u_limits[1]);
            printf("vehicle_length: %f\\n", optimization_args->vehicle_length);
            */

            // rollout the trajectory -- assume we are placing the result in the larger u_dist/u_weight arrays
            const State *x_init_state = reinterpret_cast<const State *>(optimization_args->x_init);

            State state_step = {0,0,0,0};
            update_state(x_init_state, &state_step, 0, &x_dist[0]);    // initialize the first state
/*
            printf( "initial state.x: %f\\n", x_dist[0].x );
            printf( "initial state.y: %f\\n", x_dist[0].y );
            printf( "initial state.v: %f\\n", x_dist[0].v );
            printf( "initial state.theta: %f\\n", x_dist[0].theta );
*/
            for (int i = 1; i <= num_controls; i++) {
                // generate the next state
                Control c = {0,0};
                generate_control(globalState, start_sample_index, &u_nom[i-1], u_limits, u_dist_limits, &c);

                switch (vehicle_model_id) {
                    case ACKERMANN4:
                        A4_runge_kutta_step(&x_dist[i-1], &c, optimization_args->vehicle_length, dt, &state_step);
                        break;
                    case UNICYCLE:
                        UNI_runge_kutta_step(&x_dist[i-1], &c, dt, &state_step);
                        state_step.v = c.first;
                        break;
                    case ACKERMANN5:
                    case SKIDSTEER:
                        // not implemented
                        break;
                }

                update_state(&x_dist[i-1], &state_step, dt, &x_dist[i]);
/*
                printf( "current_state.x: %f\\n", x_dists[i].x );
                printf( "current_state.y: %f\\n", x_dists[i].y );
                printf( "current_state.v: %f\\n", x_dists[i].v );
                printf( "current_state.theta: %f\\n", x_dists[i].theta );
                printf( "c.first: %f\\n", c.first );
                printf( "c.second: %f\\n", c.second );
*/
            }
        }
    }


    //
    // Based on a comment from the following link on checking for zero:
    //
    // https://forums.developer.nvidia.com/t/on-tackling-float-point-precision-issues-in-cuda/79060
    //
    __device__
    inline bool isZero(float f){
        return f >= -FLT_EPSILON && f <= FLT_EPSILON;
    }

    __device__
    inline bool isEqual(float f1, float f2){
        return fabs(f1 - f2) < FLT_EPSILON;
    }


    """,
        no_extern_c=True,
    )

    def __init__(
        self,
        vehicle,
        samples,
        seed,
        u_limits,
        u_dist_limits,
    ):
        self.vehicle = vehicle
        self.vehicle_model_id = ControlVariations.vehicle_model_ids[vehicle.__class__.__name__]
        if hasattr(vehicle, "L"):
            self.vehicle_length = vehicle.L
        else:
            self.vehicle_length = 0

        self.samples = np.int32(samples)
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)

        block = (BLOCK_SIZE, 1, 1)
        grid = (int((self.samples + block[0] - 1) / block[0]), 1)

        # setup the random number generator
        self.globalState_gpu = cuda.mem_alloc(
            block[0] * grid[0] * characterize.sizeof("curandState", "#include <curand_kernel.h>")
        )
        setup_kernel = ControlVariations.mod.get_function("setup_kernel")
        setup_kernel(
            self.globalState_gpu,
            np.uint32(seed),
            block=block,
            grid=grid,
        )

        self.optimization_dtype = np.dtype(
            [
                ("samples", np.int32),
                ("dt", np.float32),
                ("num_controls", np.int32),
                ("x_init", np.float32, 4),
                ("u_limits", np.float32, 2),
                ("u_dist_limits", np.float32, 2),
                ("vehicle_length", np.float32),
                ("vehicle_model_id", np.int32),
            ]
        )

        self.optimization_args = np.zeros(1, dtype=self.optimization_dtype)
        self.optimization_args["samples"] = np.int32(self.samples)
        self.optimization_args["u_limits"] = np.array(u_limits, dtype=np.float32)
        self.optimization_args["u_dist_limits"] = np.array(u_dist_limits, dtype=np.float32)
        self.optimization_args["vehicle_length"] = np.float32(self.vehicle_length)
        self.optimization_args["vehicle_model_id"] = np.int32(self.vehicle_model_id)

        self.optimization_args_gpu = cuda.mem_alloc(self.optimization_args.nbytes)

    def predict(self, x_init, u_nom, dt):
        u_nom = np.array(u_nom, dtype=np.float32)
        controls_size = u_nom.nbytes
        num_controls, num_control_elements = u_nom.shape
        num_states = num_controls + 1
        x_init = np.array(x_init, dtype=np.float32)
        num_state_elements = x_init.shape[0]
        states_size = num_state_elements * num_states * np.float32().nbytes
        u_nom_gpu = cuda.mem_alloc(u_nom.nbytes)
        cuda.memcpy_htod(u_nom_gpu, u_nom)

        # allocate space for the outputs
        u_var_gpu = cuda.mem_alloc(controls_size)
        cuda.memset_d8(u_var_gpu, 0, controls_size)

        x_dist_gpu = cuda.mem_alloc(int(states_size * self.samples))

        # 1D blocks -- 1 thread per sample
        block = (BLOCK_SIZE, 1, 1)
        grid = (int((self.samples + block[0] - 1) / block[0]), 1)

        # update the optimization parameters
        self.optimization_args["dt"] = np.float32(dt)
        self.optimization_args["num_controls"] = np.int32(num_controls)
        self.optimization_args["x_init"] = x_init
        cuda.memcpy_htod(self.optimization_args_gpu, self.optimization_args)

        # # Synchronize the device
        # cuda.Context.synchronize()

        # # perform the rollouts
        func = ControlVariations.mod.get_function("perform_rollout")
        func(
            self.globalState_gpu,
            u_nom_gpu,
            self.optimization_args_gpu,
            x_dist_gpu,
            block=block,
            grid=grid,
        )

        # # Synchronize the device
        # cuda.Context.synchronize()

        # copy the results back
        x_dist = np.zeros((self.samples * num_states * num_state_elements), dtype=np.float32)
        cuda.memcpy_dtoh(x_dist, x_dist_gpu)
        x_dist = x_dist.reshape((self.samples, -1, num_state_elements))
        return x_dist


if __name__ == "__main__":

    from time import time
    from controller.ModelParameters.Ackermann import Ackermann4
    from controller.ModelParameters.Unicycle import Unicycle

    samples = 100
    seed = 123
    u_dist_limits = [2, np.pi / 5]
    u_limits = [4, np.inf]

    vehicle = Unicycle()

    tic = time()
    mppi = MPPI(
        vehicle=vehicle,
        samples=samples,
        seed=seed,
        u_limits=u_limits,
        u_dist_limits=u_dist_limits,
    )

    u_nom = np.ones((21, 2))
    u_nom[:, 0] = 3.4
    u_nom[:, 1] = 0

    x_init = np.array([0.0, 0.0, 1.0, 0.0])
    actors = [
        [9, 0, 5, 13, 5, np.sqrt(15 * 15 + 5 * 5)],
        [20, 20, 5, 25, 5, np.sqrt(25 * 25 + 5 * 5)],
    ]
    actors = np.array(actors, dtype=np.float32)

    dt = 0.1
    x_variations = mppi.find_variations(x_init, u_nom, dt)

    toc = time()
    print(f"Time: {toc - tic}, per sample: {(toc - tic) / samples}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for var in x_variations:
        ax.plot(var[:, 0], var[:, 1])
    ax.set_aspect("equal")
    plt.show()

    print("Done")
