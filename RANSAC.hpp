#ifndef RANSAC_HPP
#define RANSAC_HPP

#define GLM_FORCE_CUDA

#ifndef GLM_COMPILER
	#define GLM_COMPILER 0
#endif

#include <cuda.h>
#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <mutex>
#include <thread>
#include <vector>
#include <thrust/device_vector.h>

constexpr int MODEL_SIZE=3;

//structure containing a Rotation and Translation
//each index in R represents a column of the matrix, represented by a float3
struct RT
{
	__host__ __device__ RT()
	{
		R[0] = make_float3(1,0,0);
		R[1] = make_float3(0,1,0);
		R[2] = make_float3(0,0,1);
		T = make_float3(0,0,0);
	};
	__host__ __device__ ~RT(){};

	float3 R[3];
	float3 T;
};


class RANSAC
{
public:
	RANSAC( const uint max_iterations,
			const uint max_iterations_per_batch,
			const float error_tolerance_dist,
			const float inlier_ratio_to_accept_solution,
			thrust::device_vector<float3> vertices_source,
			thrust::device_vector<float3> vertices_target,
			thrust::device_vector<float3> vertices_fitness_source,
			thrust::device_vector<float3> vertices_fitness_target
			);

	~RANSAC();

	float GetCurrentScore();
	uint3 GetTargetModel();
	uint3 GetSourceModel();
	void GetCurrentSolution( RT &solution);
	void start();
	void reset();
	void stop();
	bool has_finished();

protected:

	void doIteration();

	std::mutex	_transformation_mutex;
	std::mutex	_iteration_mutex;
	RT			_current_best_solution;
	uint		_current_best_model_pair_index;
	float		_current_best_score;
	uint		_iteration_counter;
	uint		_max_iterations;
	uint		_max_iterations_per_batch;
	float		_error_tolerance_dist;
	float		_inlier_ratio_to_accept_solution;
	uint		_numThreadsPerBlock;
	uint		_numBlocks;
	uint		_numThreads;
	bool		_time_to_stop;

	thrust::host_vector<uint>		_vert_subset_source_host;
	thrust::host_vector<uint>		_vert_subset_target_host;

	thrust::device_vector<uint>		_vert_subset_source;
	thrust::device_vector<uint>		_vert_subset_target;

	thrust::device_vector<float3>	_vertices_source;
	thrust::device_vector<float3>	_vertices_target;

	thrust::device_vector<float3>	_vertices_fitness_source;
	thrust::device_vector<float3>	_vertices_fitness_target;

	thrust::device_vector<RT>	 	_candidate_solutions;
	thrust::device_vector<float>	_solution_scores;
};

#endif