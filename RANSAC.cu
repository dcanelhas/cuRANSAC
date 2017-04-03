#include "RANSAC.hpp"
#include <unistd.h>
#include <random>
#include <iomanip>
#include <algorithm>
#include <set>
#include <cmath>
#include "cutil_math.h"
#include <thrust/extrema.h>

__host__ __device__ __forceinline__
float accurateSqrt(float x)
{
    return x * rsqrt(x);
}

inline __host__ __device__ float norm(float3 v)
{
    return sqrtf(dot(v, v));
}

__host__ __device__ __forceinline__
float dist2(float x, float y, float z)
{
    return x*x+y*y+z*z;
}


template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
	typedef typename Vector::value_type T;
	std::cout << "	" << std::setw(20) << name << "	";
	thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
	std::cout << std::endl;
}

__host__ __device__ __forceinline__
void printMat3(	float a11, float a12, float a13,
				float a21, float a22, float a23,
				float a31, float a32, float a33)
{
	printf("%f %f %f \n", a11, a12, a13);
	printf("%f %f %f \n", a21, a22, a23);
	printf("%f %f %f \n", a31, a32, a33);
}

__global__ void computeRT(
	const uint 	offset,
	const float* vertices_source,
	const float* vertices_target,
	const uint* selection_source,
	const uint* selection_target,
	RT* candidate_solutions,
	uint num_candidates)
{
	uint thread = blockIdx.x *blockDim.x + threadIdx.x;

	if(thread + offset >= num_candidates) return;

	float3 vs[MODEL_SIZE];
	float3 vt[MODEL_SIZE];

	for (int i = 0; i < MODEL_SIZE; ++i)
	{
		vs[i] = make_float3(vertices_source[ selection_source[ (offset + thread) * MODEL_SIZE + i] * 3 + 0],
							vertices_source[ selection_source[ (offset + thread) * MODEL_SIZE + i] * 3 + 1],
							vertices_source[ selection_source[ (offset + thread) * MODEL_SIZE + i] * 3 + 2]);

		vt[i] = make_float3(vertices_target[ selection_target[ (offset + thread) * MODEL_SIZE + i] * 3 + 0],
							vertices_target[ selection_target[ (offset + thread) * MODEL_SIZE + i] * 3 + 1],
							vertices_target[ selection_target[ (offset + thread) * MODEL_SIZE + i] * 3 + 2]);
	}

	float3 s_mean = make_float3(0,0,0);
	float3 t_mean = make_float3(0,0,0);

	for (int i = 0; i < MODEL_SIZE; ++i)
	{
		s_mean += vs[i];
		t_mean += vt[i];
	}
	s_mean /= MODEL_SIZE;
	t_mean /= MODEL_SIZE;

	for (int i = 0; i < MODEL_SIZE; ++i)
	{
		vs[i] -= s_mean;
		vt[i] -= t_mean;
	}

	float3 rs_2 = vs[2] - vs[0];
	float3 rs_1 = vs[1] - vs[0];
	float3 rt_2 = vt[2] - vt[0];
	float3 rt_1 = vt[1] - vt[0];
	float3 ns = cross(rs_2, rs_1);
	float3 ns_n = normalize(ns);
	float3 nt = cross(rt_2, rt_1);
	float3 nt_n = normalize(nt);
	float3 a = normalize(cross(ns,nt));

	//phi is the out-of-plane rotation along the intersection between the two planes formed by the vertex triplets vs and vt
	float sin_phi = norm(cross(ns_n, nt_n));
	float cos_phi = dot(ns_n, nt_n);

    float t = 1.0 - cos_phi;

    float ra00 = cos_phi + a.x*a.x*t;
    float ra11 = cos_phi + a.y*a.y*t;
    float ra22 = cos_phi + a.z*a.z*t;
    float tmp1 = a.x*a.y*t;
    float tmp2 = a.z*sin_phi;
    float ra10 = tmp1 + tmp2;
    float ra01 = tmp1 - tmp2;
    tmp1 = a.x*a.z*t;
    tmp2 = a.y*sin_phi;
    float ra20 = tmp1 - tmp2;
    float ra02 = tmp1 + tmp2;
    tmp1 = a.y*a.z*t;
    tmp2 = a.x*sin_phi;
    float ra21 = tmp1 + tmp2;
    float ra12 = tmp1 - tmp2;

    float S = 0;
    float C = 0;
    for (int i = 0; i < MODEL_SIZE; ++i)
	{
		float3 vsr = vs[i]*cos_phi + cross(a,vs[i])*sin_phi + a * ( dot(a,vs[i]) * (1.0f-cos_phi) );
		S += dot(cross(vt[i], vsr), nt_n);
		C += dot(vt[i], vsr);
	}

	float sin_theta = S/sqrt(S*S + C*C);
	float cos_theta = C/sqrt(S*S + C*C);

	t = 1.0 - cos_theta;

	float3 b = nt_n;

    float rb00 = cos_theta + b.x*b.x*t;
    float rb11 = cos_theta + b.y*b.y*t;
    float rb22 = cos_theta + b.z*b.z*t;
    tmp1 = b.x*b.y*t;
    tmp2 = b.z*sin_theta;
    float rb10 = tmp1 + tmp2;
    float rb01 = tmp1 - tmp2;
    tmp1 = b.x*b.z*t;
    tmp2 = b.y*sin_theta;
    float rb20 = tmp1 - tmp2;
    float rb02 = tmp1 + tmp2;
    tmp1 = b.y*b.z*t;
    tmp2 = b.x*sin_theta;
    float rb21 = tmp1 + tmp2;
    float rb12 = tmp1 - tmp2;

	float R00 = ra00*rb00 + ra01*rb10 + ra02*rb20; float R01 = ra00*rb01 + ra01*rb11 + ra02*rb21; float R02 = ra00*rb02 + ra01*rb12 + ra02*rb22;
	float R10 = ra10*rb00 + ra11*rb10 + ra12*rb20; float R11 = ra10*rb01 + ra11*rb11 + ra12*rb21; float R12 = ra10*rb02 + ra11*rb12 + ra12*rb22;
	float R20 = ra20*rb00 + ra21*rb10 + ra22*rb20; float R21 = ra20*rb01 + ra21*rb11 + ra22*rb21; float R22 = ra20*rb02 + ra21*rb12 + ra22*rb22;


	RT a_solution;
	a_solution.R[0] = make_float3(R00, R10, R20);
	a_solution.R[1] = make_float3(R01, R11, R21);
	a_solution.R[2] = make_float3(R02, R12, R22);

	// solve for the translation vector
	// T = Ym - R*Xm
	float3 T = t_mean - make_float3(R00 * s_mean.x + R01 * s_mean.y + R02 * s_mean.z,
									R10 * s_mean.x + R11 * s_mean.y + R12 * s_mean.z,
									R20 * s_mean.x + R21 * s_mean.y + R22 * s_mean.z);
	a_solution.T = T;

	candidate_solutions[thread] = a_solution;

	/*
	if (thread==0)
	{
		printf("selection =\n");
		for (int i = 0; i < MODEL_SIZE; ++i)
		{
			printf("%d %d %d = %f %f %f \n",
				offset,
				(offset + thread) * MODEL_SIZE + i,
				selection_source[ (offset + thread) * MODEL_SIZE + i],
				vertices_source[ selection_source[ (offset + thread) * MODEL_SIZE + i] * 3 + 0],
				vertices_source[ selection_source[ (offset + thread) * MODEL_SIZE + i] * 3 + 1],
				vertices_source[ selection_source[ (offset + thread) * MODEL_SIZE + i] * 3 + 2]
				);
		}
		printf("s_mean = %f %f %f \n", s_mean.x, s_mean.y, s_mean.z);
		printf("t_mean = %f %f %f \n", t_mean.x, t_mean.y, t_mean.z);

		printf("T = %f %f %f \n", a_solution.T.x, a_solution.T.y, a_solution.T.z);

		printMat3(	R_11, R_12, R_13,
					R_21, R_22, R_23,
					R_31, R_32, R_33);


	}
	*/

}

__global__ void computeRT(
	const uint 	offset,
	const float3* vertices_source,
	const float3* vertices_target,
	const uint* selection_source,
	const uint* selection_target,
	RT* candidate_solutions,
	uint num_candidates)
{
	uint thread = blockIdx.x *blockDim.x + threadIdx.x;

	if( offset + thread >= num_candidates) return;

	float3 vs[MODEL_SIZE];
	float3 vt[MODEL_SIZE];

	for (int i = 0; i < MODEL_SIZE; ++i)
	{
		vs[i] = vertices_source[ selection_source[ (offset + thread) * MODEL_SIZE + i]];
		vt[i] = vertices_target[ selection_target[ (offset + thread) * MODEL_SIZE + i]];
	}

	float3 s_mean = make_float3(0,0,0);
	float3 t_mean = make_float3(0,0,0);

	for (int i = 0; i < MODEL_SIZE; ++i)
	{
		s_mean += vs[i];
		t_mean += vt[i];
	}
	s_mean /= MODEL_SIZE;
	t_mean /= MODEL_SIZE;

	for (int i = 0; i < MODEL_SIZE; ++i)
	{
		vs[i] -= s_mean;
		vt[i] -= t_mean;
	}

	float3 rs_2 = vs[2] - vs[0];
	float3 rs_1 = vs[1] - vs[0];
	float3 rt_2 = vt[2] - vt[0];
	float3 rt_1 = vt[1] - vt[0];
	float3 ns = cross(rs_2, rs_1);
	float3 ns_n = normalize(ns);
	float3 nt = cross(rt_2, rt_1);
	float3 nt_n = normalize(nt);
	float3 a = normalize(cross(ns,nt));

	//phi is the out-of-plane rotation along the intersection between the two planes formed by the vertex triplets vs and vt
	float sin_phi = norm(cross(ns_n, nt_n));
	float cos_phi = dot(ns_n, nt_n);

    float t = 1.0 - cos_phi;

    float ra00 = cos_phi + a.x*a.x*t;
    float ra11 = cos_phi + a.y*a.y*t;
    float ra22 = cos_phi + a.z*a.z*t;
    float tmp1 = a.x*a.y*t;
    float tmp2 = a.z*sin_phi;
    float ra10 = tmp1 + tmp2;
    float ra01 = tmp1 - tmp2;
    tmp1 = a.x*a.z*t;
    tmp2 = a.y*sin_phi;
    float ra20 = tmp1 - tmp2;
    float ra02 = tmp1 + tmp2;
    tmp1 = a.y*a.z*t;
    tmp2 = a.x*sin_phi;
    float ra21 = tmp1 + tmp2;
    float ra12 = tmp1 - tmp2;

    float S = 0;
    float C = 0;
    for (int i = 0; i < MODEL_SIZE; ++i)
	{
		float3 vsr = vs[i]*cos_phi + cross(a,vs[i])*sin_phi + a * ( dot(a,vs[i]) * (1.0f-cos_phi) );
		S += dot(cross(vt[i], vsr), nt_n);
		C += dot(vt[i], vsr);
	}

	float sin_theta = S/sqrt(S*S + C*C);
	float cos_theta = C/sqrt(S*S + C*C);

	t = 1.0 - cos_theta;

	float3 b = nt_n;

    float rb00 = cos_theta + b.x*b.x*t;
    float rb11 = cos_theta + b.y*b.y*t;
    float rb22 = cos_theta + b.z*b.z*t;
    tmp1 = b.x*b.y*t;
    tmp2 = b.z*sin_theta;
    float rb10 = tmp1 + tmp2;
    float rb01 = tmp1 - tmp2;
    tmp1 = b.x*b.z*t;
    tmp2 = b.y*sin_theta;
    float rb20 = tmp1 - tmp2;
    float rb02 = tmp1 + tmp2;
    tmp1 = b.y*b.z*t;
    tmp2 = b.x*sin_theta;
    float rb21 = tmp1 + tmp2;
    float rb12 = tmp1 - tmp2;

	float R00 = ra00*rb00 + ra01*rb10 + ra02*rb20; float R01 = ra00*rb01 + ra01*rb11 + ra02*rb21; float R02 = ra00*rb02 + ra01*rb12 + ra02*rb22;
	float R10 = ra10*rb00 + ra11*rb10 + ra12*rb20; float R11 = ra10*rb01 + ra11*rb11 + ra12*rb21; float R12 = ra10*rb02 + ra11*rb12 + ra12*rb22;
	float R20 = ra20*rb00 + ra21*rb10 + ra22*rb20; float R21 = ra20*rb01 + ra21*rb11 + ra22*rb21; float R22 = ra20*rb02 + ra21*rb12 + ra22*rb22;


	RT a_solution;
	a_solution.R[0] = make_float3(R00, R10, R20);
	a_solution.R[1] = make_float3(R01, R11, R21);
	a_solution.R[2] = make_float3(R02, R12, R22);

	// solve for the translation vector
	// T = Ym - R*Xm
	float3 T = t_mean - make_float3(R00 * s_mean.x + R01 * s_mean.y + R02 * s_mean.z,
									R10 * s_mean.x + R11 * s_mean.y + R12 * s_mean.z,
									R20 * s_mean.x + R21 * s_mean.y + R22 * s_mean.z);
	a_solution.T = T;
	candidate_solutions[thread] = a_solution;
}


__global__ void computeFitnessInverse(
	float * score,
	const RT* candidate_solutions,
	const float3* vertices_source,
	const uint num_verts_source,
	const float3* vertices_target,
	const uint num_verts_target,
	const float error_tolerance_dist,
	const float reject_model_if_below_this)
{

	uint thread = blockIdx.x *blockDim.x + threadIdx.x;
	RT warp = candidate_solutions[thread];
	uint inlier_count = 0;
	float fitness = 0;

	for (uint i = 0; i < num_verts_source; ++i)
	{
		float smallest_score = HUGE_VALF;

		for (uint j = 0; j < num_verts_target; ++j)
		{

			float3 X = vertices_source[i];
			float3 Y = vertices_target[j];

			float3 Yf = make_float3(-warp.T.x + ( warp.R[0].x * X.x + warp.R[0].y *X.y + warp.R[0].z * X.z),
									-warp.T.y + ( warp.R[1].x * X.x + warp.R[1].y *X.y + warp.R[1].z * X.z),
									-warp.T.z + ( warp.R[2].x * X.x + warp.R[2].y *X.y + warp.R[2].z * X.z) );

			float this_score = dist2(Yf.x-Y.x, Yf.y-Y.y, Yf.z-Y.z);

			if (this_score < smallest_score)
			{
				smallest_score = this_score;
			}
		}
		if (smallest_score < error_tolerance_dist)
		{
			fitness+=smallest_score;
			inlier_count++;
		}
	}
	score[thread] = ((float(inlier_count)/float(num_verts_source)) < reject_model_if_below_this) ? 1e10 : fitness/float(inlier_count);
}

__global__ void computeFitnessInverse(
	float * score,
	const RT* candidate_solutions,
	const float* vertices_source,
	const uint num_verts_source,
	const float* vertices_target,
	const uint num_verts_target,
	const float error_tolerance_dist,
	const float reject_model_if_below_this)
{

	uint thread = blockIdx.x *blockDim.x + threadIdx.x;
	RT warp = candidate_solutions[thread];
	uint inlier_count = 0;
	float fitness = 0;

	for (uint i = 0; i < num_verts_source; ++i)
	{
		float smallest_score = HUGE_VALF;

		for (uint j = 0; j < num_verts_target; ++j)
		{

			float3 X = make_float3(vertices_source[i*3+0], vertices_source[i*3+1], vertices_source[i*3+2]);
			float3 Y = make_float3(vertices_target[j*3+0], vertices_target[j*3+1], vertices_target[j*3+2]);

			float3 Yf = make_float3(-warp.T.x + ( warp.R[0].x * X.x + warp.R[0].y *X.y + warp.R[0].z * X.z),
									-warp.T.y + ( warp.R[1].x * X.x + warp.R[1].y *X.y + warp.R[1].z * X.z),
									-warp.T.z + ( warp.R[2].x * X.x + warp.R[2].y *X.y + warp.R[2].z * X.z) );

			float this_score = dist2(Yf.x-Y.x, Yf.y-Y.y, Yf.z-Y.z);

			if (this_score < smallest_score)
			{
				smallest_score = this_score;
			}
		}
		if (smallest_score < error_tolerance_dist)
		{
			fitness+=smallest_score;
			inlier_count++;
		}
	}
	score[thread] = ((float(inlier_count)/float(num_verts_source)) < reject_model_if_below_this) ? 1e10 : fitness/float(inlier_count);
}

__global__ void computeFitness(
	float * score,
	const RT* candidate_solutions,
	const float3* vertices_source,
	const uint num_verts_source,
	const float3* vertices_target,
	const uint num_verts_target,
	const float error_tolerance_dist,
	const float reject_model_if_below_this)
{
	uint thread = blockIdx.x *blockDim.x + threadIdx.x;
	RT warp = candidate_solutions[thread];
	uint inlier_count = 0;
	float fitness = 0;

	for (uint i = 0; i < num_verts_source; ++i)
	{
		float smallest_score = HUGE_VALF;

		for (uint j = 0; j < num_verts_target; ++j)
		{
			float3 X = vertices_source[i];
			float3 Y = vertices_target[j];

			float3 Yf = make_float3(warp.T.x + ( warp.R[0].x * X.x + warp.R[1].x *X.y + warp.R[2].x * X.z),
									warp.T.y + ( warp.R[0].y * X.x + warp.R[1].y *X.y + warp.R[2].y * X.z),
									warp.T.z + ( warp.R[0].z * X.x + warp.R[1].z *X.y + warp.R[2].z * X.z) );

			float this_score = dist2(Yf.x-Y.x, Yf.y-Y.y, Yf.z-Y.z);

			if (this_score < smallest_score)
			{
				smallest_score = this_score;
			}
		}
		if (smallest_score < error_tolerance_dist)
		{
			fitness+=smallest_score;
			inlier_count++;
		}
	}
	score[thread] = ((float(inlier_count)/float(num_verts_source)) < reject_model_if_below_this) ? 1e10 : fitness/float(inlier_count);
}

__global__ void computeFitness(
	float * score,
	const RT* candidate_solutions,
	const float* vertices_source,
	const uint num_verts_source,
	const float* vertices_target,
	const uint num_verts_target,
	const float error_tolerance_dist,
	const float reject_model_if_below_this)
{
	uint thread = blockIdx.x *blockDim.x + threadIdx.x;
	RT warp = candidate_solutions[thread];
	uint inlier_count = 0;
	float fitness = 0;

	for (uint i = 0; i < num_verts_source; ++i)
	{
		float smallest_score = HUGE_VALF;

		for (uint j = 0; j < num_verts_target; ++j)
		{
			float3 X = make_float3(vertices_source[i*3+0], vertices_source[i*3+1], vertices_source[i*3+2]);
			float3 Y = make_float3(vertices_target[j*3+0], vertices_target[j*3+1], vertices_target[j*3+2]);

			float3 Yf = make_float3(warp.T.x + ( warp.R[0].x * X.x + warp.R[1].x *X.y + warp.R[2].x * X.z),
									warp.T.y + ( warp.R[0].y * X.x + warp.R[1].y *X.y + warp.R[2].y * X.z),
									warp.T.z + ( warp.R[0].z * X.x + warp.R[1].z *X.y + warp.R[2].z * X.z) );

			float this_score = dist2(Yf.x-Y.x, Yf.y-Y.y, Yf.z-Y.z);

			if (this_score < smallest_score)
			{
				smallest_score = this_score;
			}
		}
		if (smallest_score < error_tolerance_dist)
		{
			fitness+=smallest_score;
			inlier_count++;
		}
	}
	score[thread] = ((float(inlier_count)/float(num_verts_source)) < reject_model_if_below_this) ? 1e10 : fitness/float(inlier_count);
}


	RANSAC::RANSAC( 
	const uint max_iterations,
	const uint max_iterations_per_batch,
	const float error_tolerance_dist,
	const float inlier_ratio_to_accept_solution,
	thrust::device_vector<float3> vertices_source,
	thrust::device_vector<float3> vertices_target,
	thrust::device_vector<float3> vertices_fitness_source,
	thrust::device_vector<float3> vertices_fitness_target):
		_numThreadsPerBlock 				(256),
		_current_best_score 				(HUGE_VALF),
		_current_best_model_pair_index		(0),
		_time_to_stop 						(false),
		_iteration_counter					(0),
		_max_iterations						(max_iterations),
		_max_iterations_per_batch			(max_iterations_per_batch),
		_error_tolerance_dist				(error_tolerance_dist),
		_inlier_ratio_to_accept_solution	(inlier_ratio_to_accept_solution),
		_vertices_source					(vertices_source),
		_vertices_target					(vertices_target),
		_vertices_fitness_source			(vertices_fitness_source),
		_vertices_fitness_target			(vertices_fitness_target)
{
	//check that the number of threads per block doesn't exceed the batch size
	if (_numThreadsPerBlock > max_iterations_per_batch)
		_numThreadsPerBlock = max_iterations_per_batch;

	//random number generator
	std::random_device rand_dev;
	std::mt19937 generator(rand_dev());

	//host side containers for the indices to be used by ransac
	_vert_subset_source_host.resize(_max_iterations * MODEL_SIZE);
	_vert_subset_target_host.resize(_max_iterations * MODEL_SIZE);

	//sample randomly, while avoiding repeated points in the model subsets
	std::uniform_int_distribution<uint> distr_source( 0, vertices_source.size()-1);

	for (uint i = 0; i < _max_iterations; i++)
	{
		std::set<uint> a_subset;
		while(a_subset.size()<MODEL_SIZE)
			a_subset.insert(distr_source(generator));

		for (int j = 0; j < MODEL_SIZE; ++j)
			_vert_subset_source_host[i*MODEL_SIZE + j] = *std::next(a_subset.begin(), j);
	}

	std::uniform_int_distribution<uint> distr_target( 0, vertices_target.size()-1);
	//sample randomly, while avoiding repeated points in the model subsets
	for (uint i = 0; i < _max_iterations; i++)
	{
		std::set<uint> a_subset;
		while(a_subset.size()<MODEL_SIZE)
			a_subset.insert(distr_target(generator));

		for (int j = 0; j < MODEL_SIZE; ++j)
			_vert_subset_target_host[i*MODEL_SIZE + j] = *std::next(a_subset.begin(), j);
	}


	_vert_subset_source = _vert_subset_source_host;
	_vert_subset_target = _vert_subset_target_host;

	_numBlocks = max(1,int(floorf(float(_max_iterations_per_batch)/_numThreadsPerBlock + 0.5f)));
	_numThreads = _numThreadsPerBlock*_numBlocks;

	_candidate_solutions.resize(_numThreads);
	_solution_scores.resize(_numThreads);
}

	RANSAC::~RANSAC()
	{}


	void RANSAC::doIteration()
	{

		float best_score_of_batch = 1e10f;

		dim3 dimGrid(_numBlocks);
		dim3 dimThreads = dim3(_numThreadsPerBlock);

		computeRT<<<dimGrid,dimThreads>>>(
			_iteration_counter,
			thrust::raw_pointer_cast( &_vertices_source[0]),
			thrust::raw_pointer_cast( &_vertices_target[0]),
			thrust::raw_pointer_cast( &_vert_subset_source[0]),
			thrust::raw_pointer_cast( &_vert_subset_target[0]),
			thrust::raw_pointer_cast( &_candidate_solutions[0]),
			_max_iterations);

	 	cudaGetLastError();


		computeFitnessInverse<<<dimGrid, dimThreads>>>(
			thrust::raw_pointer_cast(&_solution_scores[0]),
			thrust::raw_pointer_cast(&_candidate_solutions[0]),
			thrust::raw_pointer_cast(&_vertices_source[0]),
			_vertices_source.size(),
			thrust::raw_pointer_cast(&_vertices_target[0]),
			_vertices_target.size(),
			_error_tolerance_dist,
			_inlier_ratio_to_accept_solution
			);
	 	 (cudaGetLastError ());

		thrust::device_vector<float>::iterator iter_inv =
			thrust::min_element(_solution_scores.begin(), _solution_scores.end());
	 	 (cudaGetLastError ());

		int best_of_batch_inv = iter_inv - _solution_scores.begin();
		float best_score_of_batch_inv = _solution_scores[best_of_batch_inv];

		computeFitness<<<dimGrid, dimThreads>>>(
			thrust::raw_pointer_cast(&_solution_scores[0]),
			thrust::raw_pointer_cast(&_candidate_solutions[0]),
			thrust::raw_pointer_cast(&_vertices_source[0]),
			_vertices_source.size(),
			thrust::raw_pointer_cast(&_vertices_target[0]),
			_vertices_target.size(),
			_error_tolerance_dist,
			_inlier_ratio_to_accept_solution
			);
	 	 (cudaGetLastError ());

		thrust::device_vector<float>::iterator iter_fwd =
			thrust::min_element(_solution_scores.begin(), _solution_scores.end());
	 	 (cudaGetLastError ());

	 	int best_of_batch_fwd = iter_fwd - _solution_scores.begin();
		float best_score_of_batch_fwd = _solution_scores[best_of_batch_fwd];


		if(best_score_of_batch_fwd < 1e6 && best_score_of_batch_inv <1e6)
		{
			best_score_of_batch = 0.5*(best_score_of_batch_fwd + best_score_of_batch_inv);
		}

		if(best_score_of_batch < _current_best_score)
		{
			_current_best_score = best_score_of_batch;//_solution_scores[best_of_batch];
			// printf("current best: %f\n",_current_best_score);

			_iteration_mutex.lock();
			_current_best_model_pair_index = _iteration_counter + best_of_batch_fwd;
			_iteration_mutex.unlock();

			_transformation_mutex.lock();
			_current_best_solution = _candidate_solutions[best_of_batch_fwd];
			_transformation_mutex.unlock();
		}

		_iteration_mutex.lock();
		_iteration_counter+= _numThreads;
		_iteration_mutex.unlock();

	}

	void RANSAC::GetCurrentSolution( RT &solution)
	{
		_transformation_mutex.lock();
		solution = _current_best_solution;
		_transformation_mutex.unlock();
	}

	uint3 RANSAC::GetSourceModel()
	{

		return make_uint3(  _vert_subset_source_host[MODEL_SIZE * _current_best_model_pair_index + 0],
							_vert_subset_source_host[MODEL_SIZE * _current_best_model_pair_index + 1],
							_vert_subset_source_host[MODEL_SIZE * _current_best_model_pair_index + 2]);
	}

	uint3 RANSAC::GetTargetModel()
	{
		return make_uint3(  _vert_subset_target_host[MODEL_SIZE * _current_best_model_pair_index + 0],
							_vert_subset_target_host[MODEL_SIZE * _current_best_model_pair_index + 1],
							_vert_subset_target_host[MODEL_SIZE * _current_best_model_pair_index + 2]);
	}

	bool RANSAC::has_finished()
	{
		return (_iteration_counter >= _max_iterations);
	}


	float RANSAC::GetCurrentScore()
	{
		return _current_best_score;
	}

	void RANSAC::start()
	{
		while( (_iteration_counter < _max_iterations) && !_time_to_stop)
			doIteration();
	}

	void RANSAC::stop()
	{
		_iteration_mutex.lock();
		_time_to_stop = true;
		_iteration_mutex.unlock();
	}

	void RANSAC::reset()
	{
		_current_best_score=HUGE_VALF;
		_time_to_stop=false;
		_iteration_mutex.lock();
		_iteration_counter= 0;
		_iteration_mutex.unlock();

	}
