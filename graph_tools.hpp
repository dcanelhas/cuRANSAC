#ifndef GRAPH_TOOLS_HPP
#define GRAPH_TOOLS_HPP
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

void compute_indegree(thrust::device_vector<uint> &vertex_indegree, const thrust::device_vector<uint> &edges, const int num_verts);
void compute_indegree(thrust::device_vector<uint> &vertex_indegree, const thrust::device_vector<uint2> &edges, const int num_verts);

void compute_edge_midpoints(thrust::device_vector<float> &midpoints, const thrust::device_vector<float> &vertices, const thrust::device_vector<uint> &edges);
void compute_edge_midpoints(thrust::device_vector<float3> &midpoints, const thrust::device_vector<float3> &vertices, const thrust::device_vector<uint2> &edges);

uint delete_unreferenced_vertices(thrust::device_vector<float> &vertices, thrust::device_vector<uint> &edges, const thrust::device_vector<uint> &vertex_indegree);
uint delete_unreferenced_vertices(thrust::device_vector<float3> &vertices, thrust::device_vector<uint2> &edges, const thrust::device_vector<uint> &vertex_indegree);

void write_graph(const thrust::host_vector<float> &vertices, const thrust::host_vector<uint> &edges, const std::string &filename);
void read_graph( thrust::host_vector<float3> &vertices, thrust::host_vector<uint2> &edges, const std::string &filename);

#endif
