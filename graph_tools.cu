#include "graph_tools.hpp"

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkCellArray.h>

#include <iomanip>
#include <iostream>
#include <iterator>

struct has_no_attribute
{
  __host__ __device__
  bool operator()(const int x)
  {
    return x == 0;
  }
  __host__ __device__
  bool operator()(const uint x)
  {
    return x == 0;
  }
};

__host__ __device__ __forceinline__
uint divUp(uint a, uint b){
    return (a+b-1)/b;
}

__global__ 
void remap_edge_indices(uint* edges, const uint* adjustment, const uint num_edges)
{
	uint thread = blockIdx.x *blockDim.x + threadIdx.x;
	if(thread>=num_edges) return;

	uint* this_edge = &edges[thread*2 + 0];
	uint2 adjust_by = make_uint2( adjustment[*this_edge], adjustment[*(this_edge+1)]);

	*this_edge 		-= adjust_by.x;
	*(this_edge+1) 	-= adjust_by.y;
}


__global__ 
void remap_edge_indices(uint2* edges, const uint* adjustment, const uint num_edges)
{
	uint thread = blockIdx.x *blockDim.x + threadIdx.x;
	if(thread>=num_edges) return;

	uint2 this_edge = edges[thread];
	uint2 adjust_by = make_uint2( adjustment[this_edge.x], adjustment[this_edge.y]);

	this_edge.x	-= adjust_by.x;
	this_edge.y	-= adjust_by.y;

  edges[thread] = this_edge;
}

struct increment : public thrust::unary_function<uint,uint>
{
  __host__ __device__
  uint operator()(uint x) { return ++x; }
};

template <typename Iterator>
class strided_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }
    
    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};


template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
	typedef typename Vector::value_type T;
	std::cout << "	" << std::setw(20) << name << "	";
	thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
	std::cout << std::endl;
}

// dense histogram using binary search
template <typename Vector1, typename Vector2>
void dense_histogram(const Vector1& input, Vector2& histogram)
{
	typedef typename Vector1::value_type ValueType; // input value type
	typedef typename Vector2::value_type IndexType; // histogram index type

	// copy input data (could be skipped if input is allowed to be modified)
	thrust::device_vector<ValueType> data(input);
	// sort data to bring equal elements together
	thrust::sort(data.begin(), data.end());
	
	// number of histogram bins is equal to the maximum value plus one
	IndexType num_bins = data.back() + 1;

	// resize histogram storage
	histogram.resize(num_bins);
	
	// find the end of each bin of values
	thrust::counting_iterator<IndexType> search_begin(0);
	thrust::upper_bound(data.begin(), data.end(),
						search_begin, search_begin + num_bins,
						histogram.begin());

	// compute the histogram by taking differences of the cumulative histogram
	thrust::adjacent_difference(histogram.begin(), histogram.end(),
								histogram.begin());
}

// sparse histogram using reduce_by_key
template 	<typename Vector1, typename Vector2, typename Vector3>
void sparse_histogram(const Vector1& input,
							Vector2& histogram_values,
							Vector3& histogram_counts)
{
	typedef typename Vector1::value_type ValueType; // input value type
	typedef typename Vector3::value_type IndexType; // histogram index type

	// copy input data (could be skipped if input is allowed to be modified)
	thrust::device_vector<ValueType> data(input);
	
	// sort data to bring equal elements together
	thrust::sort(data.begin(), data.end());

	// number of histogram bins is equal to number of unique values (assumes data.size() > 0)
	IndexType num_bins = thrust::inner_product(data.begin(), data.end() - 1,
											 data.begin() + 1,
											 IndexType(1),
											 thrust::plus<IndexType>(),
											 thrust::not_equal_to<ValueType>());

	// resize histogram storage
	histogram_values.resize(num_bins);
	histogram_counts.resize(num_bins);
	
	// compact find the end of each bin of values
	thrust::reduce_by_key(data.begin(), data.end(),
						thrust::constant_iterator<IndexType>(1),
						histogram_values.begin(),
						histogram_counts.begin());
}


__global__  
void midpoint_GPU( float* midpoints,  const float* vertices, const uint* edges, const int num_edges)
{
    uint thread = blockIdx.x *blockDim.x + threadIdx.x;
    if(thread>=num_edges) return;
    
    midpoints[thread*3+0] =	0.5f*vertices[ edges[thread*2 + 0]*3+0] + 0.5*vertices[edges[thread*2+1]*3+0];
	midpoints[thread*3+1] =	0.5f*vertices[ edges[thread*2 + 0]*3+1] + 0.5*vertices[edges[thread*2+1]*3+1];
	midpoints[thread*3+2] =	0.5f*vertices[ edges[thread*2 + 0]*3+2] + 0.5*vertices[edges[thread*2+1]*3+2];
}


__global__  
void midpoint_GPU( float3* midpoints,  const float3* vertices, const uint2* edges, const int num_edges)
{
    uint thread = blockIdx.x *blockDim.x + threadIdx.x;
    if(thread>=num_edges) return;
    
    midpoints[thread] = make_float3( 	0.5f*vertices[ edges[thread].x].x + 0.5f*vertices[edges[thread].y].x,
    									0.5f*vertices[ edges[thread].x].y + 0.5f*vertices[edges[thread].y].y,
    									0.5f*vertices[ edges[thread].x].z + 0.5f*vertices[edges[thread].y].z);
}


void compute_indegree(thrust::device_vector<uint> &vertex_indegree, const thrust::device_vector<uint2> &edges, const int num_verts)
{
	uint* edges_flat_ptr = (uint*)thrust::raw_pointer_cast(&edges[0]);

	thrust::device_vector<uint> edges_flat(edges_flat_ptr, edges_flat_ptr + edges.size()*2);
	compute_indegree(vertex_indegree, edges_flat, num_verts);
}
	


void compute_indegree(thrust::device_vector<uint> &vertex_indegree, const thrust::device_vector<uint> &edges, const int num_verts)
{
	// // using sparse histogram method
	thrust::device_vector<uint> histogram_values;
	thrust::device_vector<uint> histogram_counts;

	// counts the number of times a vertex is referred to in the edge list
	sparse_histogram(edges, histogram_values, histogram_counts);
	
	//since some vertices may not be referenced, we may need a larger container to make it a proper vertex attribute list
	vertex_indegree.resize(num_verts);

	//init as zero
	thrust::fill(vertex_indegree.begin(), vertex_indegree.end(), 0);
	
	//scatter the histogram counts into the buckets corresponding to the vertex index
	thrust::scatter(histogram_counts.begin(), histogram_counts.end(),histogram_values.begin(), vertex_indegree.begin());

	// std::cout << "Sparse Histogram" << std::endl;
	// print_vector("histogram values", histogram_values);
	// print_vector("histogram counts", histogram_counts);
	// std::cout << "histogram size : " << histogram_counts.size() << std::endl;
	// std::cout << "number of vertices : " << num_verts << std::endl;
	// print_vector("vertex_indegree", vertex_indegree);

	// // using dense histogram method
	// {
	// 	std::cout << "Dense Histogram" << std::endl;
	// 	thrust::device_vector<int> histogram;
	// 	dense_histogram(edges, histogram);
	// 	print_vector("histogram", histogram);
	// 	std::cout << "Numel : " << histogram.size()<< std::endl;
	// }
}

uint delete_unreferenced_vertices(thrust::device_vector<float> &vertices, thrust::device_vector<uint> &edges, const thrust::device_vector<uint> &vertex_indegree)
{
    typedef thrust::device_vector<float>::iterator float_dev_iterator;
    typedef thrust::device_vector<uint>::iterator uint_dev_iterator;

	thrust::counting_iterator<uint> initial_vertex_list(0);
	increment add_one;
 	
 	thrust::device_vector<uint> adjustment; 
 	adjustment.resize(vertex_indegree.size());
 	thrust::fill(adjustment.begin(), adjustment.end(),0u);

	thrust::transform_if(vertex_indegree.begin(), vertex_indegree.end(), adjustment.begin(), add_one, has_no_attribute());

    thrust::exclusive_scan(adjustment.begin(), adjustment.end(), adjustment.begin()); // in-place scan
	
    uint num_edges = edges.size()/2; 
    dim3 threads(256);
    dim3 blocks(divUp(num_edges,threads.x));

			
	remap_edge_indices<<<blocks,threads>>>( thrust::raw_pointer_cast(&edges[0]),
											thrust::raw_pointer_cast(&adjustment[0]),
											num_edges);

    // create strided_range with indices [0,2,4,6]
    strided_range<float_dev_iterator> all_components(vertices.begin() + 0, vertices.end(), 1);
    strided_range<float_dev_iterator> x_components(vertices.begin() + 0, vertices.end(), 3);
    strided_range<float_dev_iterator> y_components(vertices.begin() + 1, vertices.end(), 3);
    strided_range<float_dev_iterator> z_components(vertices.begin() + 2, vertices.end(), 3);
    
    strided_range<float_dev_iterator>::iterator new_end_x = thrust::remove_if(x_components.begin(), x_components.end(), vertex_indegree.begin(), has_no_attribute());
    strided_range<float_dev_iterator>::iterator new_end_y = thrust::remove_if(y_components.begin(), y_components.end(), vertex_indegree.begin(), has_no_attribute());
    strided_range<float_dev_iterator>::iterator new_end_z = thrust::remove_if(z_components.begin(), z_components.end(), vertex_indegree.begin(), has_no_attribute());

    vertices.resize( (new_end_z - all_components.begin())*3);

    return vertices.size();
}


uint delete_unreferenced_vertices(thrust::device_vector<float3> &vertices, thrust::device_vector<uint2> &edges, const thrust::device_vector<uint> &vertex_indegree)
{
  typedef thrust::device_vector<float3>::iterator float3_dev_iterator;

	thrust::counting_iterator<uint> initial_vertex_list(0);
	increment add_one;
  

 	thrust::device_vector<uint> adjustment; 
 	adjustment.resize(vertex_indegree.size());
 	thrust::fill(adjustment.begin(), adjustment.end(),0u);

	thrust::transform_if(vertex_indegree.begin(), vertex_indegree.end(), adjustment.begin(), add_one, has_no_attribute());

  thrust::exclusive_scan(adjustment.begin(), adjustment.end(), adjustment.begin()); // in-place scan
	
  uint num_edges = edges.size(); 
  dim3 threads(256);
  dim3 blocks(divUp(num_edges,threads.x));

			
	remap_edge_indices<<<blocks,threads>>>( 
                      thrust::raw_pointer_cast(&edges[0]),
											thrust::raw_pointer_cast(&adjustment[0]),
											num_edges);
  (cudaGetLastError ());
  // create strided_range with indices [0,2,4,6]
  strided_range<float3_dev_iterator> all_components(vertices.begin(), vertices.end(), 1);
  // strided_range<float3_dev_iterator> x_components(vertices.begin() + 0, vertices.end(), 3);
  // strided_range<float3_dev_iterator> y_components(vertices.begin() + 1, vertices.end(), 3);
  // strided_range<float3_dev_iterator> z_components(vertices.begin() + 2, vertices.end(), 3);
  
  strided_range<float3_dev_iterator>::iterator new_end = thrust::remove_if(all_components.begin(), all_components.end(), vertex_indegree.begin(), has_no_attribute());
  // strided_range<float3_dev_iterator>::iterator new_end_y = thrust::remove_if(y_components.begin(), y_components.end(), vertex_indegree.begin(), has_no_attribute());
  // strided_range<float3_dev_iterator>::iterator new_end_z = thrust::remove_if(z_components.begin(), z_components.end(), vertex_indegree.begin(), has_no_attribute());
  
  vertices.resize( (new_end - all_components.begin()));

  return vertices.size();
}



void compute_edge_midpoints(thrust::device_vector<float> &midpoints, const thrust::device_vector<float> &vertices, const thrust::device_vector<uint> &edges)
{
  int num_edges = edges.size()/2; 
  midpoints.clear();
  midpoints.resize(num_edges*3);
  
  if(num_edges>0)
  {
    dim3 threads(256);
    dim3 blocks(divUp(num_edges,threads.x));
    midpoint_GPU<<<blocks, threads>>>(
        thrust::raw_pointer_cast(&midpoints[0]), 
        thrust::raw_pointer_cast(&vertices[0]), 
        thrust::raw_pointer_cast(&edges[0]),
        num_edges);

  }
}

void compute_edge_midpoints(thrust::device_vector<float3> &midpoints, const thrust::device_vector<float3> &vertices, const thrust::device_vector<uint2> &edges)
{
  int num_edges = edges.size(); 
  midpoints.clear();
  midpoints.resize(num_edges);
  
  if(num_edges>0)
  {
    dim3 threads(256);
    dim3 blocks(divUp(num_edges,threads.x));
    midpoint_GPU<<<blocks, threads>>>(
        thrust::raw_pointer_cast(&midpoints[0]), 
        thrust::raw_pointer_cast(&vertices[0]), 
        thrust::raw_pointer_cast(&edges[0]),
        num_edges);

  }
}


void write_graph(const thrust::host_vector<float> &vertices, const thrust::host_vector<uint> &edges, const std::string &filename)
{
	   vtkSmartPointer<vtkPoints> lineEndPoints = vtkSmartPointer<vtkPoints>::New();
       vtkSmartPointer<vtkPolyData> line = vtkSmartPointer<vtkPolyData>::New();
       line->Allocate();

       for ( uint i = 0; i < vertices.size()/3; ++i )
       { 
         lineEndPoints->InsertNextPoint(vertices[i*3 + 0], vertices[i*3 + 1], vertices[i*3 + 2]);
       }
         line->SetPoints(lineEndPoints);


       for (int i = 0; i < edges.size()/2; ++i)
       {

         vtkIdType connectivity[2];
         connectivity[0] = edges[i*2 + 0];
         connectivity[1] = edges[i*2 + 1];
         line->InsertNextCell(VTK_LINE,2,connectivity); //Connects the first and fourth point we inserted into a line
       }

	   vtkSmartPointer<vtkXMLPolyDataWriter> line_writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    
       line_writer->SetFileName(filename.c_str());
        
      #if VTK_MAJOR_VERSION <= 5
             line_writer->SetInput(line);
      #else
             line_writer->SetInputData(line);
      #endif        
       line_writer->SetDataModeToBinary();
       line_writer->Write();
}

void read_graph(thrust::host_vector<float3> &vertices, thrust::host_vector<uint2> &edges, const std::string &filename)
{
    vtkSmartPointer<vtkXMLPolyDataReader> reader;
    vtkPolyData* input_graph;
    vtkCellArray* lines;

    vtkIdType npts = 0;
    vtkIdType* pt = nullptr;
    uint id = 0;

    reader = vtkSmartPointer<vtkXMLPolyDataReader>::New();

    reader->SetFileName(filename.c_str());
    reader->Update();
    input_graph = reader->GetOutput();
    uint num_nodes = input_graph->GetNumberOfPoints();
    uint num_edges = input_graph->GetNumberOfLines();
    vertices.resize(num_nodes);
    edges.resize(num_edges);

    for(id = 0; id < num_nodes; id++)
    {
      vertices[id].x = static_cast <float> (input_graph->GetPoints()->GetData()->GetTuple3(id)[0]); 
      vertices[id].y = static_cast <float> (input_graph->GetPoints()->GetData()->GetTuple3(id)[1]); 
      vertices[id].z = static_cast <float> (input_graph->GetPoints()->GetData()->GetTuple3(id)[2]); 
    }
    lines = input_graph->GetLines();

    id = 0;
    lines->InitTraversal();
    do
    {
      lines->GetNextCell(npts, pt);
      if(npts)
      {
        edges[id].x = pt[0]; 
        edges[id].y = pt[1]; 
      }
      id++;
    }
    while(npts);



}