# cuRANSAC
An implementation of RANSAC (RANdom SAmple Consensus)for absolute pose estimation of graph-like representations, using CUDA. 

## How it works
The input is a set of nodes connected by edges (a graph).
The program then does the following:
 - Randomly select N sets of unique triplets from the vertices of the source graph 
 - Randomly select N sets of unique triplets from the vertices of the target graph 
 - Compute N rigid-body poses to align source triplet to the target triplet via Horn's 3-point method (in parallel)
 - Score the N poses by the distance between edge midpoints in the transformed source and target graph (in parallel)
 - Get the transformation corresponding to the minimum score.

## How to use it
The program is run from the command-line with two graphs as arguments. The graphs should be provided in the visualization toolkit (VTK) format vtp. The .vtp files can also be opened using the paraview program, which may possibly be made available from your linux distribution's software repositories, or downloadable from http://www.paraview.org/ .

Internally the representation of a graph is simply a vector of types:
 - float3 for vertices (containing components float x, float y, and float z) and 
 - uint2 for edges (containing components uint x, and uint y) indicating the indices to pairs of vertices that have an edge between them.

example - running the program:
```sh
./cuRANSAC ../graph_data/graph1.vtp ../graph_data/graph2.vtp 
./cuRANSAC ../graph_data/graph1.vtp ../graph_data/graph3.vtp 
./cuRANSAC ../graph_data/graph2.vtp ../graph_data/graph3.vtp 
```
etc.

The OpenGL rendering window displays the two graphs as red and green. The visualization is done in a separate thread and is updated as the RANSAC iterations progress.
The bright triangles represent the triplets picked from each model, used to perform the current best estimate for the pose

You may navigate the virtual camera around using keyboard controls mapped to  W,A,S,D,Ctrl,shift and mouselook (scroll for zoom)

You may also:
hard-reset the RANSAC pose-estimation with new random seed by pressing "H"
reset the the RANSAC pose-estimation to its first iteration by pressing "R"

You may additionally modify this software to suit your needs according to the LICENCE.txt file

## How to REALLY use it
Include RANSAC.hpp in your source code and define nodes_source_device and nodes_source_target as vectors of type ```thrust::device_vector<float3>```. 
You may re-use these vectors for the fitness scoring too. Alternatively you may use the functions declared in the graph_tools.hpp header to compute the midpoints along the connected vertices and use the distance between these as a fitness measure, instead. Then create and initialize the RANSAC object as such, for example:

```cpp
const uint max_iterations = 1<<16;
const uint max_iterations_per_batch = 1<<8;
const float error_tolerance_dist = 0.8f;
const float inlier_ratio_to_accept_solution = 0.1; //get at least 10% within limits?
RANSAC* ransac_object = new RANSAC( max_iterations,
                      max_iterations_per_batch,
                      error_tolerance_dist,
                      inlier_ratio_to_accept_solution,
                      nodes_source_device,
                      nodes_target_device,
                      midpoints_source_device,
                      midpoints_target_device);

ransac_object->start();

    while(!ransac_object->has_finished())
    {
   	    usleep(1000);
    }
    ransac_object->stop();
    RT sol;
  	ransac_object->GetCurrentSolution(sol);
```
RT is a structure containing a rotation matrix and translation vector, defined in the RANSAC.hpp header. See the file read_graphs.cpp for a complete example.

## Compilation

from the build directory:

```bash
cmake ..
make
```

Compilation depends on:
glfw3
cuda (probably => 6.5 required, maybe even 7.0)
glm
vtk (tested 5.6 6.0 and 7.0)

There may be issues with different versions of GCC, CUDA your current GPU, and other things. 
Look into the CMakeLists.txt and modify it according to your needs. 
Open an issue or submit a pull request if you find a bug / enhancement
