
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>

// GL Extension Wrangler
#include <cmath>
#include <thread>
#include <random>
#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "readfile.hpp"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define GLM_FORCE_RADIANS
#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/gtc/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale, glm::perspective
#include <glm/gtc/type_ptr.hpp> //glm::value_ptr

#include "graph_tools.hpp"
#include "RANSAC.hpp"


enum COLOR { RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW, WHITE, BLACK };

void set_color(COLOR c, float opacity, GLuint shader_program)
{
  switch(c)
  {
    case RED:
      glUniform1f(glGetUniformLocation(shader_program ,"R_val"), 0.5);
      glUniform1f(glGetUniformLocation(shader_program ,"G_val"), 0.1f);
      glUniform1f(glGetUniformLocation(shader_program ,"B_val"), 0.1f);
      glUniform1f(glGetUniformLocation(shader_program ,"A_val"), opacity);
      break;
    case GREEN:
      glUniform1f(glGetUniformLocation(shader_program ,"R_val"), 0.1f);
      glUniform1f(glGetUniformLocation(shader_program ,"G_val"), 0.5f);
      glUniform1f(glGetUniformLocation(shader_program ,"B_val"), 0.1f);
      glUniform1f(glGetUniformLocation(shader_program ,"A_val"), opacity);
      break;
    case BLUE:
      glUniform1f(glGetUniformLocation(shader_program ,"R_val"), 0.1f);
      glUniform1f(glGetUniformLocation(shader_program ,"G_val"), 0.1f);
      glUniform1f(glGetUniformLocation(shader_program ,"B_val"), 0.5f);
      glUniform1f(glGetUniformLocation(shader_program ,"A_val"), opacity);
      break;
    case CYAN:
      glUniform1f(glGetUniformLocation(shader_program ,"R_val"), 0.1f);
      glUniform1f(glGetUniformLocation(shader_program ,"G_val"), 0.5f);
      glUniform1f(glGetUniformLocation(shader_program ,"B_val"), 0.5f);
      glUniform1f(glGetUniformLocation(shader_program ,"A_val"), opacity);
      break;
    case MAGENTA:
      glUniform1f(glGetUniformLocation(shader_program ,"R_val"), 1.0f);
      glUniform1f(glGetUniformLocation(shader_program ,"G_val"), 0.f);
      glUniform1f(glGetUniformLocation(shader_program ,"B_val"), 1.0f);
      glUniform1f(glGetUniformLocation(shader_program ,"A_val"), opacity);
      break;
    case YELLOW:
      glUniform1f(glGetUniformLocation(shader_program ,"R_val"), 1.0f);
      glUniform1f(glGetUniformLocation(shader_program ,"G_val"), 1.0f);
      glUniform1f(glGetUniformLocation(shader_program ,"B_val"), 0.0f);
      glUniform1f(glGetUniformLocation(shader_program ,"A_val"), opacity);
      break;
    case WHITE:
      glUniform1f(glGetUniformLocation(shader_program ,"R_val"), 1.0f);
      glUniform1f(glGetUniformLocation(shader_program ,"G_val"), 1.0f);
      glUniform1f(glGetUniformLocation(shader_program ,"B_val"), 1.0f);
      glUniform1f(glGetUniformLocation(shader_program ,"A_val"), opacity);
      break;
    case BLACK:
      glUniform1f(glGetUniformLocation(shader_program ,"R_val"), 0.0f);
      glUniform1f(glGetUniformLocation(shader_program ,"G_val"), 0.0f);
      glUniform1f(glGetUniformLocation(shader_program ,"B_val"), 0.0f);
      glUniform1f(glGetUniformLocation(shader_program ,"A_val"), opacity);
      break;
    default :
      break;
  }
}

// #define DEBUG_OFFSET

const GLuint w_width = 1024;
const GLuint w_height = 768;
// Camera
glm::vec3 cameraPos   = glm::vec3(10.0f, 0.0f,  -4.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f,  1.0f);
glm::vec3 cameraDown    = glm::vec3(0.0f, 1.0f,  0.0f);
GLfloat yaw    = -90.0f;
GLfloat pitch  = 0.0f;
GLfloat lastX  =  w_width  / 2.0;
GLfloat lastY  =  w_height / 2.0;
GLfloat fov =  42.0f;
bool keys[1024];

// Deltatime
GLfloat deltaTime = 0.0f; // Time between current frame and last frame
GLfloat lastFrame = 0.0f;   // Time of last frame

static void error_callback(int error, const char* description)
{
    std::cerr << description;
}

// Is called whenever a key is pressed/released via GLFW
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
    if (key >= 0 && key < 1024)
    {
        if (action == GLFW_PRESS)
            keys[key] = true;
        else if (action == GLFW_RELEASE)
            keys[key] = false;
    }
}

bool reset_key()
{
  return keys[GLFW_KEY_R];
}

bool hard_reset_key()
{
  return keys[GLFW_KEY_H];
}

void do_movement()
{
    // Camera controls
    GLfloat cameraSpeed = 5.0f * deltaTime;
    if (keys[GLFW_KEY_LEFT_CONTROL])
        cameraPos += cameraSpeed * cameraDown;
    if (keys[GLFW_KEY_LEFT_SHIFT])
        cameraPos -= cameraSpeed * cameraDown;
    if (keys[GLFW_KEY_W])
        cameraPos += cameraSpeed * cameraFront;
    if (keys[GLFW_KEY_S])
        cameraPos -= cameraSpeed * cameraFront;
    if (keys[GLFW_KEY_A])
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraDown)) * cameraSpeed;
    if (keys[GLFW_KEY_D])
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraDown)) * cameraSpeed;
}

bool firstMouse = true;
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    GLfloat xoffset = xpos - lastX;
    GLfloat yoffset = lastY - ypos; // Reversed since y-coordinates go from bottom to left
    lastX = xpos;
    lastY = ypos;

    GLfloat sensitivity = 0.05; // Change this value to your liking
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw   += xoffset;
    pitch += yoffset;

    // Make sure that when pitch is out of bounds, screen doesn't get flipped
    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(-yaw)) * cos(glm::radians(-pitch));
    front.y = sin(glm::radians(-pitch));
    front.z = sin(glm::radians(-yaw)) * cos(glm::radians(-pitch));
    cameraFront = glm::normalize(front);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (fov >= 41.0f && fov <= 44.0f)
        fov -= yoffset*0.02;
    if (fov <= 41.0f)
        fov = 41.0f;
    if (fov >= 44.0f)
        fov = 44.0f;
}



int main ( int argc, char *argv[] )
{
  cudaGLSetGLDevice(0);
  // Parse command line arguments

  float weight_a = 0.0;
  float weight_b = 0.0;

  std::string filename_source;
  std::string filename_target;

  if(argc == 3)
  {
    filename_source = argv[1];
    filename_target = argv[2];
  }
  else
  {
    std::cerr << "Usage (1): " << argv[0] << " source_graph_filename(.vtp) target_graph_filename(.vtp)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // start GL context and O/S window using the GLFW helper library
  glfwSetErrorCallback(error_callback);
  if( !glfwInit() )
    exit(EXIT_FAILURE);

  GLFWwindow* window = glfwCreateWindow (w_width, w_height, "Graphs", nullptr, nullptr);

  if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
  }

  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwMakeContextCurrent (window);

  // Set the required callback functions
  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetScrollCallback(window, scroll_callback);
  glfwSetKeyCallback(window, key_callback);

  // Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
  glewExperimental = GL_TRUE;
  glewInit();

  // Define the viewport dimensions
  glViewport(0, 0, w_width, w_height);

  //set blend mode
  glEnable( GL_BLEND );
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


  // Define containers and read graphs
  thrust::host_vector<float3> nodes_source; 
  thrust::host_vector<float3> nodes_target; 
  thrust::host_vector<uint2> edges_source;  
  thrust::host_vector<uint2> edges_target;  

  std::cout << "Reading graphs from file...";
  read_graph(nodes_source, edges_source, filename_source);
  read_graph(nodes_target, edges_target, filename_target);
  std::cout << "done.\n";

  thrust::device_vector<float3> nodes_source_device; 
  thrust::device_vector<float3> nodes_target_device; 
  thrust::device_vector<uint2> edges_source_device; 
  thrust::device_vector<uint2> edges_target_device; 
  nodes_source_device = nodes_source;
  nodes_target_device = nodes_target;
  edges_source_device = edges_source;
  edges_target_device = edges_target;

  uint num_nodes_source=nodes_source.size();
  uint num_nodes_target=nodes_target.size();
  uint num_edges_source=edges_source.size();
  uint num_edges_target=edges_target.size();

  //compute number of connections in graph
  thrust::device_vector<uint> indegree_target, indegree_source;
  compute_indegree(indegree_target, edges_target_device, num_nodes_target);
  compute_indegree(indegree_source, edges_source_device, num_nodes_source);
  
  //remove disconnected vertices (optional)
  delete_unreferenced_vertices(nodes_target_device, edges_target_device, indegree_target);
  delete_unreferenced_vertices(nodes_source_device, edges_source_device, indegree_source);
  nodes_source = nodes_source_device;
  edges_source = edges_source_device;
  nodes_target = nodes_target_device;
  edges_target = edges_target_device;

  //compute edge midpoints (optional)
  thrust::device_vector<float3> midpoints_target_device; 
  thrust::device_vector<float3> midpoints_source_device; 
  compute_edge_midpoints(midpoints_target_device, nodes_target_device, edges_target_device);
  compute_edge_midpoints(midpoints_source_device, nodes_source_device, edges_source_device);


  //set up OpenGL data structures
  GLuint vao_src = 0;
  glGenVertexArrays (1, &vao_src);
  glBindVertexArray (vao_src);
  glEnableVertexAttribArray (0);

  GLuint vbo_src = 0;
  glGenBuffers (1, &vbo_src);
  glBindBuffer (GL_ARRAY_BUFFER, vbo_src);
  glBufferData (GL_ARRAY_BUFFER, num_nodes_source * 3 * sizeof (float), thrust::raw_pointer_cast(&nodes_source[0]), GL_DYNAMIC_DRAW);

  GLuint ebo_src = 0; //element buffer object, or index-buffer
  glGenBuffers(1, &ebo_src);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_src);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, num_edges_source * 2 * sizeof(uint), thrust::raw_pointer_cast(&edges_source[0]), GL_DYNAMIC_DRAW);
  glVertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

  GLuint vao_target = 0;
  glGenVertexArrays (1, &vao_target);
  glBindVertexArray (vao_target);
  glEnableVertexAttribArray (0);

  GLuint vbo_target = 0;
  glGenBuffers (1, &vbo_target);
  glBindBuffer (GL_ARRAY_BUFFER, vbo_target);
  glBufferData (GL_ARRAY_BUFFER, num_nodes_target * 3 * sizeof (float), thrust::raw_pointer_cast(&nodes_target[0]), GL_STATIC_DRAW);

  GLuint ebo_target = 0; //element buffer object, or index-buffer
  glGenBuffers(1, &ebo_target);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_target);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, num_edges_target * 2 * sizeof(uint), thrust::raw_pointer_cast(&edges_target[0]), GL_STATIC_DRAW);
  glVertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE, 0, NULL);


  GLuint vertex_array_source = 0;
  GLuint vertex_buffer_source = 0;
  GLuint index_buffer_source = 0;

  GLuint vertex_array_target = 0;
  GLuint vertex_buffer_target = 0;
  GLuint index_buffer_target = 0;


  //load shaders
  std::string vertex_shader_edges_str = readFile("../vs_edges.glsl");
  std::string fragment_shader_edges_str = readFile("../fs_edges.glsl");

  const char *vertex_shader_edges_src = vertex_shader_edges_str.c_str();
  const char *fragment_shader_edges_src = fragment_shader_edges_str.c_str();

  GLuint shader_program = glCreateProgram ();
  GLuint vs_edges = glCreateShader (GL_VERTEX_SHADER);
  GLuint fs_edges = glCreateShader (GL_FRAGMENT_SHADER);


  std::cout << "Compiling OpenGL shaders...";
  glShaderSource (vs_edges, 1, &vertex_shader_edges_src, NULL);
  glCompileShader (vs_edges);
  glShaderSource (fs_edges, 1, &fragment_shader_edges_src, NULL);
  glCompileShader (fs_edges);
  glAttachShader (shader_program, fs_edges);
  glAttachShader (shader_program, vs_edges);
  glLinkProgram (shader_program);
  glDeleteShader(vs_edges);
  glDeleteShader(fs_edges);
  std::cout << "done.\n";


  std::cout << "Initializing RANSAC object...";
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
  std::cout << "done.\n";

  std::thread *t1 = new std::thread(&RANSAC::start, ransac_object);
  glPointSize(3);
  while (!glfwWindowShouldClose(window))
  {

    // Calculate deltatime of current frame
    GLfloat currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    // Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions
     glfwPollEvents();
     do_movement();

     if(reset_key())
     {
        ransac_object->stop();
        t1->join();
        delete t1;
        ransac_object->reset();
        t1 = new std::thread(&RANSAC::start, ransac_object);
     }

     if(hard_reset_key())
     {
        ransac_object->stop();
        t1->join();
        delete t1;
        delete ransac_object;

        ransac_object = new RANSAC( max_iterations,
                              max_iterations_per_batch,
                              error_tolerance_dist,
                              inlier_ratio_to_accept_solution,
                              nodes_source_device,
                              nodes_target_device,
                              midpoints_source_device,
                              midpoints_target_device);
        t1 = new std::thread(&RANSAC::start, ransac_object);
     }



     glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // // Camera/View transformation
    glm::mat4 projection;
    projection = glm::perspective(fov, (GLfloat)w_width/(GLfloat)w_height, 0.1f, 1000.0f);

    glm::mat4 view;
    view = glm::lookAt(cameraPos , cameraPos + cameraFront, cameraDown);

    glm::mat4 model;
    glm::mat4 MVP = projection * view * model;

    glUseProgram (shader_program);

    glUniformMatrix4fv(glGetUniformLocation(shader_program ,"MVP"), 1, GL_FALSE, glm::value_ptr(MVP));

    glBindVertexArray (vao_target);
    set_color(RED, 1.0, shader_program);
    glDrawRangeElements(GL_LINES, 0, num_edges_target, num_edges_target*2, GL_UNSIGNED_INT, NULL);
    glDrawRangeElements(GL_POINTS, 0, num_edges_target, num_edges_target*2, GL_UNSIGNED_INT, NULL);


    uint3 target_model_indices = ransac_object->GetTargetModel();
    float target_model[9] =
    {

     nodes_target[target_model_indices.x].x, nodes_target[target_model_indices.x].y, nodes_target[target_model_indices.x].z, 
     nodes_target[target_model_indices.y].x, nodes_target[target_model_indices.y].y, nodes_target[target_model_indices.y].z, 
     nodes_target[target_model_indices.z].x, nodes_target[target_model_indices.z].y, nodes_target[target_model_indices.z].z 

    };

    GLuint solution_vertex_array_target = 0;
    glGenVertexArrays (1, &solution_vertex_array_target);
    glBindVertexArray (solution_vertex_array_target);
    GLuint model_buffer_object_target = 0;
    glGenBuffers (1, &model_buffer_object_target);
    glBindBuffer (GL_ARRAY_BUFFER, model_buffer_object_target);
    glBufferData (GL_ARRAY_BUFFER, sizeof (target_model), target_model, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);
    glUniform1f(glGetUniformLocation(shader_program ,"R_val"), 1.0f);
    glDrawArrays (GL_LINE_LOOP, 0, 3);


    //GET NEW MODELS HERE FROM RANSAC
    RT warp;
    ransac_object->GetCurrentSolution( warp );

    model[0][0] = warp.R[0].x;  model[1][0] = warp.R[1].x; model[2][0] = warp.R[2].x; model[3][0] = warp.T.x;
    model[0][1] = warp.R[0].y;  model[1][1] = warp.R[1].y; model[2][1] = warp.R[2].y; model[3][1] = warp.T.y;
    model[0][2] = warp.R[0].z;  model[1][2] = warp.R[1].z; model[2][2] = warp.R[2].z; model[3][2] = warp.T.z;
    model[0][3] =           0;  model[1][3] =           0; model[2][3] =           0; model[3][3] =        1;

    MVP = projection * view * model;
    glUniformMatrix4fv(glGetUniformLocation(shader_program ,"MVP"), 1, GL_FALSE, glm::value_ptr(MVP));


    glBindVertexArray (vao_src);
    set_color(GREEN, 1.0, shader_program);
    glDrawRangeElements(GL_LINES, 0, num_edges_source, num_edges_source*2, GL_UNSIGNED_INT, NULL);
    glDrawRangeElements(GL_POINTS, 0, num_edges_source, num_edges_source*2, GL_UNSIGNED_INT, NULL);


    uint3 source_model_indices = ransac_object->GetSourceModel();
    float source_model[9] =
    {
     nodes_source[source_model_indices.x].x, nodes_source[source_model_indices.x].y, nodes_source[source_model_indices.x].z,  
     nodes_source[source_model_indices.y].x, nodes_source[source_model_indices.y].y, nodes_source[source_model_indices.y].z,  
     nodes_source[source_model_indices.z].x, nodes_source[source_model_indices.z].y, nodes_source[source_model_indices.z].z 


    };
    GLuint solution_vertex_array_source = 0;
    glGenVertexArrays (1, &solution_vertex_array_source);
    glBindVertexArray (solution_vertex_array_source);
    GLuint model_buffer_object_source = 0;
    glGenBuffers (1, &model_buffer_object_source);
    glBindBuffer (GL_ARRAY_BUFFER, model_buffer_object_source);
    glBufferData (GL_ARRAY_BUFFER, sizeof (source_model), source_model, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    glUniform1f(glGetUniformLocation(shader_program ,"G_val"), 1.0f);
    glDrawArrays (GL_LINE_LOOP, 0, 3);

    glfwSwapBuffers (window);
    glBindVertexArray(0);

    // if (ransac_object->has_finished())
        // glfwSetWindowShouldClose(window, GL_TRUE);
  
  }

  RT sol;
  ransac_object->GetCurrentSolution(sol);

  std::cout << "best candidate solution was: " << std::endl;
  std::cout << sol.R[0].x << "\t"   << sol.R[1].x << "\t"   << sol.R[2].x << "\t"   << sol.T.x  << std::endl;
  std::cout << sol.R[0].y << "\t"   << sol.R[1].y << "\t"   << sol.R[2].y << "\t"   << sol.T.y  << std::endl;
  std::cout << sol.R[0].z << "\t"   << sol.R[1].z << "\t"   << sol.R[2].z << "\t"   << sol.T.z  << std::endl;
  std::cout << 0          << "\t\t" << 0          << "\t\t" << 0          << "\t\t" << 1        << std::endl;

  std::cout << "with a score of: "<< ransac_object->GetCurrentScore() << std::endl;

  glfwTerminate();
  ransac_object->stop();
  t1->join();
  delete t1;

  return EXIT_SUCCESS;
}
