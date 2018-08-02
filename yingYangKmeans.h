#ifndef YINGYANGKMEANS_H
#define YINGYANGKMEANS_H

#include <stdlib.h>
#include <iostream>
#include <limits>

//vivado HLS specific headers
#include "ap_fixed.h"
#include "ap_int.h"
//#include "fxp_sqrt_top.h"

// dataset dimension specific to certain data file
#define D 28  

// distance type
typedef double double_type;
typedef size_t vertex_num_t;

#define NUM_VERTEX 65554
#define NUM_CLUSTERS 256
#define NUM_C_CLUSTERS 26
#define MAX_C_ITERATION 5
#define MAX_ITERATION 100

extern double_type C_CLUSTERS_changemax[NUM_C_CLUSTERS];

struct cluster{
  cluster(): count(0),count_new(0),count_new_add(0),count_new_sub(0),changed(false),label(0), distChange(0){
    for(size_t i=0;i<D;i++)
    {
      center[i] = 0;
    }
  }
  double_type center[D];
  size_t count;
  size_t count_new; 
  size_t count_new_add;
  size_t count_new_sub;
  double_type distChange;
  bool changed;
  size_t label;
};

struct vertex_data{
	vertex_data(): best_cluster(-1),changed(false), outofdata(false), best_cluster_old(-1), upbound(0), best_distance(0){
		for(size_t i=0;i<D;i++)
			{
			  point[i] = 0;
			}
	}
  double_type point[D];
  size_t best_cluster;
  double_type best_distance;
  bool changed;
  size_t best_cluster_old;  
  double_type lowerbounds[NUM_C_CLUSTERS];
  double_type upbound;
  bool   outofdata;
};

extern cluster CLUSTERS[NUM_CLUSTERS];
extern size_t KMEANS_INITIALIZATION;
extern vertex_data data_points[NUM_VERTEX];


double_type vectordiff_distance(double_type a[D]);
double_type sqr_distance(double_type a[D], double_type b[D]);
void plus_equal_vector(double_type a[D], double_type b[D]);
void kmeans_iteration_initialization(vertex_data& v);
void scale_vector(double_type a[D], float d);
void kmeans_iteration_initialization(vertex_data& v);
void kmeans_iteration_paper(vertex_data& v);
void cluster_center_reducer(cluster cc[NUM_CLUSTERS], vertex_num_t &num_changed);
void changedAssignment(bool changed_vertex[NUM_VERTEX]);
void getClusterChanged(cluster cc[NUM_CLUSTERS], bool changed_vertex[NUM_VERTEX], vertex_num_t& num_changed);
void getassignment(vertex_data cvector[NUM_CLUSTERS], cluster cclusters[NUM_C_CLUSTERS]);
void updatecenter(vertex_data cvector[NUM_CLUSTERS], cluster cclusters[NUM_C_CLUSTERS]);
void C_clusters(cluster clusters[NUM_CLUSTERS]);
void firstkcenters(cluster init_clusters[NUM_CLUSTERS]);
void re_init_clusters(cluster cc[NUM_CLUSTERS]);
void yingYangKmeans_top();

#endif 
