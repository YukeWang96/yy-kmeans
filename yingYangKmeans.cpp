#include "yingYangKmeans.h"

double_type C_CLUSTERS_changemax[NUM_C_CLUSTERS];
cluster CLUSTERS[NUM_CLUSTERS];
size_t KMEANS_INITIALIZATION;
vertex_data data_points[NUM_VERTEX];


// helper function to compute distance between points
double_type sqr_distance(double_type a[D], double_type b[D])
{
#pragma HLS array_partition variable=a complete
#pragma HLS array_partition variable=b complete

  double_type total = 0;

  for (size_t i = 0;i < D; ++i) {
#pragma HLS pipeline
    double_type d = a[i] - b[i];
    total += d * d;
  }
  return sqrt(total);
}


// helper function to compute distance directly from vector difference
double_type vectordiff_distance(double_type a[D])
{
#pragma HLS array_partition variable=a complete

  double_type total = 0;
  for (size_t i = 0;i < D; ++i) {
#pragma HLS pipeline
    double_type val = a[i];
    total += val * val;
  }
  return sqrt(total);
}



// helper function to add two vectors
void plus_equal_vector(double_type a[D], double_type b[D])
{
#pragma HLS array_partition variable=a complete
#pragma HLS array_partition variable=b complete

  for (size_t i = 0;i < D; ++i) {
#pragma HLS unroll
    a[i] += b[i];
  }
}

void copy_equal_vector(double_type a[D], double_type b[D])
{
#pragma HLS array_partition variable=a complete
#pragma HLS array_partition variable=b complete

  for (size_t i = 0;i < D; ++i) {
#pragma HLS unroll
    a[i] = b[i];
  }
}

void scale_vector(double_type a[D], float d)
{
#pragma HLS array_partition variable=a complete
#pragma HLS interface ap_stable port=d

  for (size_t i = 0;i < D; ++i) {
#pragma HLS unroll
    a[i] *= d;
  }
}

// For each point: get lower bounds and upper bound.
void kmeans_iteration_initialization(vertex_data& v)
{
#pragma HLS interface ap_none port=v

  size_t prev_asg = v.best_cluster;
  v.best_cluster_old = v.best_cluster;

  double_type best_distance;
  size_t best_cluster;

  best_cluster = (size_t)(-1);
  best_distance = std::numeric_limits<double>::infinity();

  for (size_t i = 0;i < NUM_C_CLUSTERS; ++i) {
#pragma HLS unroll
    v.lowerbounds[i]=std::numeric_limits<double>::infinity();
  }

  for (size_t i = 0;i < NUM_CLUSTERS; ++i) {
      double_type d = 0.0;
      d = sqr_distance(v.point, CLUSTERS[i].center);
      if (d < best_distance) {
        if(best_cluster != (size_t)(-1))
          v.lowerbounds[CLUSTERS[best_cluster].label]=best_distance;
        best_distance = d;
        best_cluster = i;
      }
      else{
        if(v.lowerbounds[CLUSTERS[i].label] > d)
          v.lowerbounds[CLUSTERS[i].label] = d;
      }
  }

  v.best_cluster = best_cluster;
  v.upbound = best_distance;
  v.outofdata = false;
  v.changed = (prev_asg != v.best_cluster);
}

// Step 3.2 and Step 3.3
// what we called for later iterations
void kmeans_iteration_paper(vertex_data& v)
{
#pragma HLS interface ap_none port=v

  size_t prev_asg = v.best_cluster;
  v.best_cluster_old = v.best_cluster;

  //step1: update all group lowerbounds and upbound.
  double_type templowerbounds[NUM_C_CLUSTERS];
  double_type globallowerbound = std::numeric_limits<double>::infinity();

  for (size_t i = 0;i < NUM_C_CLUSTERS; ++i) {
    templowerbounds[i] = v.lowerbounds[i];
    v.lowerbounds[i] = v.lowerbounds[i] - C_CLUSTERS_changemax[i];
    if(globallowerbound > v.lowerbounds[i]){
      globallowerbound = v.lowerbounds[i];
    }
  }

  if(CLUSTERS[v.best_cluster].distChange > 0){
    v.upbound += CLUSTERS[v.best_cluster].distChange;
    v.outofdata = true;
  }

  //step2: update point assignment
  //Filtering1: this is the "global" filtering.
  if(v.upbound > globallowerbound){
    //Filtering2: otherwise, prepare for group filtering
    bool updateub = false;
    bool updatewholeornot[NUM_C_CLUSTERS];
    //mark groups that did not pass the group filtering.
    for (size_t i = 0;i < NUM_C_CLUSTERS; ++i) {
      updatewholeornot[i] = false;
      if(v.upbound > v.lowerbounds[i]){
        updateub = true;
        updatewholeornot[i] =  true;
        v.lowerbounds[i] =  std::numeric_limits<double>::infinity();
      }
    }

    // recalculate the previous best_cluster
    // update upbound if necessary
    if(v.outofdata && updateub) {
      double_type d = 0.0;
      d = sqr_distance(v.point, CLUSTERS[v.best_cluster].center);
      v.upbound = d;
      v.outofdata = false;
    }

    //another way to iterate over all clusters is group by group.
    for (size_t i = 0;i < NUM_CLUSTERS; ++i) {
      if(i!=prev_asg && updatewholeornot[CLUSTERS[i].label]){
          //Filtering 3: left side is the group second best; right side is the point to center lower bound
          if(v.lowerbounds[CLUSTERS[i].label] > templowerbounds[CLUSTERS[i].label] - CLUSTERS[i].distChange){
            double_type di;
            di = sqr_distance(v.point, CLUSTERS[i].center);
            if(di < v.lowerbounds[CLUSTERS[i].label]){
              if(di < v.upbound){
                v.lowerbounds[CLUSTERS[v.best_cluster].label] = v.upbound;
                v.upbound = di;
                v.outofdata = false;
                v.best_cluster = i;
              }
              else{
                v.lowerbounds[CLUSTERS[i].label] = di;
              }
            }
          }
      }
    }
  }
  v.changed = (prev_asg != v.best_cluster);
}

/*
 * computes new cluster centers
 * Also accumulates a counter counting the number of vertices which
 * assignments changed.
 */
void cluster_center_reducer(cluster cc[NUM_CLUSTERS], vertex_num_t &num_changed)
{
#pragma HLS array_partition variable=cc complete
#pragma HLS interface ap_none port=num_changed

  for(vertex_num_t i=0;i<NUM_VERTEX;i++){
#pragma HLS unroll
	plus_equal_vector(cc[data_points[i].best_cluster].center, data_points[i].point);
    cc[data_points[i].best_cluster].count += 1;
    num_changed += (data_points[i].changed == true);
  }
}

void changedAssignment(bool changed_vertex[NUM_VERTEX])
{
#pragma HLS array_partition variable=changed_vertex complete

  for(vertex_num_t i=0;i<NUM_VERTEX;i++){
#pragma HLS unroll
  if(data_points[i].changed)
    changed_vertex[i] = true;
  else
	changed_vertex[i] = false;
  }
}


void getClusterChanged(cluster cc[NUM_CLUSTERS], bool changed_vertex[NUM_VERTEX], vertex_num_t& num_changed)
{
#pragma HLS array_partition variable=cc complete
#pragma HLS array_partition variable=changed_vertex complete

  for(vertex_num_t i=0;i<NUM_VERTEX;i++){
#pragma HLS unroll
      if(changed_vertex[i])
      {
        cc[data_points[i].best_cluster].count_new += 1;
        cc[data_points[i].best_cluster_old].count_new += 1;
        cc[data_points[i].best_cluster].count_new_add += 1;
        cc[data_points[i].best_cluster_old].count_new_sub += 1;

        vertex_data a = data_points[i];
        plus_equal_vector(cc[data_points[i].best_cluster].center, a.point);

        scale_vector(a.point,-1.0);
        plus_equal_vector(cc[data_points[i].best_cluster_old].center, a.point);

        num_changed += 1;
      }
    }
}

// decide the label/assingment of each clusters to higher level clusters
void getassignment(vertex_data cvector[NUM_CLUSTERS], cluster cclusters[NUM_C_CLUSTERS])
{
#pragma HLS array_partition variable=cvector complete
#pragma HLS array_partition variable=cclusters complete

  double_type di, dbest;
  size_t cbest;

  for (size_t i = 0; i < NUM_CLUSTERS; ++i) {
#pragma HLS unroll
    dbest = std::numeric_limits<double>::infinity();
    cbest = size_t(-1);

    for (size_t j = 0; j < NUM_C_CLUSTERS; ++j) {
      di = sqr_distance(cvector[i].point, cclusters[j].center);
      if(dbest > di){
        dbest = di;
        cbest = j;
      }
    }
    cvector[i].best_cluster = cbest;
  }
}

// decide the label/assingment of each clusters to higher level clusters
void updatecenter(vertex_data cvector[NUM_CLUSTERS], cluster cclusters[NUM_C_CLUSTERS])
{
#pragma HLS array_partition variable=cvector
#pragma HLS array_partition variable=cclusters

  double_type updatecenter[D], emptycenter[D];

  for(size_t i = 0; i < D; i++){
#pragma HLS unroll
    emptycenter[i] = 0;
  }

  for (size_t i = 0;i < NUM_C_CLUSTERS; ++i) {
#pragma HLS unroll
	plus_equal_vector(cclusters[i].center, emptycenter);
    cclusters[i].count = 0;
  }

  for (size_t i = 0;i < NUM_CLUSTERS; ++i) {
#pragma HLS pipeline
    plus_equal_vector(cclusters[cvector[i].best_cluster].center,cvector[i].point);
    cclusters[cvector[i].best_cluster].count  +=1;
  }

  for (size_t i = 0;i < NUM_C_CLUSTERS; ++i) {
#pragma HLS unroll
    float d = cclusters[i].count;
    scale_vector(cclusters[i].center, 1.0/d);
  }
}

// decide the label/assingment of each clusters to higher level clusters
void C_clusters(cluster clusters[NUM_CLUSTERS])
{
#pragma HLS array_partition variable=clusters complete

  // define low-level cluster centers
  vertex_data cvector[NUM_CLUSTERS];

  // define upper-level clusters
  cluster cclusters[NUM_C_CLUSTERS];


  for (size_t i = 0;i < NUM_CLUSTERS; ++i) {
#pragma HLS unroll
	  plus_equal_vector(cvector[i].point, clusters[i].center);
  }

  for (size_t i = 0;i < NUM_C_CLUSTERS; ++i) {
#pragma HLS unroll
	  plus_equal_vector(cclusters[i].center, clusters[i].center);
  }

  for(size_t i = 0; i < MAX_C_ITERATION; ++i ){
    getassignment(cvector,cclusters); // get each low-level clusters upper-level clusters
    updatecenter(cvector,cclusters); // update high-level clusters using low-level clusters
  }

  for (size_t i = 0;i < NUM_CLUSTERS; ++i) {
#pragma HLS unroll
    clusters[i].label = cvector[i].best_cluster;
  }
}

void firstkcenters(cluster init_clusters[NUM_CLUSTERS])
{
#pragma HLS array_partition variable=init_clusters complete

    for(size_t i=0;i<NUM_CLUSTERS;i++){
#pragma HLS unroll
    	plus_equal_vector(init_clusters[i].center, data_points[i].point);
    }
}

void re_init_clusters(cluster cc[NUM_CLUSTERS]){
	for(size_t i=0;i<NUM_CLUSTERS;i++){
#pragma HLS unroll
		cc[i].count=0;
		cc[i].count_new=0;
		cc[i].count_new_add=0;
		cc[i].count_new_sub=0;
		cc[i].changed=false;
		cc[i].label=0;
		cc[i].distChange=0;
		for(size_t j=0;j<D;j++)
			cc[i].center[j] = 0;
	}
}

void yingYangKmeans_top()
{
    firstkcenters(CLUSTERS);

    /* "reset" all clusters*/
    for (size_t i = 0; i < NUM_CLUSTERS; ++i){
#pragma HLS unroll
        CLUSTERS[i].changed = true;
    }

    /* Grouping all lower level clusters*/
    C_clusters(CLUSTERS);
    
    for(vertex_num_t i=0; i<NUM_VERTEX; i++)
#pragma HLS unroll
        kmeans_iteration_initialization(data_points[i]);

    bool clusters_changed = true;
    bool changed_vertices[NUM_VERTEX];
    size_t iteration_count = 0;
    cluster cc[NUM_CLUSTERS];
    vertex_num_t num_changed;

    /*
    main iteration of K-means clustering
    */
   for(size_t iteration_count=0;iteration_count<MAX_ITERATION;iteration_count++)
   {
	   printf("Iteration: %d\t Num changed: %d\n", iteration_count, num_changed);

	   if(!clusters_changed)
            break;

        if (iteration_count==0)  // initial iteration
        {
            num_changed = 0;

            // Gather all vertices of each clusters
            cluster_center_reducer(cc, num_changed);

            for(size_t c=0;c<NUM_CLUSTERS;c++)
            {
                double_type d = cc[c].count;

                if (d > 0) scale_vector(cc[c].center, 1.0 / d);
                if (cc[c].count == 0 && CLUSTERS[c].count > 0) {
                    CLUSTERS[c].count = 0;
                    CLUSTERS[c].changed = false;
                }
                else {
                    size_t label = CLUSTERS[c].label;
                    CLUSTERS[c] = cc[c];
                    CLUSTERS[c].label = label;
                    CLUSTERS[c].changed = true;
                }
            }
            clusters_changed = iteration_count == 0 || num_changed > 0;
        }

        // iteration_count != 0
        else{
            num_changed = 0;
            re_init_clusters(cc);

            //get the changed vertices
            changedAssignment(changed_vertices);

            //get the changed clusters
            getClusterChanged(cc, changed_vertices, num_changed);


            for(size_t i=0;i<NUM_C_CLUSTERS;i++){
#pragma HLS unroll
                C_CLUSTERS_changemax[i]=0;
            }

            for(size_t i=0;i<NUM_CLUSTERS;i++)
            {
                cc[i].count = CLUSTERS[i].count + cc[i].count_new_add - cc[i].count_new_sub;
                double_type d=cc[i].count;
                double_type d1=cc[i].count_new;

                if (d > 0){
                    if (d1 > 0){
                        // compute the distantce of older and current clusters position drift
                        vertex_data center_temp;
                        copy_equal_vector(center_temp.point, CLUSTERS[i].center);
                        scale_vector(CLUSTERS[i].center, CLUSTERS[i].count);
                        plus_equal_vector(cc[i].center, CLUSTERS[i].center);
                        scale_vector(cc[i].center, 1.0 / d);

                        double_type d_update = sqr_distance(cc[i].center, center_temp.point);
                        if(C_CLUSTERS_changemax[CLUSTERS[i].label] < d_update){
                            C_CLUSTERS_changemax[CLUSTERS[i].label] = d_update;
                        }
                        size_t label = CLUSTERS[i].label;
                        CLUSTERS[i] = cc[i];
                        CLUSTERS[i].label = label;
                        CLUSTERS[i].distChange = d_update;
                    }
                    else{
                        CLUSTERS[i].distChange = 0;
                    }
                }

                if(d == 0 && CLUSTERS[i].count > 0){
                    CLUSTERS[i].count = 0;
                    CLUSTERS[i].changed = false;
                    CLUSTERS[i].distChange = 0;
                }
                else if(d1==0){  // nothing to be added to the clusters.
                    CLUSTERS[i].distChange = 0;
                    CLUSTERS[i].changed = false;
                }
            }
            clusters_changed = iteration_count == 0 || num_changed > 0;
        }

      // Step 3.1
      if(iteration_count == 0){
       for(vertex_num_t i=0; i<NUM_VERTEX; i++){
#pragma HLS unroll
            kmeans_iteration_initialization(data_points[i]);
        }
      }
      else{ // Step 3.2 and Step 3.3
        for(vertex_num_t i=0; i<NUM_VERTEX; i++){
#pragma HLS unroll
            kmeans_iteration_paper(data_points[i]);
        }
      }
   }
}
