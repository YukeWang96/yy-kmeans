#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/tokenizer.hpp>

#include <fstream>
#include <iostream>

#include "yingYangKmeans.h"

// Read a line from a file and creates a vertex
bool vertex_loader(const std::string& line, vertex_data &vtx) {

  if (line.empty()) 
    return true;

  namespace qi = boost::spirit::qi;
  namespace ascii = boost::spirit::ascii;
  namespace phoenix = boost::phoenix;
  std::vector<double> temp_point;

  const bool success = qi::phrase_parse
    (line.begin(), line.end(),
     //  Begin grammar
     (
      (qi::double_[phoenix::push_back(phoenix::ref(temp_point), qi::_1)] % -qi::char_(",") )
     )
     ,
     //  End grammar
     ascii::space);
    
  for(size_t i=0;i<D;i++)
  {
    vtx.point[i]=temp_point[i];
//    std::cout<<vtx.point[i]<<" ";
  }

//  std::cout<<std::endl;

  if (!success) 
    return false;

  vtx.best_cluster = (size_t)(-1);
  vtx.best_distance = std::numeric_limits<double>::infinity();
  vtx.changed = false;

  return true;
}

int main(int argc, char const *argv[])
{
    std::ifstream infile("kegg_shuffled_normal.txt");
    std::string line;
    size_t count = 0;

    while(getline(infile, line))
    {
        vertex_loader(line, data_points[count]);
        count++;
    }

    yingYangKmeans_top();

     // ** Test_1 initial clusters grouping result

//    for (size_t i = 0; i < NUM_CLUSTERS; ++i){
//        std::cout<<CLUSTERS[i].label<<std::endl;
//    }

    // ** Test_2 test initial vertex clustering result.

//        for (size_t i = 0; i < NUM_VERTEX; ++i){
//            std::cout<<data_points[i].best_cluster<<"\t"
//            		<<data_points[i].upbound<<std::endl;
//        }



    return 0;
}
