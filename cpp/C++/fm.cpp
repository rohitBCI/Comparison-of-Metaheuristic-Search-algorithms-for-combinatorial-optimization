#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <list>
using namespace std;

// typedef struct gain_bucket_struct
// {
// 	int vertex_id;
// 	struct gainbucket_struct *next, *previous;
// }gainbucket;

// int fm_pass()
// {
// 	initialize_buckets();
// 	while(int v = find_maximum_gain())
// (num_cuts = initial_cuts - gain)
// 	{
// 		remove_from_buckets(v);
// 		partition[v] = !partition[v];

// 		for each (net n connected to v)
// 		{
// 			recalc_gains(n);
// 			if (gain_changed)
// 			{
// 				for each (vertex v2 in n)
// 					update_buckets(v2)
// 			}

// 		}
// 	}
// 	roll_back_to_best_observed();
// 	return best_cut;
// }
 
// int fm()
// {
// 	cut = randomize_partition();

// 	do
// 	{
// 		last_cut = cut;
// 		cut = fm_pass();
// 	} while (cut < last_cut);

// 	return last_cut;
// }

//partition value(0,1) for each vertex
//to calculate initial num_cuts, iterate over vertices in left partition and find connected vertices that are not in the current partition

struct Vertex_ {
    int VertexID;
    int NumOfAdjacentVertices;
    list<std::pair<struct Vertex_*, bool>> Adj;
};
// Bucket[0] = left Bucket
// Bucket[1] = right Bucket
vector<vector<struct Vertex_*>> Bucket[2];

int main()
{
	ifstream infile;
	infile.open("Graph4.txt");
	int vertex;
	string unneeded;
	int numberofadj;
	while(!infile.eof()){

		infile >> vertex >> unneeded >> numberofadj;
		int adjaceny_list[numberofadj];
		for(int i=0; i<numberofadj; ++i){
			infile >> adjaceny_list[i];
		}
		Vertex_ v;
		v.VertexID = vertex;
		v.NumOfAdjacentVertices = numberofadj;
		//v.Adj = adjaceny_list;	
		}
	return 0;
}

	// const int num_vertices = 4;
	// int vertex[4];
 //    // 1-4 to 0-3
  
 //    // for(int i=1; i<=num_vertices/2; ++i) {
 //    //     srand(time(nullptr));
 //    //     //check duplicate
 //    //     int currentNode = rand() % num_vertices;
 //    // }
