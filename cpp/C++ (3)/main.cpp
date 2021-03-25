#include <iostream>
#include "Graph.h"
#include "Vertex.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h> 
#include <algorithm>
#include <memory>
using namespace std;

int myrandom(int i) { return std::rand() % i; }

template <typename T> bool PComp(const T const & a, const T const & b)
{
	return stoi(a->name) < stoi(b->name);
}

int totalSteps;
int totalSegments;

void createGraph(Graph& g, string filename) {

	string line;
	ifstream myfile(filename);
	if (myfile.is_open())
	{
		getline(myfile, line);
		getline(myfile, line); //Skip the first two lines
		while (getline(myfile, line))
		{
			istringstream iss(line);
			string v1, v2;
			iss >> v1 >> v1 >> v2;
			g.addEdge(v1, v2);
			//cout << "Substring: " << v1 << "  " << v2 << endl;			
		}
		myfile.close();
	}
	else cout << "Unable to open file";
}

int main() {
	string filename = "graph_10_11_0.txt"; //"expr-data//graph_10_11_0.txt";
	Graph grr;
	createGraph(grr, filename);
	sort(grr.vertices.begin(), grr.vertices.end(), PComp<VertexPtr>);
	// Initialize bitVector
	grr.initializeBitVector();
}